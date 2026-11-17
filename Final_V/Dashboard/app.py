"""
Dashboard for the final version of the project: ML-Enabled Non-Cooperative Orbit
Estimation and Anomaly Detection with LEO Satellite Signals.

Reads the result files in Final_V directly (nothing is recomputed except the plots):
  Trajectory_GPU_v2/results_summary.csv                  scores for every label, model and noise level
  Trajectory_GPU_v2/sweep/sigma_*/output/                delay differences, test predictions, split
  Solar_Flare/output/flare_*.csv                         noise burst results, predictions and labels
  Solar_Flare/delay_differences.csv                      clean passes used for the burst task

Run from this folder (so the blue theme in .streamlit/config.toml is used):
    streamlit run app.py
"""
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------- constants
BASE = Path(__file__).resolve().parent.parent          # Final_V
TRAJ = BASE / "Trajectory_GPU_v2"
FLARE = BASE / "Solar_Flare"

C_LIGHT = 299_792_458.0      # m/s
DELTA_T = 0.001              # s
CADENCE = 30.0               # s between readings
SIGMAS = [0.0, 0.5, 1.0, 2.0, 4.0]
SIGMA_DIR = {0.0: "sigma_0", 0.5: "sigma_0p5", 1.0: "sigma_1", 2.0: "sigma_2", 4.0: "sigma_4"}

# burst settings, as in solar_flare.py
BURST_SIGMA, BURST_MIN, BURST_MAX, BURST_SEED = 5.0, 0.3, 0.7, 2024

# label -> (name, unit, decimals for model errors, decimals for the guessing error), as in the report
NUMERIC = {
    "label_alt_ca_km": ("Altitude", "km", 1, 1),
    "label_elev_ca_deg": ("Elevation", "deg", 2, 1),
    "label_az_fold_deg": ("Folded azimuth", "deg", 2, 1),
    "label_ang_rate_ca_deg_s": ("Angular rate", "deg/s", 4, 3),
    "label_radial_vel_ca_mps": ("Radial velocity", "m/s", 1, 1),
}
BINARY = {
    "label_eastbound": "Eastbound",
    "label_northbound": "Northbound",
    "label_passes_east": "Passes east",
}
MODELS = {"lightgbm": "LightGBM", "random_forest": "Random forest", "cnn": "CNN"}
MODEL_ORDER = list(MODELS.values())
COLORS = {"LightGBM": "#0072B2", "Random forest": "#009E73", "CNN": "#E69F00"}
GUESS_COLOR = "#8A96A8"
INK, INK_SOFT, GRID = "#14213D", "#5B6B82", "#E3E9F2"

st.set_page_config(page_title="LEO Orbit Estimation Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    h1 {font-size: 1.9rem !important; margin-bottom: 0 !important;}
    .subtitle {color: #5B6B82; margin-top: 0.2rem; margin-bottom: 0.6rem;}
    .note {color: #5B6B82; font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- data
@st.cache_data
def load_summary():
    df = pd.read_csv(TRAJ / "results_summary.csv")
    df["Model"] = df["model"].map(MODELS)
    return df


@st.cache_data
def load_test_ids():
    s = pd.read_csv(TRAJ / "sweep" / "sigma_0" / "output" / "train_test_split.csv")
    return sorted(s.loc[s["split"] == "test", "sat_idx"].tolist())


@st.cache_data
def load_predictions(sigma):
    out = TRAJ / "sweep" / SIGMA_DIR[sigma] / "output"
    df = pd.concat([pd.read_csv(out / "test_predictions_forest.csv"),
                    pd.read_csv(out / "test_predictions_cnn.csv")], ignore_index=True)
    df["Model"] = df["model"].map(MODELS)
    return df


@st.cache_data
def load_delays(path):
    df = pd.read_csv(path)
    return df.set_index("sat_idx")


@st.cache_data
def load_flare():
    res = pd.read_csv(FLARE / "output" / "flare_results.csv")
    pred = pd.read_csv(FLARE / "output" / "flare_predictions.csv")
    lab = pd.read_csv(FLARE / "output" / "flare_labels.csv")
    split = pd.read_csv(FLARE / "output" / "train_test_split.csv")
    return res, pred, lab, split


def readings(delays, sat):
    """Range rate readings (m/s) of one pass and their times (s from the first reading)."""
    row = delays.loc[sat]
    n = int(row["num_valid_samples"])
    dd = row[[f"dd_{k:03d}" for k in range(n)]].to_numpy(float)
    return np.arange(n) * CADENCE, dd * C_LIGHT / DELTA_T


def zero_crossing(t, rr):
    """First reading pair going from negative to zero or positive, joined by a straight line."""
    for i in range(len(rr) - 1):
        if rr[i] < 0 <= rr[i + 1]:
            return t[i] + (-rr[i] / (rr[i + 1] - rr[i])) * (t[i + 1] - t[i])
    return None


def flare_burst(n, sat_idx):
    """Same burst as solar_flare.py for a pass of n readings: (start, end, noise)."""
    rng = np.random.default_rng([BURST_SEED, int(sat_idx)])
    length = int(round(rng.uniform(BURST_MIN, BURST_MAX) * n))
    start = int(rng.integers(0, n - length + 1))
    noise = np.zeros(n)
    noise[start:start + length] = BURST_SIGMA * rng.standard_normal(length)
    return start, start + length, noise


# ----------------------------------------------------------------------------- chart helpers
def style(chart, height=320):
    return (chart.properties(height=height)
            .configure_view(stroke=None)
            .configure_axis(labelColor=INK_SOFT, titleColor=INK, gridColor=GRID,
                            domainColor=GRID, tickColor=GRID, labelFontSize=12, titleFontSize=12,
                            titleFontWeight="normal")
            .configure_legend(labelColor=INK, titleColor=INK, orient="top", title=None,
                              labelFontSize=12))


MODEL_SCALE = alt.Scale(domain=MODEL_ORDER, range=[COLORS[m] for m in MODEL_ORDER])


def bold_best(cols, best="min", decimals=3, guess=None):
    """Row styler: bold the best model (ties at the shown precision are all bold).
    With guess given, bold only if the best model beats it."""
    def row_style(row):
        vals = pd.Series({c: float(half_up(row[c], decimals)) for c in cols})
        target = vals.min() if best == "min" else vals.max()
        ok = True
        if guess is not None:
            g = float(half_up(row[guess], decimals))
            ok = target < g if best == "min" else target > g
        return ["font-weight: 700" if (c in cols and ok and float(half_up(row[c], decimals)) == target)
                else "" for c in row.index]
    return row_style


RIGHT = [{"selector": "td", "props": "text-align: right"}]


def half_up(value, decimals):
    """Round half up, as in the report (85.25% -> 85.3%), not Python's round-half-even."""
    q = Decimal(1).scaleb(-decimals)
    return format(Decimal(repr(round(float(value), 8))).quantize(q, rounding=ROUND_HALF_UP), "f")


def pct(value):
    return half_up(value, 1) + "%"


# ----------------------------------------------------------------------------- info dialog
def pipeline_dot():
    node = 'node [shape=box, style="rounded,filled", fillcolor="#EEF3FA", color="#9DB4D6", ' \
           'fontname="Helvetica", fontsize=10, fontcolor="#14213D", margin="0.18,0.08"];'
    return f"""
    digraph G {{
      rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.3;
      {node}
      edge [color="#5B7FB5", arrowsize=0.7];
      sim   [label="Simulate 10,000 passes\\norbdetpy (Orekit)\\nstation on the equator"];
      dd    [label="Delay difference\\nevery 30 s while the\\nsatellite is visible"];
      rr    [label="Range rate\\n(delay difference × c / δt)"];
      noise [label="Gaussian noise added\\n(σ = 0, 0.5, 1, 2, 4 m/s)", fillcolor="#FFFFFF", style="rounded,dashed,filled"];
      align [label="Align at the zero crossing\\n(closest approach)"];
      pad   [label="Pad and mask\\n94 inputs per pass"];
      mod   [label="LightGBM, random forest,\\n1D CNN\\n8,000 train / 2,000 test"];
      out   [label="8 closest approach\\nlabels", fillcolor="#1F5FAD", fontcolor="white", color="#1F5FAD"];
      burst [label="One 5 m/s burst added\\nto half of the clean passes", fillcolor="#FFFFFF", style="rounded,dashed,filled"];
      diff  [label="Readings plus 1st, 2nd\\nand 3rd differences"];
      cls   [label="LightGBM and random forest\\nclassifiers"];
      out2  [label="Burst: yes or no", fillcolor="#1F5FAD", fontcolor="white", color="#1F5FAD"];
      sim -> dd -> rr;
      rr -> noise -> align -> pad -> mod -> out;
      rr -> burst -> diff -> cls -> out2;
    }}
    """


def diagram(lines, points, texts, size=250):
    """Small top-down map: north up, east to the right."""
    x = alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-1.3, 1.3]))
    y = alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-1.3, 1.3]))
    layers = []
    for df, color, dash in lines:
        layers.append(alt.Chart(df).mark_line(color=color, strokeWidth=2.5 if not dash else 1.2,
                                              strokeDash=dash or [1, 0]).encode(x=x, y=y, order="o:O"))
        if not dash:   # arrow head at the end of a path
            a, b = df.iloc[-2], df.iloc[-1]
            ang = math.degrees(math.atan2(b["x"] - a["x"], b["y"] - a["y"]))
            head = pd.DataFrame({"x": [b["x"]], "y": [b["y"]], "a": [ang]})
            layers.append(alt.Chart(head).mark_point(shape="triangle-up", filled=True, size=110,
                                                     color=color, opacity=1)
                          .encode(x=x, y=y, angle=alt.Angle("a:Q", scale=None)))
    layers.append(alt.Chart(points).mark_point(filled=True, size=90, color=INK, opacity=1).encode(x=x, y=y))
    layers.append(alt.Chart(texts).mark_text(fontSize=11, align="left", color=INK_SOFT, lineBreak="\n")
                  .encode(x=x, y=y, text="t:N"))
    return alt.layer(*layers).properties(width=size, height=size).configure_view(stroke=None)


def seg(x0, y0, x1, y1):
    return pd.DataFrame({"x": [x0, x1], "y": [y0, y1], "o": [0, 1]})


@st.dialog("How it works", width="large")
def show_info():
    st.markdown("#### Pipeline")
    st.graphviz_chart(pipeline_dot(), use_container_width=True)

    st.markdown("#### The direction labels")
    st.markdown("The station sits on the equator. That decides which directions one pass can reveal.")
    c1, c2 = st.columns(2)
    with c1:
        eq = pd.DataFrame({"x": [-1.25, 1.25], "y": [0, 0], "o": [0, 1]})
        lines = [(seg(-1.0, 0.25, 1.0, 0.95), COLORS["LightGBM"], None),
                 (seg(-1.0, -0.25, 1.0, -0.95), COLORS["CNN"], None),
                 (eq, INK_SOFT, [4, 3])]
        pts = pd.DataFrame({"x": [0], "y": [0]})
        txt = pd.DataFrame({"x": [0.08, -1.25, -1.25, -1.25, 0.95],
                            "y": [0.12, -0.12, 0.85, -0.85, 1.22],
                            "t": ["Station", "Equator", "Northbound\n(heading north-east)",
                                  "Its reflection, southbound\n(heading south-east)", "North ↑"]})
        st.altair_chart(diagram(lines, pts, txt, size=280), use_container_width=False)
        st.markdown(
            "**Northbound: cannot be learned.** A pass and its reflection across the equator stay the "
            "same distance from the station at every moment, so they give the same curve."
        )
    with c2:
        ns = pd.DataFrame({"x": [0, 0], "y": [-1.25, 1.25], "o": [0, 1]})
        lines = [(seg(0.55, -1.0, 0.75, 1.0), COLORS["LightGBM"], None),
                 (seg(-0.75, -1.0, -0.55, 1.0), COLORS["CNN"], None),
                 (ns, INK_SOFT, [4, 3])]
        pts = pd.DataFrame({"x": [0], "y": [0]})
        txt = pd.DataFrame({"x": [0.08, 0.05, 0.4, -1.25], "y": [-0.1, 1.15, -1.18, -1.18],
                            "t": ["Station", "North", "Passes east: yes", "Passes east: no"]})
        st.altair_chart(diagram(lines, pts, txt, size=280), use_container_width=False)
        st.markdown(
            "**Passes east: learned weakly.** Passing on either side of the station gives almost the same "
            "curve."
        )
    st.markdown(
        "**Eastbound: learned well.** The station turns east with the Earth, so a satellite moving east "
        "passes it more slowly than one moving west, and the curve shows this."
    )
    st.markdown(
        "**Folded azimuth.** A satellite seen 30° east of north has a reflection seen 30° east of south, "
        "compass directions 30° and 150°. Measuring only the angle from the north-south line gives 30° for "
        "both, so the same curve never has two correct answers."
    )


# ----------------------------------------------------------------------------- header
head, btn = st.columns([6, 1], vertical_alignment="center")
with head:
    st.title("LEO Orbit Estimation Dashboard")
    st.markdown('<p class="subtitle">10,000 simulated passes &nbsp;·&nbsp; 8 closest approach labels '
                '&nbsp;·&nbsp; 3 models &nbsp;·&nbsp; scored on 2,000 test passes</p>', unsafe_allow_html=True)
with btn:
    if st.button("How it works", icon=":material/info:", use_container_width=True):
        show_info()

summary = load_summary()
tab_clean, tab_noise, tab_burst, tab_pass = st.tabs(
    ["Clean data", "Measurement noise", "Noise bursts", "Explore a pass"])


# ----------------------------------------------------------------------------- tab 1: clean data
with tab_clean:
    clean = summary[summary["noise_sigma_ms"] == 0]

    st.subheader("Numeric labels")
    rows_rmse, rows_txt, rows_r2 = [], [], []
    for lab, (name, unit, d, dg) in NUMERIC.items():
        sub = clean[clean["target"] == lab].set_index("Model")
        vals = {m: float(half_up(sub.loc[m, "rmse"], d)) for m in MODEL_ORDER}
        rows_rmse.append({"Label": f"{name} ({unit})", **vals})
        rows_txt.append({"Label": f"{name} ({unit})", "Guessing": half_up(sub["baseline_rmse"].iloc[0], dg),
                         **{m: half_up(v, d) for m, v in vals.items()}})
        rows_r2.append({"Label": name, **{m: sub.loc[m, "r2"] for m in MODEL_ORDER}})
    rmse = pd.DataFrame(rows_rmse).set_index("Label")
    rmse_txt = pd.DataFrame(rows_txt).set_index("Label")
    r2 = pd.DataFrame(rows_r2).set_index("Label")

    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.markdown("**RMSE** (lower is better). Guessing always predicts the average.")
        rule = bold_best(MODEL_ORDER, "min", decimals=6)
        st.table(rmse_txt.style.apply(lambda row: [""] + rule(rmse.loc[row.name]), axis=1)
                 .set_table_styles(RIGHT))
    with c2:
        st.markdown("**R²** (1 is perfect, 0 is no better than guessing).")
        st.table(r2.style.apply(bold_best(MODEL_ORDER, "max", decimals=3), axis=1).format("{:.3f}")
                 .set_table_styles(RIGHT))

    st.markdown("**Predicted against true values** for the 2,000 test passes")
    s1, s2 = st.columns(2)
    lab_name = s1.selectbox("Label", [v[0] for v in NUMERIC.values()], key="pvt_label")
    model_name = s2.selectbox("Model", MODEL_ORDER, index=2, key="pvt_model")
    lab = [k for k, v in NUMERIC.items() if v[0] == lab_name][0]
    unit = NUMERIC[lab][1]
    p = load_predictions(0.0)
    p = p[(p["target"] == lab) & (p["Model"] == model_name)]
    lo, hi = float(min(p["y_true"].min(), p["y_pred"].min())), float(max(p["y_true"].max(), p["y_pred"].max()))
    line = alt.Chart(pd.DataFrame({"a": [lo, hi]})).mark_line(color=GUESS_COLOR, strokeDash=[4, 3]) \
        .encode(x="a:Q", y="a:Q")
    pts = alt.Chart(p).mark_circle(size=14, opacity=0.45, color=COLORS[model_name]).encode(
        x=alt.X("y_true:Q", title=f"True {lab_name.lower()} ({unit})", scale=alt.Scale(zero=False)),
        y=alt.Y("y_pred:Q", title=f"Predicted ({unit})", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("sat_idx:Q", title="Satellite"),
                 alt.Tooltip("y_true:Q", title="True", format=".4g"),
                 alt.Tooltip("y_pred:Q", title="Predicted", format=".4g")])
    st.altair_chart(style(line + pts, 380), use_container_width=True)
    st.markdown('<p class="note">Points on the dashed line are predicted exactly.</p>', unsafe_allow_html=True)

    st.subheader("Yes/no labels")
    rows = []
    for lab, name in BINARY.items():
        sub = clean[clean["target"] == lab].set_index("Model")
        rows.append({"Label": name, "Guessing": sub["baseline_accuracy"].iloc[0] * 100,
                     **{m: sub.loc[m, "accuracy"] * 100 for m in MODEL_ORDER}})
    acc = pd.DataFrame(rows).set_index("Label")
    c1, c2 = st.columns([1, 1.25])
    with c1:
        st.markdown("**Accuracy** (higher is better). Guessing always gives the more common answer.")
        st.table(acc.style.apply(bold_best(MODEL_ORDER, "max", decimals=1, guess="Guessing"), axis=1)
                 .format(pct).set_table_styles(RIGHT))
    with c2:
        long = acc.reset_index().melt(id_vars="Label", value_vars=MODEL_ORDER, var_name="Model",
                                      value_name="Accuracy")
        bars = alt.Chart(long).mark_bar(size=16, cornerRadiusEnd=3).encode(
            x=alt.X("Label:N", title=None, sort=list(BINARY.values()), axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Model:N", sort=MODEL_ORDER),
            y=alt.Y("Accuracy:Q", title="Accuracy (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Model:N", scale=MODEL_SCALE, sort=MODEL_ORDER),
            tooltip=["Label", "Model", alt.Tooltip("Accuracy:Q", format=".1f")])
        g = acc.reset_index()[["Label", "Guessing"]]
        guess = alt.Chart(g).mark_tick(color=INK, thickness=2, size=62).encode(
            x=alt.X("Label:N", sort=list(BINARY.values())), y="Guessing:Q",
            tooltip=["Label", alt.Tooltip("Guessing:Q", format=".1f")])
        st.altair_chart(style(bars + guess, 300), use_container_width=True)
        st.markdown('<p class="note">Black line: guessing.</p>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- tab 2: noise
with tab_noise:
    st.subheader("Effect of measurement noise")
    names = {**{k: v[0] for k, v in NUMERIC.items()}, **BINARY}
    choice = st.selectbox("Label", list(names.values()), key="noise_label")
    lab = [k for k, v in names.items() if v == choice][0]
    sub = summary[summary["target"] == lab].copy()
    is_bin = lab in BINARY

    if is_bin:
        sub["value"] = sub["accuracy"] * 100
        ytitle, fmt = "Accuracy (%)", ".1f"
    else:
        sub["value"] = sub["rmse"]
        ytitle, fmt = f"RMSE ({NUMERIC[lab][1]})", ".4g"

    x = alt.X("noise_sigma_ms:Q", title="Noise level σ (m/s)",
              scale=alt.Scale(domain=[0, 4]), axis=alt.Axis(values=SIGMAS, format=".1~f"))
    lines = alt.Chart(sub).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True)).encode(
        x=x, y=alt.Y("value:Q", title=ytitle, scale=alt.Scale(zero=False)),
        color=alt.Color("Model:N", scale=MODEL_SCALE, sort=MODEL_ORDER),
        tooltip=["Model", alt.Tooltip("noise_sigma_ms:Q", title="σ (m/s)"),
                 alt.Tooltip("value:Q", title=ytitle, format=fmt)])
    chart = lines
    if is_bin:
        gdf = sub.groupby("noise_sigma_ms", as_index=False)["baseline_accuracy"].first()
        gdf["value"] = gdf["baseline_accuracy"] * 100
        chart = chart + alt.Chart(gdf).mark_line(color=GUESS_COLOR, strokeDash=[5, 4], strokeWidth=2).encode(
            x=x, y="value:Q", tooltip=[alt.Tooltip("value:Q", title="Guessing", format=".1f")])
    st.altair_chart(style(chart, 360), use_container_width=True)
    if is_bin:
        st.markdown('<p class="note">Dashed line: guessing.</p>', unsafe_allow_html=True)

    metric = "accuracy (%)" if is_bin else f"RMSE ({NUMERIC[lab][1]})"
    st.markdown(f"**{choice} {metric} at each noise level**")
    table = sub.pivot(index="Model", columns="noise_sigma_ms", values="value").loc[MODEL_ORDER]
    table.columns = [f"σ = {c:g} m/s" for c in table.columns]
    if is_bin:
        st.table(table.style.format(pct).set_table_styles(RIGHT))
    else:
        d = NUMERIC[lab][2]
        st.table(table.style.format(lambda v: half_up(v, d)).set_table_styles(RIGHT))


# ----------------------------------------------------------------------------- tab 3: noise bursts
with tab_burst:
    res, fpred, flab, fsplit = load_flare()
    st.subheader("Noise burst detection")
    names_f = {"LightGBM": "LightGBM", "RandomForest": "Random forest"}
    cols = st.columns(3)
    for col, (_, r) in zip(cols, res.iterrows()):
        with col.container(border=True):
            st.markdown(f"**{names_f[r['model']]}**")
            a, b = st.columns(2)
            a.metric("Accuracy", pct(r["accuracy"] * 100))
            b.metric("F1", f"{r['f1']:.3f}")
            a.metric("Precision", f"{r['precision']:.3f}")
            b.metric("Recall", f"{r['recall']:.3f}")
    with cols[2].container(border=True):
        st.markdown("**Guessing**")
        st.metric("Accuracy", pct(res["baseline_accuracy"].iloc[0] * 100))
        st.markdown('<p class="note">Always answers "burst", the more common answer among the '
                    'test passes.</p>', unsafe_allow_html=True)

    st.markdown("**Results on the 2,000 test passes**")
    cm_cols = st.columns(2)
    for col, (_, r) in zip(cm_cols, res.iterrows()):
        cm = pd.DataFrame({"Flagged": [int(r["tp"]), int(r["fp"])],
                           "Not flagged": [int(r["fn"]), int(r["tn"])]},
                          index=[f"Had a burst ({int(r['tp'] + r['fn']):,})",
                                 f"Clean ({int(r['fp'] + r['tn']):,})"])
        with col:
            st.markdown(names_f[r["model"]])
            st.table(cm)

    st.subheader("How a burst shows up")
    delays_clean = load_delays(FLARE / "delay_differences.csv")
    test_ids = set(fsplit.loc[fsplit["split"] == "test", "sat_idx"])
    hit = flab[(flab["flare_affected"] == 1) & flab["sat_idx"].isin(test_ids)]["sat_idx"].tolist()
    def away_from_crossing(s):
        t_, rr_ = readings(delays_clean, s)
        i = int(np.argmax(rr_ >= 0))
        r = flab.set_index("sat_idx").loc[s]
        return r["burst_end"] <= i - 3 or r["burst_start"] >= i + 3
    default = next((i for i, s in enumerate(hit) if away_from_crossing(s)), 0)
    sat = st.selectbox("Test pass with a burst", hit, index=default, key="burst_sat",
                       format_func=lambda s: f"Satellite {s}")
    t, rr = readings(delays_clean, sat)
    b0, b1, noise = flare_burst(len(rr), sat)
    rr_b = rr + noise
    minutes = t / 60.0
    zone = pd.DataFrame({"start": [minutes[b0]], "end": [minutes[b1 - 1]]})
    shade = alt.Chart(zone).mark_rect(color="#FCE3C4", opacity=0.6).encode(x="start:Q", x2="end:Q")

    d3_clean, d3_burst = np.diff(rr, 3), np.diff(rr_b, 3)
    c1, c2 = st.columns(2)
    with c1:
        df = pd.DataFrame({"min": minutes, "rr": rr_b / 1000})
        ch = alt.Chart(df).mark_line(color=COLORS["LightGBM"], strokeWidth=2,
                                     point=alt.OverlayMarkDef(size=18, filled=True)).encode(
            x=alt.X("min:Q", title="Minutes from the first reading"),
            y=alt.Y("rr:Q", title="Range rate (km/s)"))
        st.markdown("Readings with the burst (shaded)")
        st.altair_chart(style(shade + ch, 280), use_container_width=True)
    with c2:
        df = pd.concat([
            pd.DataFrame({"min": minutes[3:], "d3": d3_burst, "Pass": "With the burst"}),
            pd.DataFrame({"min": minutes[3:], "d3": d3_clean, "Pass": "Without the burst"})])
        ch = alt.Chart(df).mark_line(strokeWidth=2).encode(
            x=alt.X("min:Q", title="Minutes from the first reading"),
            y=alt.Y("d3:Q", title="Third difference (m/s)"),
            color=alt.Color("Pass:N", scale=alt.Scale(domain=["With the burst", "Without the burst"],
                                                      range=[COLORS["CNN"], GUESS_COLOR]),
                            legend=alt.Legend(title=None)),
            strokeDash=alt.StrokeDash("Pass:N", scale=alt.Scale(domain=["With the burst", "Without the burst"],
                                                                range=[[1, 0], [5, 3]]),
                                      legend=alt.Legend(title=None)))
        st.markdown("Third differences")
        st.altair_chart(style(shade + ch, 280), use_container_width=True)

    probs = fpred[fpred["sat_idx"] == sat]
    if len(probs):
        parts = [f"{names_f[r['model']]}: {r['prob_flare'] * 100:.0f}% "
                 f"({'flagged' if r['predicted'] == 1 else 'missed'})" for _, r in probs.iterrows()]
        st.markdown('<p class="note">Predicted probability of a burst for this pass. '
                    + " &nbsp;·&nbsp; ".join(parts) + "</p>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- tab 4: explore a pass
with tab_pass:
    ids = load_test_ids()
    c1, c2 = st.columns([1, 2])
    sat = c1.selectbox("Test pass", ids, key="pass_sat", format_func=lambda s: f"Satellite {s}")
    sigma = c2.select_slider("Noise level σ (m/s)", options=SIGMAS, value=0.0, key="pass_sigma",
                             format_func=lambda v: f"{v:g}")

    delays = load_delays(TRAJ / "sweep" / SIGMA_DIR[sigma] / "output" / "delay_differences.csv")
    t, rr = readings(delays, sat)
    tc = zero_crossing(t, rr)
    xmin = (t - tc) / 60.0 if tc is not None else t / 60.0
    df = pd.DataFrame({"min": xmin, "rr": rr / 1000})
    zero = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color=GRID, strokeWidth=1.5)
    curve = alt.Chart(df).mark_line(color=COLORS["LightGBM"], strokeWidth=2,
                                    point=alt.OverlayMarkDef(size=24, filled=True)).encode(
        x=alt.X("min:Q", title="Minutes from closest approach"),
        y=alt.Y("rr:Q", title="Range rate (km/s)"),
        tooltip=[alt.Tooltip("min:Q", title="Minutes", format=".2f"),
                 alt.Tooltip("rr:Q", title="km/s", format=".3f")])
    ca = alt.Chart(pd.DataFrame({"x": [0], "y": [0]})).mark_point(
        shape="circle", size=120, filled=False, color=COLORS["CNN"], strokeWidth=2).encode(x="x:Q", y="y:Q")
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.altair_chart(style(zero.encode(y="v:Q") + curve + ca, 340), use_container_width=True)
        st.markdown(f'<p class="note">{len(rr)} readings, 30 s apart. Negative while approaching, '
                    f'positive while moving away. The circle marks closest approach.</p>',
                    unsafe_allow_html=True)
    with c2:
        p = load_predictions(sigma)
        p = p[p["sat_idx"] == sat]
        rows = []
        for lab, (name, unit, d, _) in NUMERIC.items():
            q = p[p["target"] == lab].set_index("Model")
            rows.append({"Label": f"{name} ({unit})", "True": half_up(q["y_true"].iloc[0], d),
                         **{m: half_up(q.loc[m, "y_pred"], d) for m in MODEL_ORDER}})
        for lab, name in BINARY.items():
            q = p[p["target"] == lab].set_index("Model")
            yn = lambda v: "Yes" if v >= 0.5 else "No"
            rows.append({"Label": name, "True": yn(q["y_true"].iloc[0]),
                         **{m: yn(q.loc[m, "y_pred"]) for m in MODEL_ORDER}})
        st.markdown("**True values and predictions**")
        st.table(pd.DataFrame(rows).set_index("Label"))
