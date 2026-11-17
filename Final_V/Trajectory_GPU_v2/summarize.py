"""
summarize.py : collect the test results of every noise level into one table and figures.

    python summarize.py        (run_all.py calls it at the end)

Writes, in this folder:
    results_summary.csv       every level x label x model: error sizes (linear labels) or
                              accuracy (binary labels), with the no-skill baselines
    results.png               clean level: predicted vs true for each linear label
                              (CNN, or LightGBM if the CNN was skipped), and accuracy
                              of every model for each binary label
    results_vs_noise.png      only with more than one noise level: error / accuracy vs noise
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import TARGETS, kind

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "sweep")
MODELS = {  # fixed order and colors
    "cnn":           ("CNN", "#2a78d6", "o"),
    "lightgbm":      ("LightGBM", "#eb6834", "s"),
    "random_forest": ("Random Forest", "#1baf7a", "^"),
}
INK, INK_2, GRID, SURFACE = "#2b2b29", "#6b6a63", "#e4e3de", "#fcfcfb"


def levels():
    out = []
    if os.path.isdir(SWEEP):
        for name in os.listdir(SWEEP):
            d = os.path.join(SWEEP, name, "output")
            if name.startswith("sigma_") and os.path.isdir(d):
                out.append((float(name[len("sigma_"):].replace("p", ".")), d))
    return sorted(out)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)


def title(ax, target):
    t, unit = TARGETS[target][1], TARGETS[target][2]
    ax.set_title(t + (f" ({unit})" if unit else ""), loc="left", fontsize=10, color=INK, fontweight="bold")


def figure_clean(out_dir, path):
    preds = [pd.read_csv(os.path.join(out_dir, f)) for f in ("test_predictions_cnn.csv", "test_predictions_forest.csv")
             if os.path.exists(os.path.join(out_dir, f))]
    if not preds:
        return
    pred = pd.concat(preds, ignore_index=True)
    scatter_model = "cnn" if (pred["model"] == "cnn").any() else "lightgbm"
    fig, axs = plt.subplots(3, 3, figsize=(15, 13), facecolor=SURFACE)
    for ax, target in zip(axs.ravel(), TARGETS):
        style(ax)
        title(ax, target)
        sub = pred[pred["target"] == target]
        if kind(target) == "linear":
            g = sub[sub["model"] == scatter_model]
            t, p = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
            lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
            ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1.1, linestyle=(0, (4, 3)))
            ax.scatter(t, p, s=6, color=MODELS[scatter_model][1], alpha=0.35, linewidths=0)
            rmse = np.sqrt(np.mean((p - t) ** 2))
            ax.text(0.03, 0.97, f"{MODELS[scatter_model][0]}  RMS error {rmse:.3g}\nguess the mean {t.std():.3g}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK_2)
            ax.set_xlabel("true", fontsize=8.5, color=INK_2)
            ax.set_ylabel("predicted", fontsize=8.5, color=INK_2)
        else:
            names, accs, colors = [], [], []
            for m, (name, color, _) in MODELS.items():
                g = sub[sub["model"] == m]
                if len(g):
                    names.append(name); colors.append(color)
                    accs.append(np.mean(g["y_pred"] == g["y_true"]))
            y = np.arange(len(names))[::-1]
            ax.barh(y, accs, color=colors, height=0.6)
            share = sub["y_true"].mean() if len(sub) else 0.5
            base = max(share, 1 - share)
            ax.axvline(base, color=INK_2, linewidth=1.1, linestyle=(0, (4, 3)))
            ax.text(base, y.max() + 0.55, f" guess the common answer: {base:.2f}", fontsize=8.5, color=INK_2, va="bottom")
            for yy, acc in zip(y, accs):
                ax.text(acc + 0.01, yy, f"{acc:.2f}", va="center", fontsize=9, color=INK)
            ax.set_yticks(y, names)
            ax.set_xlim(0, 1.12)
            ax.set_ylim(-0.6, y.max() + 1.1)
            ax.set_xlabel("test accuracy", fontsize=8.5, color=INK_2)
            ax.grid(axis="y", visible=False)
    for ax in axs.ravel()[len(TARGETS):]:
        ax.set_visible(False)
    fig.suptitle("Closest-approach labels, clean data: held-out test satellites", x=0.01, ha="left",
                 fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def figure_noise(df, path):
    sigmas = sorted(df["noise_sigma_ms"].unique())
    x = np.arange(len(sigmas))
    fig, axs = plt.subplots(3, 3, figsize=(15, 13), facecolor=SURFACE)
    for ax, target in zip(axs.ravel(), TARGETS):
        style(ax)
        title(ax, target)
        sub = df[df["target"] == target]
        col, base_col = ("rmse", "baseline_rmse") if kind(target) == "linear" else ("accuracy", "baseline_accuracy")
        base = sub[base_col].mean()
        ax.axhline(base, color=INK_2, linewidth=1.1, linestyle=(0, (4, 3)), label="no skill")
        for m, (name, color, marker) in MODELS.items():
            g = sub[sub["model"] == m].set_index("noise_sigma_ms").reindex(sigmas)
            if g[col].notna().any():
                ax.plot(x, g[col], color=color, linewidth=1.5, marker=marker, markersize=6.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, label=name)
        top = max(base, sub[col].max())
        ax.set_ylim(0, 1.05 if col == "accuracy" else top * 1.15)
        ax.set_xticks(x, [f"{s:g}" for s in sigmas])
        ax.set_xlabel("range-rate noise sigma (m/s)", fontsize=8.5, color=INK_2)
        ax.set_ylabel("RMS error" if col == "rmse" else "accuracy", fontsize=8.5, color=INK_2)
    for ax in axs.ravel()[len(TARGETS):]:
        ax.set_visible(False)
    h, n = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, n, loc="upper left", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.005, 0.965))
    fig.suptitle("Closest-approach labels: held-out test error vs measurement noise", x=0.01, ha="left",
                 fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    frames = []
    for sigma, d in levels():
        for f in ("test_results_forest.csv", "test_results_cnn.csv"):
            if os.path.exists(os.path.join(d, f)):
                frames.append(pd.read_csv(os.path.join(d, f)).assign(noise_sigma_ms=sigma))
    if not frames:
        raise SystemExit("No results yet.")
    df = pd.concat(frames, ignore_index=True)
    df["target"] = pd.Categorical(df["target"], list(TARGETS))
    df["model"] = pd.Categorical(df["model"], list(MODELS))
    df = df.sort_values(["noise_sigma_ms", "target", "model"])
    cols = ["noise_sigma_ms", "target", "model", "n_train", "n_test", "rmse", "mae", "median_abs", "p90_abs", "bias",
            "r2", "baseline_rmse", "accuracy", "baseline_accuracy"] + (["best_epoch"] if "best_epoch" in df else [])
    df[cols].to_csv(os.path.join(HERE, "results_summary.csv"), index=False, float_format="%.6g")
    print(f"wrote results_summary.csv ({len(df)} rows)")
    lv = levels()
    figure_clean(lv[0][1], os.path.join(HERE, "results.png"))
    print("wrote results.png")
    if df["noise_sigma_ms"].nunique() > 1:
        figure_noise(df, os.path.join(HERE, "results_vs_noise.png"))
        print("wrote results_vs_noise.png")
