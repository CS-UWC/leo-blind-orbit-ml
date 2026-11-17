"""
plot_pass.py : figure of one real satellite pass for the report (Section 3.5).

Reads the clean (sigma 0) delay_differences.csv, turns one satellite's delay differences back
into range rate, and plots the readings against time from closest approach. Closest approach
is the zero crossing of the range rate, found the same way as build_dataset.py: the first pair
of readings where the range rate goes from negative to zero or positive, joined by a straight
line.

Run from a folder that contains delay_differences.csv:

    python plot_pass.py
    python plot_pass.py --sat 0          pick another satellite by its sat_idx

Writes (in output/)
  pass_example.pdf   for the LaTeX report (vector, sized for one column)
  pass_example.png   for a quick look
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_LIGHT = 299792458.0   # m/s
DELTA_T = 0.001         # s, the same for every satellite in the clean run
CADENCE_S = 30.0        # s, measurement cadence
OUT = "output"

LINE = "#2a78d6"        # the pass
MARK = "#c0392b"        # the zero crossing
INK, INK2 = "#1a1a1a", "#555555"


def zero_crossing(t, rr):
    """Same rule as build_dataset.py: first pair going from negative to zero or positive."""
    for i in range(len(rr) - 1):
        if rr[i] < 0 <= rr[i + 1]:
            return t[i] + (-rr[i] / (rr[i + 1] - rr[i])) * (t[i + 1] - t[i])
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--delays", default="delay_differences.csv")
    ap.add_argument("--sat", type=int, default=0, help="sat_idx of the satellite to plot (default 0)")
    a = ap.parse_args()
    if not os.path.exists(a.delays):
        raise SystemExit(f"{a.delays} not found. Run this next to the clean sigma 0 delay_differences.csv.")
    os.makedirs(OUT, exist_ok=True)

    d = pd.read_csv(a.delays)
    row = d[d["sat_idx"] == a.sat]
    if row.empty:
        raise SystemExit(f"sat_idx {a.sat} not in {a.delays}")
    row = row.iloc[0]
    n = int(row["num_valid_samples"])
    dd_cols = [c for c in d.columns if c.startswith("dd_")]
    rr = row[dd_cols[:n]].to_numpy(float) * C_LIGHT / DELTA_T          # range rate, m/s
    t = float(row["t_aos_s"]) + np.arange(n) * CADENCE_S               # s, from window start
    t_c = zero_crossing(t, rr)
    if t_c is None:
        raise SystemExit(f"No zero crossing found for satellite {a.sat}.")

    x = (t - t_c) / 60.0        # minutes from closest approach
    y = rr / 1000.0             # km/s

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axvline(0, color=INK2, lw=0.6, ls=":")
    ax.plot(x, y, color=LINE, lw=1.2, marker="o", ms=3)
    ax.plot([0], [0], marker="o", ms=6, mfc="white", mec=MARK, mew=1.5, ls="none")
    ax.annotate("zero crossing\n(closest approach)", xy=(0, 0), xytext=(0.05, 0.78),
                textcoords="axes fraction", fontsize=7, color=MARK, ha="left",
                arrowprops=dict(arrowstyle="->", color=MARK, lw=0.8))
    ax.text(0.05, 0.56, "approaching\n(range rate < 0)", transform=ax.transAxes, fontsize=7, color=INK2)
    ax.text(0.95, 0.30, "receding\n(range rate > 0)", transform=ax.transAxes, fontsize=7, color=INK2,
            ha="right")
    ax.set_xlabel("Time from closest approach (min)", color=INK)
    ax.set_ylabel("Range rate (km/s)", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)
    ax.grid(alpha=0.2, lw=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "pass_example.pdf"))
    fig.savefig(os.path.join(OUT, "pass_example.png"), dpi=300)
    plt.close(fig)

    print(f"Satellite {a.sat}: {n} readings, closest approach {(t_c - t[0]) / 60:.1f} min after the first reading")
    print(f"  range rate from {rr.min():.0f} m/s to {rr.max():.0f} m/s")
    print(f"Saved {OUT}/pass_example.pdf and {OUT}/pass_example.png")
