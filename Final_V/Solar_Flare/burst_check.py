"""
burst_check.py : rebuilds the noise burst data of solar_flare.py, checks it, and plots one pass.

solar_flare.py builds the burst readings and their 1st, 2nd and 3rd differences in memory and
only saves the labels, results and predictions. This script rebuilds that same table with the
functions from solar_flare.py itself (build_table and flare_burst), so nothing is re-implemented,
and checks it against the saved flare_labels.csv.

Run from this folder (Final_V/Solar_Flare), next to solar_flare.py and delay_differences.csv:

    python burst_check.py                    plot the default example pass
    python burst_check.py --sat 106          plot another satellite (must have a burst)
    python burst_check.py --save-features    also save the full rebuilt table (about 30 MB)

Writes (in output/)
  burst_example.png        readings with the burst shaded, and their third differences
                           with and without the burst
  burst_example.csv        the values behind the plot, one row per reading
  flare_features.csv       only with --save-features: the exact table the classifiers were
                           trained and tested on (number of readings, readings, 1st to 3rd
                           differences)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solar_flare import C_LIGHT, DELTA_T, DELAYS, OUT, build_table, flare_burst

CADENCE_S = 30.0


def clean_readings(delays, sat):
    row = delays.set_index("sat_idx").loc[sat]
    n = int(row["num_valid_samples"])
    dd = row[[c for c in delays.columns if c.startswith("dd_")]].to_numpy(float)[:n]
    return dd * C_LIGHT / DELTA_T


def away_from_crossing(rr, b0, b1):
    """Same default choice as the dashboard: a burst that ends or starts 3+ readings from the crossing."""
    i = int(np.argmax(rr >= 0))
    return b1 <= i - 3 or b0 >= i + 3


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sat", type=int, default=None, help="sat_idx of a pass with a burst")
    ap.add_argument("--save-features", action="store_true", help="save the rebuilt feature table")
    a = ap.parse_args()
    if not os.path.exists(DELAYS):
        sys.exit(f"{DELAYS} not found. Run this next to the clean sigma 0 delay_differences.csv.")
    os.makedirs(OUT, exist_ok=True)

    # 1. Rebuild the table exactly as solar_flare.py does, and check it against the saved labels.
    delays = pd.read_csv(DELAYS)
    X, lab = build_table(delays)
    saved = pd.read_csv(os.path.join(OUT, "flare_labels.csv"))
    same = (len(saved) == len(lab)
            and (saved[["sat_idx", "flare_affected", "burst_start", "burst_end"]].to_numpy()
                 == lab[["sat_idx", "flare_affected", "burst_start", "burst_end"]].to_numpy()).all())
    n_hit = int(lab["flare_affected"].sum())
    print(f"Rebuilt {len(lab)} passes, {n_hit} with a burst.")
    print("Matches the saved flare_labels.csv (which passes, burst start and end): "
          + ("yes" if same else "NO"))
    if a.save_features:
        X.to_csv(os.path.join(OUT, "flare_features.csv"), index=False)
        print(f"Saved {OUT}/flare_features.csv ({X.shape[0]} rows, {X.shape[1]} columns)")

    # 2. Pick the example pass: a test pass with a burst.
    split = pd.read_csv(os.path.join(OUT, "train_test_split.csv"))
    test_ids = set(split.loc[split["split"] == "test", "sat_idx"])
    hit = lab[(lab["flare_affected"] == 1) & lab["sat_idx"].isin(test_ids)].reset_index(drop=True)
    if a.sat is None:
        sat = next((int(r.sat_idx) for r in hit.itertuples()
                    if away_from_crossing(clean_readings(delays, int(r.sat_idx)), r.burst_start, r.burst_end)),
                   int(hit.loc[0, "sat_idx"]))
    else:
        sat = a.sat
        if sat not in set(lab.loc[lab["flare_affected"] == 1, "sat_idx"]):
            sys.exit(f"Satellite {sat} has no burst. Pick one with flare_affected = 1 in flare_labels.csv.")

    # 3. The same burst solar_flare.py added to this pass, and the differences.
    rr = clean_readings(delays, sat)
    b0, b1, noise = flare_burst(len(rr), sat)
    rr_b = rr + noise
    # check against the rebuilt feature row (readings with the burst, as the classifiers saw them)
    feat = X.set_index("sat_idx").loc[sat]
    assert np.allclose(feat[[f"rr_{k:03d}" for k in range(len(rr))]].to_numpy(float), rr_b)
    d3_clean, d3_burst = np.diff(rr, 3), np.diff(rr_b, 3)
    minutes = np.arange(len(rr)) * CADENCE_S / 60.0

    out = pd.DataFrame({"reading": np.arange(len(rr)), "minutes": minutes,
                        "range_rate_clean_mps": rr, "range_rate_burst_mps": rr_b,
                        "in_burst": [(b0 <= k < b1) for k in range(len(rr))]})
    out["third_diff_clean_mps"] = np.r_[[np.nan] * 3, d3_clean]
    out["third_diff_burst_mps"] = np.r_[[np.nan] * 3, d3_burst]
    out.to_csv(os.path.join(OUT, "burst_example.csv"), index=False)

    # 4. Plot.
    plt.rcParams.update({"font.size": 9})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax in (ax1, ax2):
        ax.axvspan(minutes[b0], minutes[b1 - 1], color="#FCE3C4", alpha=0.7, lw=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlabel("Minutes from the first reading")
    ax1.plot(minutes, rr_b / 1000, color="#0072B2", lw=1.5, marker="o", ms=3)
    ax1.set_ylabel("Range rate (km/s)")
    ax1.set_title("Readings with the burst (shaded)", fontsize=9)
    ax2.plot(minutes[3:], d3_clean, color="#8A96A8", lw=1.5, ls="--", label="Without the burst")
    ax2.plot(minutes[3:], d3_burst, color="#E69F00", lw=1.5, label="With the burst")
    ax2.set_ylabel("Third difference (m/s)")
    ax2.set_title("Third differences", fontsize=9)
    ax2.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Satellite {sat}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "burst_example.png"), dpi=200)
    plt.close(fig)

    print(f"\nSatellite {sat}: {len(rr)} readings, burst on readings {b0} to {b1 - 1} "
          f"({(b1 - b0) / len(rr):.0%} of the pass)")
    print(f"  largest |third difference| without the burst: {np.abs(d3_clean).max():.1f} m/s")
    print(f"  largest change the burst makes to a third difference: "
          f"{np.abs(d3_burst - d3_clean).max():.1f} m/s")
    pred = pd.read_csv(os.path.join(OUT, "flare_predictions.csv"))
    for r in pred[pred["sat_idx"] == sat].itertuples():
        print(f"  {r.model}: probability of a burst {r.prob_flare:.2f} "
              f"({'flagged' if r.predicted == 1 else 'missed'})")
    print(f"\nSaved {OUT}/burst_example.png and {OUT}/burst_example.csv")
