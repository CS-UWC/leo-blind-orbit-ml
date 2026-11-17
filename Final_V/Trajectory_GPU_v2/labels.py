# =============================================================================
# labels.py -- attach the closest-approach labels to the padded training table
# =============================================================================
#
#     python labels.py --ca ../../../Closest_Approach/output/closest_approach.csv
#
# Reads output/training_table_padded.csv (build_dataset.py: rr_c_* and mask_*
# inputs, plus the old Keplerian labels, which are dropped here) and
# closest_approach.csv, and writes output/training_table_ca.csv: sat_idx, the
# same inputs, and the 8 labels in common.TARGETS.
#
# closest_approach.csv must have been computed for the same orbits. This is
# checked: its seed_elevation_deg must equal 90 deg - z0_seed_rad of
# output/random_orbital_parameters.csv for every satellite.
# =============================================================================

import argparse
import os

import numpy as np
import pandas as pd

from common import ALL_TARGETS, TABLE_PATH


def make_labels(ca):
    az = np.radians(ca["azimuth_ca_deg"])
    hd = np.radians(ca["heading_ca_deg"])
    return pd.DataFrame({
        "sat_idx": ca["sat_idx"],
        "label_alt_ca_km": ca["altitude_ca_m"] / 1e3,
        "label_elev_ca_deg": ca["elevation_ca_deg"],
        "label_az_fold_deg": np.degrees(np.arccos(np.clip(np.abs(np.cos(az)), 0.0, 1.0))),
        "label_ang_rate_ca_deg_s": ca["ang_rate_ca_deg_s"],
        "label_radial_vel_ca_mps": ca["radial_vel_ca_mps"],
        "label_eastbound": (np.sin(hd) > 0).astype(int),
        "label_northbound": (np.cos(hd) > 0).astype(int),
        # station on the left of a northbound track / on the right of a southbound one
        "label_passes_east": ((ca["turn_sense"] > 0) == (np.cos(hd) > 0)).astype(int),
    })


def main():
    ap = argparse.ArgumentParser(description="Attach closest-approach labels to the training table.")
    ap.add_argument("--ca", required=True, help="closest_approach.csv")
    a = ap.parse_args()

    table = pd.read_csv("output/training_table_padded.csv")
    ca = pd.read_csv(a.ca)
    if "radial_vel_ca_mps" not in ca.columns:
        raise SystemExit(f"{a.ca} is from an older closest_approach.py (no radial_vel_ca_mps); rerun it.")
    ca = ca[(ca["status"] == "ok") & ca["sat_idx"].isin(table["sat_idx"])]

    prm = pd.read_csv("output/random_orbital_parameters.csv").set_index("sat_idx")
    planned = 90.0 - np.degrees(prm.loc[ca["sat_idx"], "z0_seed_rad"].to_numpy())
    if np.max(np.abs(planned - ca["seed_elevation_deg"].to_numpy())) > 1e-6:
        raise SystemExit(f"{a.ca} was computed for different orbits than output/random_orbital_parameters.csv.")

    feat = [c for c in table.columns if c.startswith("rr_c_") or c.startswith("mask_")]
    out = table[["sat_idx"] + feat].merge(make_labels(ca), on="sat_idx", how="inner")
    out = out[["sat_idx"] + feat + ALL_TARGETS]
    out.to_csv(TABLE_PATH, index=False)
    print(f"{len(out)} satellites ({len(table) - len(out)} without a closest approach dropped), "
          f"{len(feat)} inputs, {len(ALL_TARGETS)} labels -> {TABLE_PATH}")
    for t in ALL_TARGETS:
        v = out[t]
        if set(v.unique()) <= {0, 1}:
            print(f"  {t:>26}: share of 1s {v.mean():.2f}")
        else:
            print(f"  {t:>26}: min {v.min():9.4g}  median {v.median():9.4g}  max {v.max():9.4g}")


if __name__ == "__main__":
    main()
