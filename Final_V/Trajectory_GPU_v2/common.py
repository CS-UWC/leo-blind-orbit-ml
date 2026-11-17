# =============================================================================
# common.py -- labels, table loading and scoring shared by forests.py and cnn.py
# =============================================================================
#
# The labels describe the satellite at its CLOSEST APPROACH to the station
# (computed by ../Closest_Approach/closest_approach.py from the simulated
# trajectory). Together they fix the satellite's Earth-fixed position and
# velocity at that instant, hence its trajectory relative to the Earth:
#
#   continuous  altitude, elevation, FOLDED azimuth (below), angular rate
#               (x range = speed), radial velocity (climb rate; the eccentricity
#               signal). At closest approach the velocity is perpendicular to the
#               line of sight.
#   flags       The continuous labels leave 8 candidate passes: 4 mirror-image
#               positions (NE/SE/SW/NW of the station) x 2 directions of motion.
#               One flag per ambiguity:
#                 eastbound    prograde vs retrograde. Earth's rotation makes
#                              this visible in the Doppler curve.        8 -> 4
#                 northbound   a pass and its reflection through the equator
#                              give EXACTLY the same curve (the station, Earth's
#                              rotation and its gravity are all symmetric about
#                              the equator): a coin flip from the curve;
#                              known only from outside information.     4 -> 2
#                 passes_east  which side of its track the satellite passes the
#                              station: east of the station for a northbound
#                              pass means the station is on its left, for a
#                              southbound pass on its right. The reflection
#                              leaves it unchanged; Earth's rotation breaks it
#                              only weakly.                              2 -> 1
# ../Closest_Approach/reconstruct_check_v2.py confirms the 8 -> 4 -> 2 -> 1 chain
# for all 10,000 clean satellites, and rebuilds the true state from these labels
# to rounding error (204 satellites within ~1 deg of a compass axis keep two
# nearly identical candidates).
#
# Folded azimuth = arccos(|cos(azimuth)|): the angle between the direction to
# the satellite and the north-south line. 0 = due north or south of the station,
# 90 = due east or west. It is the same for all four mirror images.
# =============================================================================

import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# label column -> (kind, title, unit)
TARGETS = {
    "label_alt_ca_km":         ("linear", "Altitude at closest approach", "km"),
    "label_elev_ca_deg":       ("linear", "Elevation at closest approach", "deg"),
    "label_az_fold_deg":       ("linear", "Folded azimuth (0 = N/S, 90 = E/W)", "deg"),
    "label_ang_rate_ca_deg_s": ("linear", "Angular rate at closest approach", "deg/s"),
    "label_radial_vel_ca_mps": ("linear", "Radial velocity at closest approach", "m/s"),
    "label_eastbound":         ("binary", "Moving east (1) / west (0)", ""),
    "label_northbound":        ("binary", "Moving north (1) / south (0)", ""),
    "label_passes_east":       ("binary", "Passes east of station (1) / west (0)", ""),
}
ALL_TARGETS = list(TARGETS)
TABLE_PATH = "output/training_table_ca.csv"


def kind(target):
    return TARGETS[target][0]


def load_table(path=TABLE_PATH):
    if not os.path.exists(path):
        raise SystemExit(f"Training table not found: {path}\nRun labels.py first.")
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c.startswith("rr_c_") or c.startswith("mask_")]
    return df, feat_cols


def score(y_true, y_pred, target):
    """Linear labels: RMS, mean, median and 90th-percentile absolute error, bias,
    R2, and the RMS error of always guessing the mean. Binary labels: accuracy,
    and the accuracy of always guessing the more common answer. Keys that do not
    apply are NaN, so every row of a results table has the same columns."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = dict.fromkeys(["rmse", "mae", "median_abs", "p90_abs", "bias", "r2", "baseline_rmse",
                         "accuracy", "baseline_accuracy"], float("nan"))
    if kind(target) == "binary":
        share = y_true.mean()
        out.update(accuracy=float(np.mean(y_pred == y_true)), baseline_accuracy=float(max(share, 1 - share)))
        return out
    err = y_pred - y_true
    a = np.abs(err)
    out.update(rmse=float(np.sqrt(np.mean(err ** 2))), mae=float(a.mean()), median_abs=float(np.median(a)),
               p90_abs=float(np.percentile(a, 90)), bias=float(err.mean()), r2=float(r2_score(y_true, y_pred)),
               baseline_rmse=float(y_true.std()))
    return out


def print_results(results):
    print(f"{'target':>26} {'model':>14} {'RMSE':>10} {'median':>9} {'p90':>9} {'R2':>7} "
          f"{'guess':>9} | {'accuracy':>8} {'guess':>6}")
    print("-" * 110)
    for r in results:
        if kind(r["target"]) == "binary":
            print(f"{r['target']:>26} {r['model']:>14} {'':>10} {'':>9} {'':>9} {'':>7} {'':>9} | "
                  f"{r['accuracy']:>8.3f} {r['baseline_accuracy']:>6.3f}")
        else:
            print(f"{r['target']:>26} {r['model']:>14} {r['rmse']:>10.4g} {r['median_abs']:>9.4g} "
                  f"{r['p90_abs']:>9.4g} {r['r2']:>7.3f} {r['baseline_rmse']:>9.4g} |")
