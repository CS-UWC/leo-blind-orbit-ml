# =============================================================================
# forest_testing.py -- TESTING half: scores what forest_trainer_v3.py saved
# =============================================================================
#
# PURPOSE
# -------
# Never trains. Loads the 12 saved models (output/models/forest/), predicts the
# held-out test satellites, and scores them. There is no cross-validation in this
# folder (Clean_Testing has it).
#
# Needs forest_trainer_v3.py to have been run first. After that this file can be
# rerun as often as needed (new metric, new table) without retraining.
#
# WHAT THIS FILE WRITES (all in output/)
# --------------------------------------
# 1. test_predictions_v3.csv   per-satellite held-out test predictions
#                              (sat_idx, target, model, y_true, y_pred);
#                              plot_predictions.py reads this.
# 2. test_results_v3.csv       held-out test scores: model, target, n, n_train,
#                              r2, rmse, circ_corr (r2 blank for angles,
#                              circ_corr blank for linear labels).
# =============================================================================

import os

import numpy as np
import pandas as pd
import joblib

from splits import make_or_load_split
from common import ALL_TARGETS, load_table, score
from forest_trainer_v3 import MODELS, model_path


def test_one(df, feat_cols, target, model_name, train_ids, test_ids):
    """Load one saved model, predict the held-out test satellites, score them.
    Returns (metrics_dict, predictions_df)."""
    path = model_path(model_name, target)
    if not os.path.exists(path):
        raise SystemExit(f"Saved model not found: {path}\nRun forest_trainer_v3.py first.")
    model = joblib.load(path)

    sub = df.dropna(subset=[target])
    n_dropped = len(df) - len(sub)

    train_mask = sub["sat_idx"].isin(train_ids)
    test_mask = sub["sat_idx"].isin(test_ids)
    X_test, y_test = sub.loc[test_mask, feat_cols], sub.loc[test_mask, target]
    test_sat_idx = sub.loc[test_mask, "sat_idx"].to_numpy()

    pred_test = model.predict(X_test)

    m = score(y_test.to_numpy(), pred_test, target)
    m.update({
        "target": target, "model_name": model_name,
        "n_train": int(train_mask.sum()), "n_test": len(X_test), "n_dropped_nan": n_dropped,
    })
    predictions = pd.DataFrame({
        "sat_idx": test_sat_idx,
        "target": target,
        "model": model_name,
        "y_true": y_test.to_numpy(),
        "y_pred": pred_test,
    })
    return m, predictions


if __name__ == "__main__":

    df, feat_cols = load_table()
    split_df = make_or_load_split(df["sat_idx"])
    train_ids = set(split_df.loc[split_df["split"] == "train", "sat_idx"])
    test_ids = set(split_df.loc[split_df["split"] == "test", "sat_idx"])
    print(f"Loaded {len(df)} rows. Split: {len(train_ids)} train / {len(test_ids)} test satellites "
          f"(output/train_test_split.csv)\n")

    # --- held-out test set, once per target x model ---
    test_results = []
    test_predictions = []
    for target in ALL_TARGETS:
        if target not in df.columns:
            print(f"  [skip] {target} not in table")
            continue
        for model_name in MODELS:
            m, preds = test_one(df, feat_cols, target, model_name, train_ids, test_ids)
            test_results.append(m)
            test_predictions.append(preds)

    os.makedirs("output", exist_ok=True)
    predictions_path = "output/test_predictions_v3.csv"
    pd.concat(test_predictions, ignore_index=True).to_csv(predictions_path, index=False)
    print(f"Saved per-satellite held-out test predictions to {predictions_path} "
          f"(input for plot_predictions.py)\n")

    print(f"{'target':>28} {'model':>14} {'n_train':>8} {'n_test':>7} "
          f"{'R2':>8} {'RMSE':>12} {'circ_corr':>11}")
    print("-" * 112)
    for target in ALL_TARGETS:
        rows = [r for r in test_results if r["target"] == target]
        for r in rows:
            r2_str = f"{r['r2']:.3f}" if not np.isnan(r["r2"]) else "  --  "
            corr_str = f"{r['circ_corr']:.3f}" if not np.isnan(r["circ_corr"]) else "  --  "
            print(f"{r['target']:>28} {r['model_name']:>14} {r['n_train']:>8} {r['n_test']:>7} "
                  f"{r2_str:>8} {r['rmse']:>12.4f} {corr_str:>11}")
        if rows and rows[0]["n_dropped_nan"]:
            print(f"    [note] {rows[0]['n_dropped_nan']} rows dropped (NaN in this target)")
        print()

    print("Linear labels (sma, ecc, inc): R2 near 0 or negative means no better than the mean.")
    print("Circular labels (raan, argp, anom): R2 is left blank (not meaningful on wrapped")
    print("angles); circ_corr is the circular correlation (0 = no relationship, 1 = perfect,")
    print("negative = inverse relationship).\n")

    test_path = "output/test_results_v3.csv"
    pd.DataFrame([{
        "model": r["model_name"], "target": r["target"],
        "n": r["n_test"], "n_train": r["n_train"],
        "r2": r["r2"], "rmse": r["rmse"], "circ_corr": r["circ_corr"],
    } for r in test_results]).to_csv(test_path, index=False)
    print(f"Saved held-out test results to {test_path}\n")
