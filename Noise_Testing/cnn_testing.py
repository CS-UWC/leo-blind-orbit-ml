# =============================================================================
# cnn_testing.py -- TESTING half: scores what cnn_trainer_v3.py saved
# =============================================================================
#
# PURPOSE
# -------
# Never trains. Loads the 6 saved CNNs (output/models/cnn/), predicts the
# held-out test satellites, and scores them with the same score() the trees
# use. Then draws the predicted-vs-true figure. There is no cross-validation
# in this folder (Clean_Testing has it).
#
# Needs cnn_trainer_v3.py to have been run first. After that this file can be
# rerun as often as needed (new metric, new plot) without retraining.
#
# WHAT THIS FILE WRITES (all in output/)
# --------------------------------------
# 1. test_predictions_cnn_v3.csv   per-satellite held-out test predictions
# 2. test_results_cnn_v3.csv       held-out test scores: model, target, n,
#                                  n_train, r2, rmse, circ_corr
# 3. predicted_vs_true_cnn.png     the scatter figure
# =============================================================================

import os

import numpy as np
import pandas as pd

from splits import make_or_load_split
from common import ALL_TARGETS, load_table, score
from cnn_trainer_v3 import MODEL_NAME, model_path, load_model, to_input
import plot_predictions as plot_predictions


def test_one(df, target, train_ids, test_ids):
    """Load one saved CNN, predict the held-out test satellites, score them.
    Returns (metrics_dict, predictions_df)."""
    path = model_path(target)
    if not os.path.exists(path):
        raise SystemExit(f"Saved model not found: {path}\nRun cnn_trainer_v3.py first.")
    model = load_model(path)

    sub = df.dropna(subset=[target])
    n_dropped = len(df) - len(sub)
    train = sub[sub["sat_idx"].isin(train_ids)]
    test = sub[sub["sat_idx"].isin(test_ids)]

    pred = model.predict(to_input(test))

    m = score(test[target].to_numpy(), pred, target)
    m.update({
        "target": target, "model_name": MODEL_NAME,
        "n_train": len(train), "n_test": len(test), "n_dropped_nan": n_dropped,
        "best_epoch": model.best_epoch_,
    })
    predictions = pd.DataFrame({
        "sat_idx": test["sat_idx"].to_numpy(), "target": target, "model": MODEL_NAME,
        "y_true": test[target].to_numpy(), "y_pred": pred,
    })
    return m, predictions


if __name__ == "__main__":

    df, _ = load_table()
    split_df = make_or_load_split(df["sat_idx"])
    train_ids = set(split_df.loc[split_df["split"] == "train", "sat_idx"])
    test_ids = set(split_df.loc[split_df["split"] == "test", "sat_idx"])
    print(f"Loaded {len(df)} rows. Split: {len(train_ids)} train / {len(test_ids)} test satellites "
          f"(output/train_test_split.csv)\n")

    test_results, test_predictions = [], []
    for target in ALL_TARGETS:
        m, preds = test_one(df, target, train_ids, test_ids)
        test_results.append(m)
        test_predictions.append(preds)

    print(f"{'target':>28} {'n_train':>8} {'n_test':>7} {'epochs':>7} "
          f"{'R2':>8} {'RMSE':>12} {'circ_corr':>11}")
    print("-" * 90)
    for r in test_results:
        r2_str = f"{r['r2']:.3f}" if not np.isnan(r["r2"]) else "  --  "
        corr_str = f"{r['circ_corr']:.3f}" if not np.isnan(r["circ_corr"]) else "  --  "
        print(f"{r['target']:>28} {r['n_train']:>8} {r['n_test']:>7} {r['best_epoch']:>7} "
              f"{r2_str:>8} {r['rmse']:>12.4f} {corr_str:>11}")

    # --- save ---
    os.makedirs("output", exist_ok=True)

    pred_path = "output/test_predictions_cnn_v3.csv"
    pd.concat(test_predictions, ignore_index=True).to_csv(pred_path, index=False)

    test_path = "output/test_results_cnn_v3.csv"
    pd.DataFrame([{
        "model": MODEL_NAME, "target": r["target"],
        "n": r["n_test"], "n_train": r["n_train"],
        "r2": r["r2"], "rmse": r["rmse"], "circ_corr": r["circ_corr"],
    } for r in test_results]).to_csv(test_path, index=False)

    print(f"\nSaved held-out predictions to {pred_path}")
    print(f"Saved held-out test results to {test_path}")

    plot_predictions.make_figure(
        predictions_path=pred_path, results_path=test_path,
        out_path="output/predicted_vs_true_cnn.png",
    )
