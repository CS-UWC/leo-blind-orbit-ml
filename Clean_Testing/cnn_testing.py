# =============================================================================
# cnn_testing.py -- TESTING half: scores what cnn_trainer_v3.py saved
# =============================================================================
#
# PURPOSE
# -------
# Never trains. Loads the 6 saved CNNs (output/models/cnn/), predicts the
# held-out test satellites, and scores them with the same score() the trees
# use. Also scores the saved 5-fold CV predictions
# (output/cv_predictions_cnn_v3.csv) per fold and reports the mean and std
# across the 5 folds. Then draws the predicted-vs-true figure.
#
# Needs cnn_trainer_v3.py to have been run first. After that this file can be
# rerun as often as needed (new metric, new plot) without retraining.
#
# WHAT THIS FILE WRITES (all in output/)
# --------------------------------------
# 1. test_predictions_cnn_v3.csv   per-satellite held-out test predictions
# 2. test_results_cnn_v3.csv       held-out test scores: model, target, n,
#                                  n_train, r2, rmse, circ_corr
# 3. cv_results_cnn_v3.csv         CV scores and their _std across the 5 folds
# 4. predicted_vs_true_cnn.png     the scatter figure
# =============================================================================

import os

import numpy as np
import pandas as pd

from Clean_Testing.splits import make_or_load_split
from Clean_Testing.common import ALL_TARGETS, load_table, score, _agg_stat
from Clean_Testing.cnn_trainer_v3 import MODEL_NAME, model_path, load_model, to_input
import Clean_Testing.plot_predictions as plot_predictions


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


def score_cv(cv_df, df, target):
    """Score the saved out-of-fold predictions PER FOLD, then mean +/- std over
    the folds (not pooled into one number)."""
    sub = cv_df[(cv_df["target"] == target) & (cv_df["model"] == MODEL_NAME)]
    fold_scores = []
    for fold in sorted(sub["fold"].unique()):
        f = sub[sub["fold"] == fold]
        fold_scores.append(score(f["y_true"].to_numpy(), f["y_pred"].to_numpy(), target))

    m = {}
    for key in ("r2", "rmse", "circ_corr"):
        mean_v, std_v = _agg_stat([fs[key] for fs in fold_scores])
        m[key], m[f"{key}_std"] = mean_v, std_v
    m.update({
        "target": target, "model_name": MODEL_NAME,
        "n": len(sub), "n_dropped_nan": len(df) - len(df.dropna(subset=[target])),
    })
    return m


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

    # --- 5-fold CV, scored from the saved out-of-fold predictions ---
    cv_path = "output/cv_predictions_cnn_v3.csv"
    if not os.path.exists(cv_path):
        raise SystemExit(f"CV predictions not found: {cv_path}\nRun cnn_trainer_v3.py first.")
    cv_df = pd.read_csv(cv_path, float_precision="round_trip")   # exact floats, so scores match the in-memory ones

    print("\n" + "=" * 90)
    print("5-FOLD CROSS-VALIDATION ON ALL SATELLITES (same folds as the trees)")
    print("=" * 90)

    cv_results = [score_cv(cv_df, df, target) for target in ALL_TARGETS]

    print(f"\n{'target':>28} {'n':>5} {'R2 (+/-std)':>16} {'RMSE (+/-std)':>24} "
          f"{'circ_corr (+/-std)':>20}")
    print("-" * 100)
    for r in cv_results:
        r2_str = (f"{r['r2']:.3f}+/-{r['r2_std']:.3f}"
                  if not np.isnan(r["r2"]) else "  --  ")
        corr_str = (f"{r['circ_corr']:.3f}+/-{r['circ_corr_std']:.3f}"
                     if not np.isnan(r["circ_corr"]) else "  --  ")
        rmse_str = f"{r['rmse']:.4f}+/-{r['rmse_std']:.4f}"
        print(f"{r['target']:>28} {r['n']:>5} {r2_str:>16} {rmse_str:>24} "
              f"{corr_str:>20}")

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

    cv_out = "output/cv_results_cnn_v3.csv"
    pd.DataFrame([{
        "model": MODEL_NAME, "target": r["target"], "n": r["n"],
        "r2": r["r2"], "rmse": r["rmse"], "circ_corr": r["circ_corr"],
        "r2_std": r["r2_std"], "rmse_std": r["rmse_std"],
        "circ_corr_std": r["circ_corr_std"],
    } for r in cv_results]).to_csv(cv_out, index=False)

    print(f"\nSaved held-out predictions to {pred_path}")
    print(f"Saved held-out test results to {test_path}")
    print(f"Saved CV results to {cv_out}")

    plot_predictions.make_figure(
        predictions_path=pred_path, results_path=test_path,
        out_path="output/predicted_vs_true_cnn.png",
    )
