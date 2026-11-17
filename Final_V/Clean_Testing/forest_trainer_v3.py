# =============================================================================
# forest_trainer_v3.py -- TRAINING half: fits LightGBM + Random Forest on the
#                          padded, mask-aware range-rate vectors
# =============================================================================
#
# PURPOSE
# -------
# Reads output/training_table_padded.csv and trains a separate LightGBM
# regressor and a separate Random Forest regressor for each of the six
# labels, with no hyperparameter tuning. This file only FITS. It does not
# score anything: scoring is in forest_testing.py, which loads what this file
# saves. That way a new metric, a new table or a changed plot never needs the
# models to be trained again.
#
# WHAT THIS FILE WRITES (all in output/)
# --------------------------------------
# 1. models/forest/<model>__<target>.joblib   12 files: 2 models x 6 labels,
#    each fitted on the 80% training satellites (train_test_split.csv).
# 2. cv_predictions_v3.csv   out-of-fold predictions from 5-fold CV over ALL
#    satellites: sat_idx, fold, target, model, y_true, y_pred. Each satellite
#    is predicted once, by a model that did not train on it. The 5 fold models
#    per label are not kept, only these predictions.
#
# HISTORY (from the earlier single-file version)
# ----------------------------------------------
# - Input: rr_c_* + mask_* columns from training_table_padded.csv.
# - Angular labels (raan, argp, anom): trained as sin/cos pairs and recombined
#   via atan2 (CircularAngleRegressor, now in common.py), scored with wrapped
#   circular metrics (metrics.py) instead of raw R2/RMSE.
# - Split: one fixed 80/20 split keyed by sat_idx, saved to
#   output/train_test_split.csv (splits.py), shared with the CNN.
# - CV: 5-fold over ALL satellites, separate from the held-out test score.
# - The helpers shared with the CNN trainer live in common.py.
# =============================================================================

import os
import time

import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from Clean_Testing.splits import make_or_load_split, RANDOM_STATE
from Clean_Testing.common import ALL_TARGETS, CV_FOLDS, CircularAngleRegressor, is_circular, load_table


MODELS = {
    "lightgbm": lambda seed: LGBMRegressor(random_state=seed, verbosity=-1),
    "random_forest": lambda seed: RandomForestRegressor(n_estimators=300, random_state=seed),
}

MODEL_DIR = "output/models/forest"


def model_path(model_name, target):
    return os.path.join(MODEL_DIR, f"{model_name}__{target}.joblib")


def build_estimator(model_name, target, seed):
    base = MODELS[model_name](seed)
    return CircularAngleRegressor(base_estimator=base) if is_circular(target) else base


def fit_on_train(df, feat_cols, target, model_name, train_ids, seed=RANDOM_STATE):
    """Fit one model on the training satellites only. Returns the fitted model."""
    sub = df.dropna(subset=[target])
    train_mask = sub["sat_idx"].isin(train_ids)
    X_train, y_train = sub.loc[train_mask, feat_cols], sub.loc[train_mask, target]

    model = build_estimator(model_name, target, seed)
    model.fit(X_train, y_train)
    return model


def cv_predictions(df, feat_cols, target, model_name,
                   n_folds=CV_FOLDS, seed=RANDOM_STATE):
    """5-fold CV over ALL satellites (train and test together). Returns the
    out-of-fold predictions (sat_idx, fold, target, model, y_true, y_pred).
    Scoring them, per fold, is done in forest_testing.py."""
    sub = df.dropna(subset=[target])

    X = sub[feat_cols].reset_index(drop=True)
    y = sub[target].reset_index(drop=True)
    sat = sub["sat_idx"].reset_index(drop=True)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = build_estimator(model_name, target, seed)
        model.fit(X_tr, y_tr)
        pred_val = model.predict(X_val)

        oof.append(pd.DataFrame({
            "sat_idx": sat.iloc[val_idx].to_numpy(), "fold": fold,
            "target": target, "model": model_name,
            "y_true": y_val.to_numpy(), "y_pred": pred_val,
        }))
    return pd.concat(oof, ignore_index=True)


if __name__ == "__main__":

    df, feat_cols = load_table()
    n_rr = sum(c.startswith("rr_c_") for c in feat_cols)
    n_mask = sum(c.startswith("mask_") for c in feat_cols)
    print(f"Loaded {len(df)} rows, {len(feat_cols)} features "
          f"({n_rr} rr_c_* + {n_mask} mask_*) from output/training_table_padded.csv\n")

    split_df = make_or_load_split(df["sat_idx"])
    train_ids = set(split_df.loc[split_df["split"] == "train", "sat_idx"])
    test_ids = set(split_df.loc[split_df["split"] == "test", "sat_idx"])
    print(f"Split: {len(train_ids)} train / {len(test_ids)} test satellites "
          f"(output/train_test_split.csv)\n")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- fit on the training satellites, once per target x model, and save ---
    print("Fitting on the training satellites and saving the models")
    for target in ALL_TARGETS:
        if target not in df.columns:
            print(f"  [skip] {target} not in table")
            continue
        for model_name in MODELS:
            t0 = time.time()
            model = fit_on_train(df, feat_cols, target, model_name, train_ids)
            path = model_path(model_name, target)
            joblib.dump(model, path, compress=3)
            print(f"  saved {path}  ({os.path.getsize(path) / 1e6:.1f} MB, "
                  f"{(time.time() - t0) / 60:.1f} min)")

    # --- 5-fold CV over all satellites: keep the out-of-fold predictions ---
    print("\n5-fold cross-validation on all satellites "
          "(every satellite is predicted once while unseen)")
    cv_frames = []
    for target in ALL_TARGETS:
        if target not in df.columns:
            continue
        for model_name in MODELS:
            t0 = time.time()
            cv_frames.append(cv_predictions(df, feat_cols, target, model_name))
            print(f"  cv {model_name:>13} {target:>16}  ({(time.time() - t0) / 60:.1f} min)")

    os.makedirs("output", exist_ok=True)
    cv_path = "output/cv_predictions_v3.csv"
    pd.concat(cv_frames, ignore_index=True).to_csv(cv_path, index=False)
    print(f"\nSaved out-of-fold CV predictions to {cv_path}")
    print("Next: run forest_testing.py to score the saved models and the CV predictions.")
