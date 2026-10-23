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
# 2. models/forest/COMPLETE.txt   written last, so run_all.py can tell that
#    all 12 models were saved.
#
# NO CROSS-VALIDATION IN THIS FOLDER
# ----------------------------------
# The noise sweep reports the held-out test only (Clean_Testing also does 5-fold
# CV). CV would train every model six times instead of once.
#
# HISTORY (from the earlier single-file version)
# ----------------------------------------------
# - Input: rr_c_* + mask_* columns from training_table_padded.csv.
# - Angular labels (raan, argp, anom): trained as sin/cos pairs and recombined
#   via atan2 (CircularAngleRegressor, now in common.py), scored with wrapped
#   circular metrics (metrics.py) instead of raw R2/RMSE.
# - Split: one fixed 80/20 split keyed by sat_idx, saved to
#   output/train_test_split.csv (splits.py), shared with the CNN.
# - The helpers shared with the CNN trainer live in common.py.
# =============================================================================

import os
import time

import joblib
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

from splits import make_or_load_split, RANDOM_STATE
from common import ALL_TARGETS, CircularAngleRegressor, is_circular, load_table


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

    with open(os.path.join(MODEL_DIR, "COMPLETE.txt"), "w") as f:
        f.write("all forest models saved\n")
    print("\nNext: run forest_testing.py to score the saved models.")
