# =============================================================================
# common.py -- code shared by forest_trainer_v3.py and cnn_trainer_v3.py
# =============================================================================
#
# Moved here unchanged from forest_trainer_v3.py so that the CNN trainer no
# longer has to import the forest trainer (and with it LightGBM) just to reuse
# these. Nothing in this file was rewritten: same target lists, same fold
# count, same CircularAngleRegressor, same load_table(), score() and
# _agg_stat(), so every number the two trainers produce is unchanged.
#
# Also keeps CircularAngleRegressor in a module that is imported, never run
# directly, so a model saved with joblib can be loaded again by another script.
# =============================================================================

import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

from metrics import linear_metrics, circular_metrics


LINEAR_TARGETS = ["label_sma_m", "label_ecc", "label_inc_deg"]
CIRCULAR_TARGETS = ["label_raan_deg", "label_argp_deg", "label_anom_deg"]
ALL_TARGETS = LINEAR_TARGETS + CIRCULAR_TARGETS

CV_FOLDS = 5


class CircularAngleRegressor(BaseEstimator, RegressorMixin):
    """
    Two copies of base_estimator -- one fit on sin(angle), one on
    cos(angle) (angle in degrees in, radians internally) -- recombined via
    atan2 at predict time, wrapped back to [0, 360) degrees.

    Subclasses BaseEstimator/RegressorMixin (get_params introspects
    __init__, base_estimator is cloned via sklearn's clone()) so it has a
    normal fit/predict interface -- build_estimator() and score() use it
    exactly like a plain regressor, no separate codepath for the circular
    labels in held_out_test() or cross_validate_all().
    """
    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        y_rad = np.radians(np.asarray(y, dtype=float))
        self.model_sin_ = clone(self.base_estimator)
        self.model_cos_ = clone(self.base_estimator)
        self.model_sin_.fit(X, np.sin(y_rad))
        self.model_cos_.fit(X, np.cos(y_rad))
        return self

    def predict(self, X):
        s = self.model_sin_.predict(X)
        c = self.model_cos_.predict(X)
        return np.degrees(np.arctan2(s, c)) % 360.0


def is_circular(target):
    return target in CIRCULAR_TARGETS


def load_table(path="output/training_table_padded.csv"):
    if not os.path.exists(path):
        raise SystemExit(
            f"Training table not found: {path}\nRun build_dataset_v3.py first."
        )
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c.startswith("rr_c_") or c.startswith("mask_")]
    return df, feat_cols


def score(y_true, y_pred, target):
    """Returns a dict with r2, rmse, circ_corr -- the two metric
    families share these four keys so downstream code doesn't branch on
    target type; whichever family doesn't apply is left as NaN."""
    if is_circular(target):
        m = circular_metrics(y_true, y_pred)
        m["r2"] = float("nan")  # not meaningful on wrapped angles -- see metrics.py
        return m
    m = linear_metrics(y_true, y_pred)
    m["circ_corr"] = float("nan")
    return m


def _agg_stat(values):
    """Mean and std across folds, NaN-safe: if every fold's value is NaN
    (the metric family that doesn't apply to this target -- r2 for a
    circular target, circ_corr for a linear one), returns (NaN, NaN)
    instead of letting np.nanmean warn on an all-NaN slice."""
    values = np.array(values, dtype=float)
    if np.all(np.isnan(values)):
        return float("nan"), float("nan")
    return float(np.nanmean(values)), float(np.nanstd(values))
