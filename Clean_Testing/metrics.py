# =============================================================================
# metrics.py -- evaluation metrics for linear and circular (angular) labels
# =============================================================================
#
# WHY THIS EXISTS
# -----------------
# sma, ecc, and inc_deg are ordinary linear quantities -- R2/RMSE apply
# directly. raan_deg, argp_deg, and anom_deg are angles that wrap at 360 deg:
# a prediction of 359 deg against a true value of 1 deg is a 2-degree error,
# not a 358-degree one. Per the project's own noted principle, RMSE and R2
# on the raw degree values are not reliable for these three labels -- this
# module is the wrapped replacement.
#
# CIRCULAR CORRELATION
# ----------------------
# circ_corr is the circular correlation coefficient, the angular version of
# Pearson's correlation (Jammalamadaka and SenGupta, 2001, Topics in Circular
# Statistics, Sec. 8.2; first proposed in Jammalamadaka and Sarma, 1988). It is
# Pearson's formula with the deviations replaced by sin(angle - mean direction),
# computed separately for the true and the predicted angles. Range -1 to 1:
# 0 means no relationship, 1 means the predicted angles track the true angles
# perfectly, negative means they move in opposite directions. It measures
# whether the angles move together, not how large the error is, so it is
# reported next to circular RMSE and not instead of it.
# =============================================================================

import numpy as np
from sklearn.metrics import r2_score


def linear_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }


def circular_error_deg(y_true_deg, y_pred_deg):
    """Wrapped signed difference (pred - true), in (-180, 180] degrees."""
    y_true_deg = np.asarray(y_true_deg, dtype=float)
    y_pred_deg = np.asarray(y_pred_deg, dtype=float)
    return (y_pred_deg - y_true_deg + 180.0) % 360.0 - 180.0


def circular_rmse(y_true_deg, y_pred_deg):
    err = circular_error_deg(y_true_deg, y_pred_deg)
    return float(np.sqrt(np.mean(err ** 2)))


def _mean_direction(angles_rad):
    return np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad)))


def circular_corr(y_true_deg, y_pred_deg):
    """
    Circular correlation coefficient (Jammalamadaka and SenGupta, 2001, Sec. 8.2):
    sum(sin(t - mean_t) * sin(p - mean_p)) /
        sqrt(sum(sin(t - mean_t)^2) * sum(sin(p - mean_p)^2)),
    t = true angles, p = predicted angles, mean = circular mean direction.
    Returns NaN if either set of angles has no spread.
    """
    t = np.radians(np.asarray(y_true_deg, dtype=float))
    p = np.radians(np.asarray(y_pred_deg, dtype=float))
    st = np.sin(t - _mean_direction(t))
    sp = np.sin(p - _mean_direction(p))
    denom = np.sqrt(np.sum(st ** 2) * np.sum(sp ** 2))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(st * sp) / denom)


def circular_metrics(y_true_deg, y_pred_deg):
    return {
        "rmse": circular_rmse(y_true_deg, y_pred_deg),
        "circ_corr": circular_corr(y_true_deg, y_pred_deg),
    }
