"""
solar_flare.py : anomaly detection, solar flare scenario.

A solar flare is simulated as a long burst of extra noise on the range rate. Half of the
satellites (picked with a fixed seed) get one burst, the other half stay clean. A yes/no
classifier (LightGBM and Random Forest) is trained to say whether a satellite pass was hit
by a flare.

Run from this folder (needs delay_differences.csv from the clean sigma 0 run, and splits.py):

    python solar_flare.py

Steps
1. Read the clean delays and turn them into range rate (m/s): rr = dd * c / delta_t.
2. Pick 50% of the satellites with a fixed seed: flare_affected = 1. For each of them,
   add white Gaussian noise of BURST_SIGMA m/s to one continuous stretch of the pass,
   about 30 to 70% of its readings long (rounded to whole readings), at a random position. No readings are removed.
   The other 50% are unchanged (flare_affected = 0).
3. Features per satellite: the range-rate readings and their 1st, 2nd and 3rd
   differences (padded with 0 after the last reading), plus the number of readings.
4. Same 80/20 split by sat_idx as the rest of the project (splits.py, seed 42).
   Train on the training satellites, score on the held-out test satellites. No CV.
5. Write the results to output/.

Writes (in output/)
  flare_labels.csv       sat_idx, flare_affected, burst_start, burst_end (reading index,
                         end not included), burst_fraction
  flare_results.csv      one row per model: accuracy, baseline_accuracy (always guess the
                         more common answer), precision, recall, f1, tn, fp, fn, tp
  flare_predictions.csv  per test satellite: true label, predicted label, probability
  train_test_split.csv   from splits.py (reused if it already exists)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from splits import make_or_load_split

DELAYS = "delay_differences.csv"   # clean sigma 0 run
OUT = "output"

C_LIGHT = 299792458.0   # m/s
DELTA_T = 0.001         # s, the same for every satellite in the clean run
BURST_SIGMA = 5.0       # m/s, strength of the flare noise
BURST_MIN, BURST_MAX = 0.3, 0.7   # burst length as a fraction of the pass
AFFECTED_FRACTION = 0.5
CHOOSE_SEED = 2025      # picks which satellites are affected
BURST_SEED = 2024       # burst position, length and noise draws, per satellite
MODEL_SEED = 42


def flare_burst(n, sat_idx, sigma=BURST_SIGMA, seed=BURST_SEED):
    """One burst for a pass of n readings. Returns (start, end, noise); noise is 0 outside."""
    rng = np.random.default_rng([seed, int(sat_idx)])
    length = int(round(rng.uniform(BURST_MIN, BURST_MAX) * n))
    start = int(rng.integers(0, n - length + 1))
    noise = np.zeros(n)
    noise[start:start + length] = sigma * rng.standard_normal(length)
    return start, start + length, noise


def padded(values, width):
    row = np.zeros(width)
    row[:len(values)] = values
    return row


def build_table(delays):
    dd_cols = [c for c in delays.columns if c.startswith("dd_")]
    width = len(dd_cols)
    sat_ids = delays["sat_idx"].to_numpy()

    rng = np.random.default_rng(CHOOSE_SEED)
    n_affected = int(round(AFFECTED_FRACTION * len(sat_ids)))
    affected = set(rng.permutation(sat_ids)[:n_affected].tolist())

    labels, rows = [], []
    for _, r in delays.iterrows():
        s = int(r["sat_idx"])
        n = int(r["num_valid_samples"])
        rr = r[dd_cols].to_numpy(float)[:n] * C_LIGHT / DELTA_T
        if s in affected:
            b0, b1, noise = flare_burst(n, s)
            rr = rr + noise
            labels.append((s, 1, b0, b1, (b1 - b0) / n))
        else:
            labels.append((s, 0, -1, -1, 0.0))
        feats = [padded(rr, width)]
        for k in (1, 2, 3):
            feats.append(padded(np.diff(rr, k), width - k))
        rows.append(np.concatenate([[n]] + feats))

    names = (["num_readings"] + [f"rr_{i:03d}" for i in range(width)]
             + [f"d{k}_{i:03d}" for k in (1, 2, 3) for i in range(width - k)])
    X = pd.DataFrame(rows, columns=names)
    X.insert(0, "sat_idx", sat_ids)
    lab = pd.DataFrame(labels, columns=["sat_idx", "flare_affected", "burst_start", "burst_end",
                                        "burst_fraction"])
    return X, lab


def score(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    majority = max(np.mean(y), 1 - np.mean(y))
    return {
        "accuracy": accuracy_score(y, pred), "baseline_accuracy": majority,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


if __name__ == "__main__":
    t0 = time.time()
    if not os.path.exists(DELAYS):
        sys.exit(f"{DELAYS} not found. Copy the clean sigma 0 delay_differences.csv into this folder.")
    os.makedirs(OUT, exist_ok=True)

    delays = pd.read_csv(DELAYS)
    print(f"Loaded {len(delays)} satellites from {DELAYS}")
    X, lab = build_table(delays)
    lab.to_csv(os.path.join(OUT, "flare_labels.csv"), index=False)
    hit = lab[lab["flare_affected"] == 1]
    print(f"Flare added to {len(hit)} of {len(lab)} satellites at {BURST_SIGMA} m/s, "
          f"burst covers {hit['burst_fraction'].min():.0%} to {hit['burst_fraction'].max():.0%} "
          f"of the pass (mean {hit['burst_fraction'].mean():.0%})")

    split = make_or_load_split(X["sat_idx"], split_path=os.path.join(OUT, "train_test_split.csv"))
    train_ids = set(split.loc[split["split"] == "train", "sat_idx"])
    is_train = X["sat_idx"].isin(train_ids).to_numpy()
    y = lab["flare_affected"].to_numpy()
    feats = X.drop(columns="sat_idx")
    print(f"Split: {is_train.sum()} train / {(~is_train).sum()} test satellites")

    models = {
        "LightGBM": LGBMClassifier(n_estimators=300, random_state=MODEL_SEED, verbose=-1),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=MODEL_SEED, n_jobs=-1),
    }
    results, preds = [], []
    for name, model in models.items():
        t1 = time.time()
        model.fit(feats[is_train], y[is_train])
        prob = model.predict_proba(feats[~is_train])[:, 1]
        pred = (prob >= 0.5).astype(int)
        m = score(y[~is_train], pred)
        results.append({"model": name, "burst_sigma_ms": BURST_SIGMA,
                        "n_train": int(is_train.sum()), "n_test": int((~is_train).sum()), **m})
        preds.append(pd.DataFrame({"sat_idx": X.loc[~is_train, "sat_idx"].to_numpy(), "model": name,
                                   "flare_affected": y[~is_train], "predicted": pred,
                                   "prob_flare": prob}))
        print(f"  {name}: trained and scored in {time.time() - t1:.0f} s")

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(OUT, "flare_results.csv"), index=False)
    pd.concat(preds, ignore_index=True).to_csv(os.path.join(OUT, "flare_predictions.csv"), index=False)

    print("\nHeld-out test satellites (flare = yes):")
    print(f"{'model':>13} {'accuracy':>9} {'baseline':>9} {'precision':>10} {'recall':>7} {'f1':>6}"
          f"   tn    fp    fn    tp")
    for r in results:
        print(f"{r['model']:>13} {r['accuracy']:>9.3f} {r['baseline_accuracy']:>9.3f} "
              f"{r['precision']:>10.3f} {r['recall']:>7.3f} {r['f1']:>6.3f} "
              f"{r['tn']:>5} {r['fp']:>5} {r['fn']:>5} {r['tp']:>5}")
    print(f"\ntn = clean called clean, fp = clean called flare, fn = flare missed, tp = flare caught")
    print(f"Saved results to {OUT}/. Done in {(time.time() - t0) / 60:.1f} min.")
