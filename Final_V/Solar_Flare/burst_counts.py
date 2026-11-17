"""
burst_counts.py : counts behind the noise burst results in the report (Section 4.3).

Reads the files solar_flare.py wrote and prints, for the 2,000 test satellites:
  - how many really had a burst and how many were clean (from flare_predictions.csv)
  - for each model: bursts caught, bursts missed, false alarms, clean passes called clean
  - accuracy, precision, recall and F1 worked out from those counts
and checks that they match flare_results.csv.

Run from this folder (Final_V/Solar_Flare), after solar_flare.py:

    python burst_counts.py
"""
import os
import sys

import pandas as pd

OUT = "output"
PRED = os.path.join(OUT, "flare_predictions.csv")
RES = os.path.join(OUT, "flare_results.csv")

if __name__ == "__main__":
    for p in (PRED, RES):
        if not os.path.exists(p):
            sys.exit(f"{p} not found. Run solar_flare.py first, from this folder.")
    pred = pd.read_csv(PRED)
    res = pd.read_csv(RES).set_index("model")

    for model, p in pred.groupby("model", sort=False):
        y, yhat = p["flare_affected"], p["predicted"]
        n = len(p)
        burst, clean = int((y == 1).sum()), int((y == 0).sum())
        tp = int(((y == 1) & (yhat == 1)).sum())
        fn = int(((y == 1) & (yhat == 0)).sum())
        fp = int(((y == 0) & (yhat == 1)).sum())
        tn = int(((y == 0) & (yhat == 0)).sum())
        acc = (tp + tn) / n
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        guess = max(burst, clean) / n

        print(f"{model}: {n} test satellites, {burst} with a burst, {clean} clean")
        print(f"  bursts caught (tp)        {tp:5d}")
        print(f"  bursts missed (fn)        {fn:5d}   caught + missed = {tp + fn}")
        print(f"  false alarms (fp)         {fp:5d}")
        print(f"  clean called clean (tn)   {tn:5d}   false alarms + clean = {fp + tn}")
        print(f"  accuracy  {acc:.4f}   guessing {guess:.4f}")
        print(f"  precision {prec:.4f}   recall {rec:.4f}   F1 {f1:.4f}")

        r = res.loc[model]
        same = (int(r["tp"]) == tp and int(r["fn"]) == fn and int(r["fp"]) == fp and int(r["tn"]) == tn
                and abs(r["accuracy"] - acc) < 1e-9 and abs(r["precision"] - prec) < 1e-9
                and abs(r["recall"] - rec) < 1e-9 and abs(r["f1"] - f1) < 1e-9
                and abs(r["baseline_accuracy"] - guess) < 1e-9)
        print(f"  matches flare_results.csv: {'yes' if same else 'NO, check the files'}\n")
