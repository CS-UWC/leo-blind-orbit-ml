# =============================================================================
# forests.py -- LightGBM and Random Forest on the closest-approach labels:
#               fit on the training satellites, score the held-out test ones
# =============================================================================
#
# One model per label and per method, no hyperparameter tuning (as in
# Noise_Testing). Linear labels use the regressors, binary labels the
# classifiers. Random Forest builds its trees on every CPU core (n_jobs=-1;
# same trees as single-core).
#
# WRITES (output/)
#   test_predictions_forest.csv   sat_idx, target, model, y_true, y_pred
#                                 (binary: predicted class; y_prob = P(1))
#   test_results_forest.csv       one row per target x model (common.score)
#   forest_COMPLETE.txt           written last, so run_all.py can tell it finished
# =============================================================================

import time

import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from common import ALL_TARGETS, kind, load_table, print_results, score
from splits import RANDOM_STATE, make_or_load_split

MODELS = {  # name -> (regressor, classifier)
    "lightgbm": (lambda s: LGBMRegressor(random_state=s, verbosity=-1),
                 lambda s: LGBMClassifier(random_state=s, verbosity=-1)),
    "random_forest": (lambda s: RandomForestRegressor(n_estimators=300, random_state=s, n_jobs=-1),
                      lambda s: RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1)),
}


if __name__ == "__main__":
    df, feat = load_table()
    split = make_or_load_split(df["sat_idx"])
    train_ids = set(split.loc[split["split"] == "train", "sat_idx"])
    train, test = df[df["sat_idx"].isin(train_ids)], df[~df["sat_idx"].isin(train_ids)]
    print(f"{len(train)} train / {len(test)} test satellites, {len(feat)} inputs\n")

    results, preds = [], []
    for target in ALL_TARGETS:
        for name, (make_reg, make_clf) in MODELS.items():
            t0 = time.time()
            binary = kind(target) == "binary"
            model = (make_clf if binary else make_reg)(RANDOM_STATE)
            model.fit(train[feat], train[target])
            y_pred = model.predict(test[feat])
            p = pd.DataFrame({"sat_idx": test["sat_idx"].to_numpy(), "target": target, "model": name,
                              "y_true": test[target].to_numpy(), "y_pred": y_pred})
            if binary:
                p["y_prob"] = model.predict_proba(test[feat])[:, 1]
            preds.append(p)
            results.append({"target": target, "model": name, "n_train": len(train), "n_test": len(test),
                            **score(test[target], y_pred, target)})
            print(f"  {name:>13} {target:>26}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    print()
    print_results(results)
    pd.concat(preds, ignore_index=True).to_csv("output/test_predictions_forest.csv", index=False)
    pd.DataFrame(results).to_csv("output/test_results_forest.csv", index=False)
    with open("output/forest_COMPLETE.txt", "w") as f:
        f.write("done\n")
    print("\nSaved output/test_predictions_forest.csv and output/test_results_forest.csv")
