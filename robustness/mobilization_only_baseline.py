"""
Baseline: mobilization-only logistic model under the strict 2019 hold-out.

Scores a logistic regression on the six mobilization features alone
(v2cagenmob, v2cademmob, v2caautmob and their detrended forms) with the
identical protocol as the Table 5 baseline ladder: train through 2019,
score post-2019 country-years with post-onset years excluded. Shows how
far the headline channel gets by itself, absent the multi-channel system.

Output: robustness/mobilization_only_baseline.csv
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stage5_ews.estimate import KNOWN_EPISODES

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "mobilization_only_baseline.csv")
LEAD = 5


def main():
    ews = pd.read_csv(os.path.join(REPO, "stage5_ews", "ews_signals.csv"))
    feats = [c for c in ews.columns
             if ("cademmob" in c or "cagenmob" in c or "caautmob" in c)
             and ews[c].notna().mean() > 0.5]

    pre, post = set(), set()
    for c, info in KNOWN_EPISODES.items():
        onset = info["onset"]
        for y in range(onset - LEAD, onset + 1):
            pre.add((c, y))
        for y in range(onset + 1, onset + 6):
            post.add((c, y))
    ews["lbl"] = [1 if (c, y) in pre else 0
                  for c, y in zip(ews["country_name"], ews["year"])]
    ews["post"] = [(c, y) in post
                   for c, y in zip(ews["country_name"], ews["year"])]

    d = ews[ews["combined_risk"].notna()].copy()
    X = d[feats].fillna(0).values
    y = d["lbl"].values
    train = (d["year"] <= 2019).values
    test = ((d["year"] > 2019) & (~d["post"])).values

    scaler = StandardScaler().fit(X[train])
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42,
                               class_weight="balanced")
    model.fit(scaler.transform(X[train]), y[train])
    p = model.predict_proba(scaler.transform(X[test]))[:, 1]

    row = {
        "model": "mobilization_only_logit",
        "n_features": len(feats),
        "features": ";".join(feats),
        "auc_roc": roc_auc_score(y[test], p),
        "auc_pr": average_precision_score(y[test], p),
        "n_test": int(test.sum()),
        "n_pos": int(y[test].sum()),
    }
    pd.DataFrame([row]).to_csv(OUT, index=False)
    print(f"mobilization-only ({len(feats)} features): "
          f"AUC-ROC {row['auc_roc']:.3f}  AUC-PR {row['auc_pr']:.3f}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
