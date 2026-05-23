"""
ViEWS-2024 style true-future temporal holdout evaluation.

Hegre et al. 2024 (JPR / arXiv:2407.11045) canonical CV for rare-event panel
forecasting is an EXPANDING-WINDOW ONE-SHOT temporal split holding out by
YEAR (not country-year). We replicate that here on our existing ews_signals
predictions to produce headline metrics defensible to ViEWS-literate reviewers.

Partitioning:
  TRAIN : year <= 2013        # fit
  CALIB : 2014 <= year <= 2019  # tune threshold / isotonic
  TEST  : year >= 2020          # one-shot true-future, FROZEN

We compute Brier, log-loss, BSS, AUC-ROC, AUC-PR on TEST only, with
country-cluster bootstrap CIs. This is reported alongside the k-fold and
cluster-bootstrap CIs already in robustness/bootstrap_cis.csv — not as a
replacement, but as the ViEWS-compatible headline.

Output: robustness/temporal_holdout.csv with one row per metric.

Requires: stage5_ews/ews_signals.csv with calibrated_risk + label columns.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_cis import bootstrap_auc

EWS = os.path.join(REPO, "stage5_ews", "ews_signals.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temporal_holdout.csv")

EPS = 1e-15
TRAIN_END = 2013
CALIB_START, CALIB_END = 2014, 2019
TEST_START = 2020


def _brier(y, p, w=None):
    return brier_score_loss(y, np.clip(p, EPS, 1 - EPS), sample_weight=w)


def _logloss(y, p, w=None):
    return log_loss(y, np.clip(p, EPS, 1 - EPS),
                    sample_weight=w, labels=[0, 1])


def _bss(y, p, w=None):
    p_clim = np.average(y, weights=w) if w is not None else float(np.mean(y))
    ref = (np.average((y - p_clim) ** 2, weights=w)
           if w is not None else float(np.mean((y - p_clim) ** 2)))
    return 1.0 - _brier(y, p, w) / max(ref, EPS)


def main():
    if not os.path.exists(EWS):
        sys.exit(f"Missing {EWS} — run stage5_ews/estimate.py first")
    ews = pd.read_csv(EWS)

    needed = {"country_text_id", "year", "calibrated_risk", "label"}
    missing = needed - set(ews.columns)
    if missing:
        sys.exit(f"ews_signals.csv missing required columns: {missing}")

    valid = ews.dropna(subset=["calibrated_risk", "label"]).copy()
    if "is_postonset" in valid.columns:
        valid = valid[~valid["is_postonset"].fillna(False)]

    train = valid[valid["year"] <= TRAIN_END]
    calib = valid[(valid["year"] >= CALIB_START) & (valid["year"] <= CALIB_END)]
    test = valid[valid["year"] >= TEST_START]

    print("=" * 70)
    print("ViEWS-style temporal holdout (Hegre et al. 2024 JPR)")
    print("=" * 70)
    print(f"  TRAIN (year <= {TRAIN_END}):     n={len(train):>5d}  "
          f"n_pos={int(train['label'].sum()):>4d}")
    print(f"  CALIB ({CALIB_START}-{CALIB_END}):  n={len(calib):>5d}  "
          f"n_pos={int(calib['label'].sum()):>4d}")
    print(f"  TEST  (year >= {TEST_START}):     n={len(test):>5d}  "
          f"n_pos={int(test['label'].sum()):>4d}")
    print()

    if len(test) < 50 or test["label"].sum() < 2:
        print("WARNING: TEST partition too small or no positives.")
        sys.exit(1)

    y = test["label"].astype(int).values
    s = test["calibrated_risk"].values
    clusters = test["country_text_id"].values

    rows = []
    for fn_name, fn in [
        ("auc_roc", roc_auc_score),
        ("auc_pr", average_precision_score),
        ("brier", _brier),
        ("logloss", _logloss),
        ("bss", _bss),
    ]:
        try:
            pt, lo, hi = bootstrap_auc(y, s, fn=fn, n_boot=2000, clusters=clusters)
            rows.append({
                "metric": fn_name, "point": pt,
                "ci_low": lo, "ci_high": hi,
                "n": len(y), "n_positive": int(y.sum()),
            })
            print(f"  {fn_name:8s}: {pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        except Exception as e:
            print(f"  {fn_name:8s}: error ({e})")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
