"""Pressure-test the external benchmarks before drawing conclusions.

Three diagnostics the headline CV number cannot settle on its own:
  1. per-fold stability across the expanding windows (is 0.83 one lucky fold?),
  2. the strict train-<=2019 -> predict-future hold-out, the protocol AIM4D's
     0.931 lives on and the generalization claim rests on,
  3. AUC-PR alongside AUC-ROC, since ROC flatters at an 8% base rate.

Reuses the exact loaders, models, and labels from external_benchmarks.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage5_ews.estimate import KNOWN_EPISODES, LEAD_YEARS
from baseline_comparison import build_labels, WINDOWS
from external_benchmarks import (load_panel, load_indicator_panel, make_enet,
                                 UnweightedEnsemble, REGIME_DUMMIES)

POSTONSET = set()
for _c, _info in KNOWN_EPISODES.items():
    _o = _info["onset"]
    for _y in range(_o + 1, _o + 6):
        POSTONSET.add((_c, _y))


def _postmask(df):
    return np.array([(c, y) in POSTONSET for c, y in zip(df["country_name"], df["year"])])


def perfold(df, feats, make_model, label="label"):
    out = []
    for t0, t1 in WINDOWS:
        tr = df[(df["year"] <= t0) & df[feats].notna().all(axis=1) & df[label].notna()]
        te = df[(df["year"] > t0) & (df["year"] <= t1) & df[feats].notna().all(axis=1) & df[label].notna()]
        if tr[label].sum() < 3 or te[label].nunique() < 2:
            out.append((f"{t0}-{t1}", np.nan, np.nan, int(te[label].sum())))
            continue
        sc = StandardScaler()
        m = make_model()
        m.fit(sc.fit_transform(tr[feats].values), tr[label].values)
        p = m.predict_proba(sc.transform(te[feats].values))[:, 1]
        out.append((f"{t0}-{t1}", roc_auc_score(te[label], p),
                    average_precision_score(te[label], p), int(te[label].sum())))
    return out


def perfold_score(df, score, label="label"):
    out = []
    for t0, t1 in WINDOWS:
        te = df[(df["year"] > t0) & (df["year"] <= t1) & df[score].notna() & df[label].notna()]
        if te[label].nunique() < 2:
            out.append((f"{t0}-{t1}", np.nan, np.nan, int(te[label].sum())))
            continue
        out.append((f"{t0}-{t1}", roc_auc_score(te[label], te[score]),
                    average_precision_score(te[label], te[score]), int(te[label].sum())))
    return out


def holdout(df, feats, make_model, label="label"):
    d = df[df[feats].notna().all(axis=1) & df[label].notna()].copy()
    post = _postmask(d)
    tr = d[(d["year"] <= 2019)]
    te = d[(d["year"] > 2019) & (~post)]
    if te[label].nunique() < 2:
        return np.nan, np.nan, int(te[label].sum()), len(te)
    sc = StandardScaler()
    m = make_model()
    m.fit(sc.fit_transform(tr[feats].values), tr[label].values)
    p = m.predict_proba(sc.transform(te[feats].values))[:, 1]
    return roc_auc_score(te[label], p), average_precision_score(te[label], p), int(te[label].sum()), len(te)


def holdout_score(df, score, label="label"):
    d = df[df[score].notna() & df[label].notna()].copy()
    post = _postmask(d)
    te = d[(d["year"] > 2019) & (~post)]
    if te[label].nunique() < 2:
        return np.nan, np.nan, int(te[label].sum()), len(te)
    return (roc_auc_score(te[label], te[score]), average_precision_score(te[label], te[score]),
            int(te[label].sum()), len(te))


def show(name, folds, hold):
    aucs = [f[1] for f in folds if not np.isnan(f[1])]
    print(f"\n### {name}")
    print("  per-fold (window: AUC / AUC-PR / n_pos):")
    for w, a, ap, n in folds:
        print(f"    {w:>10}:  {a:5.3f} / {ap:5.3f}  (n_pos={n})" if not np.isnan(a)
              else f"    {w:>10}:  --    (n_pos={n})")
    print(f"  WINDOWS CV: AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
    print(f"  STRICT 2019 HOLD-OUT: AUC {hold[0]:.3f}, AUC-PR {hold[1]:.3f} "
          f"(n_pos={hold[2]}, n={hold[3]})")


def main():
    panel = load_panel()
    ind, indicators = load_indicator_panel()
    ews = build_labels(pd.read_csv(os.path.join(REPO, "stage5_ews", "ews_signals.csv")))

    pitf_feats = REGIME_DUMMIES + ["imr_normed_ln", "n_backsliding_neighbors", "state_discrimination"]

    print("=" * 70)
    print("BENCHMARK STRESS TEST")
    print("=" * 70)

    show("Persistence (3yr decline)",
         perfold_score(panel, "poly_decline_3yr"),
         holdout_score(panel, "poly_decline_3yr"))

    show("PITF logit (Goldstone 2010)",
         perfold(panel, pitf_feats, lambda: LogisticRegression(
             C=1.0, max_iter=2000, class_weight="balanced", random_state=42)),
         holdout(panel, pitf_feats, lambda: LogisticRegression(
             C=1.0, max_iter=2000, class_weight="balanced", random_state=42)))

    show("Elastic-net (full V-Dem)",
         perfold(ind, indicators, make_enet),
         holdout(ind, indicators, make_enet))

    show("V-Forecast ensemble (Morgan 2019)",
         perfold(ind, indicators, UnweightedEnsemble),
         holdout(ind, indicators, UnweightedEnsemble))

    h = holdout_score(ews, "combined_risk")
    print("\n### AIM4D (combined_risk) -- 2019 hold-out is HONEST OOS; per-fold pre-2019 is in-sample")
    print(f"  STRICT 2019 HOLD-OUT: AUC {h[0]:.3f}, AUC-PR {h[1]:.3f} (n_pos={h[2]}, n={h[3]})")
    print("  [paper reports full-pipeline-refit expanding CV = 0.774; single-cutoff OOS = 0.931]")


if __name__ == "__main__":
    main()
