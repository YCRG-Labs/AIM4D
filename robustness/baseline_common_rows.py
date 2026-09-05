"""Baseline ladder scored on the same hold-out rows as the framework.

The conventional-design baseline table compared models that had each been
scored on their own available rows, so the numbers were not commensurable and
could not be reproduced from any stored output. Here every model is trained on
country-years through 2019 and scored on exactly the rows the framework is
scored on: the post-2019, non-post-onset country-years carrying both a label
and a combined_risk. Models whose inputs are missing for some of those rows are
reported with the intersection size so the shortfall is visible rather than
absorbed into the metric.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from external_benchmarks import (REGIME_DUMMIES, UnweightedEnsemble, load_indicator_panel,
                                 load_panel, make_enet)

OUT = os.path.dirname(os.path.abspath(__file__))
CUTOFF = 2019
KEY = ["country_name", "year"]


def eval_rows():
    ews = pd.read_csv(os.path.join(OUT, "..", "stage5_ews", "ews_signals.csv"))
    v = ews.dropna(subset=["combined_risk", "label"])
    if "is_postonset" in v.columns:
        v = v[~v["is_postonset"]]
    v = v[v["year"] > CUTOFF]
    return v[KEY + ["label", "combined_risk"]].copy()


def fitted(panel, feats, make_model, ev):
    p = panel.dropna(subset=["label"]).copy()
    tr = p[p["year"] <= CUTOFF]
    te = p[p["year"] > CUTOFF]
    X = tr[feats].astype(float)
    med = X.median()
    sc = StandardScaler().fit(X.fillna(med).values)
    m = make_model()
    m.fit(sc.transform(X.fillna(med).values), tr["label"].astype(int).values)
    pr = m.predict_proba(sc.transform(te[feats].astype(float).fillna(med).values))[:, 1]
    out = te[KEY].copy()
    out["score"] = pr
    return ev.merge(out, on=KEY, how="inner")


def scored(panel, col, ev):
    p = panel.dropna(subset=[col])
    p = p[p["year"] > CUTOFF][KEY + [col]].rename(columns={col: "score"})
    return ev.merge(p, on=KEY, how="inner")


N_BOOT = 2000
RNG = np.random.default_rng(20260905)


def boot_ci(y, s, countries):
    """Country-clustered percentile bootstrap CI for AUC and average precision."""
    uniq = np.unique(countries)
    idx = {c: np.where(countries == c)[0] for c in uniq}
    a, b = [], []
    for _ in range(N_BOOT):
        draw = RNG.choice(uniq, size=len(uniq), replace=True)
        j = np.concatenate([idx[c] for c in draw])
        if y[j].sum() < 3 or y[j].sum() == len(j):
            continue
        a.append(roc_auc_score(y[j], s[j]))
        b.append(average_precision_score(y[j], s[j]))
    q = lambda v: (round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3))
    return q(a), q(b)


def row(name, d):
    y = d["label"].astype(int).values
    s = d["score"].values
    if y.sum() < 2:
        return None
    (alo, ahi), (plo, phi) = boot_ci(y, s, d["country_name"].values)
    return {"model": name, "auc_roc": roc_auc_score(y, s),
            "auc_pr": average_precision_score(y, s),
            "auc_roc_lo": alo, "auc_roc_hi": ahi,
            "auc_pr_lo": plo, "auc_pr_hi": phi,
            "n": len(d), "n_pos": int(y.sum()),
            "base_rate": float(y.mean())}


def main():
    ev = eval_rows()
    print(f"canonical hold-out: n={len(ev)}, positives={int(ev['label'].sum())}, "
          f"base rate={ev['label'].mean():.3f}\n", flush=True)

    res = []
    res.append(row("Five-stage framework (AIM4D)",
                   ev.assign(score=ev["combined_risk"])))

    panel = load_panel()
    panel["persistence"] = panel["poly_decline_3yr"]
    res.append(row("Persistence (3-yr polyarchy decline)", scored(panel, "persistence", ev)))

    pitf = REGIME_DUMMIES + ["imr_normed_ln", "n_backsliding_neighbors", "state_discrimination"]
    res.append(row("PITF logit", fitted(panel, pitf, lambda: LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced", random_state=42), ev)))

    ind, indicators = load_indicator_panel()
    mob = [c for c in indicators if c.lower().startswith("v2ca")]
    print(f"indicators={len(indicators)}, mobilization block={len(mob)}", flush=True)
    res.append(row("Mobilization-only logit", fitted(ind, mob, lambda: LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced", random_state=42), ev)))
    res.append(row("Elastic net, V-Dem indicators", fitted(ind, indicators, make_enet, ev)))
    res.append(row("Gradient boosting, V-Dem indicators", fitted(
        ind, indicators, lambda: HistGradientBoostingClassifier(
            max_iter=200, max_depth=3, learning_rate=0.05, random_state=42), ev)))
    res.append(row("V-Forecast ensemble", fitted(ind, indicators, UnweightedEnsemble, ev)))

    df = pd.DataFrame([r for r in res if r])
    print("\n" + df.to_string(index=False, float_format="%.3f"), flush=True)
    df.to_csv(os.path.join(OUT, "baseline_common_rows.csv"), index=False)
    print("\nWrote baseline_common_rows.csv")


if __name__ == "__main__":
    main()
