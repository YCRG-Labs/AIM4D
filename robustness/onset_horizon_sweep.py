"""
Horizon sweep with the ensemble aggregation chosen inside the training window.

Two gaps in onset_forecast_optimized.py are closed here. That script reported
five aggregations and the best one was picked after seeing test scores, which
biases the headline by roughly the spread among them; here the aggregation is
selected at each origin by the same blocked inner validation used for
hyperparameters, so the reported figure is pre-specified in the only sense that
matters. And it fixed h=5, whereas the horizon is an empirical question: a
signal that leads onset by a year need not lead it by five.

Protocol is otherwise unchanged. At-risk pool is democratic country-years not
already inside an episode, the target is onset during t+1..t+h, all predictors
are dated t or earlier, and forecasts roll across origins.

Outputs robustness/onset_horizon_sweep.csv.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from onset_forecast_clean import build_panel, label_h
from onset_forecast_optimized import (enrich, feature_list, fit_predict, inner_select,
                                      GRID_HGB, GRID_LR, ORIGINS)

OUT = os.path.dirname(os.path.abspath(__file__))
HORIZONS = [1, 2, 3, 5]
MODELS = [("hgb", GRID_HGB), ("rf", [{}]), ("lr", GRID_LR)]
AGGS = ["hgb", "rf", "lr", "mean", "rank"]


def base_preds(tr, te, feats, h, kind, grid):
    cfg = inner_select(tr, feats, h, kind, grid)
    sc = StandardScaler()
    Xtr = sc.fit_transform(tr[feats].fillna(0).values)
    Xte = sc.transform(te[feats].fillna(0).values)
    w = np.where(tr[f"y{h}"].values == 1,
                 (len(tr) - tr[f"y{h}"].sum()) / max(tr[f"y{h}"].sum(), 1), 1.0)
    return fit_predict(cfg, kind, Xtr, tr[f"y{h}"].values, Xte, w if kind == "hgb" else None)


def combine(pm, agg, groups=None):
    cols = ["hgb", "rf", "lr"]
    if agg in cols:
        return pm[agg].values
    M = pm[cols]
    if agg == "mean":
        return M.mean(axis=1).values
    r = M.copy()
    for c in cols:
        r[c] = pd.Series(M[c].values).rank(pct=True).values
    return r.mean(axis=1).values


def select_agg(tr, feats, h):
    """Choose the aggregation on held-out blocks strictly inside training."""
    yrs = np.sort(tr.year.unique())
    if len(yrs) < 14:
        return "mean"
    cuts = [yrs[int(len(yrs) * f)] for f in (0.7, 0.85)]
    tally = {a: [] for a in AGGS}
    for c in cuts:
        a_tr = tr[tr.year <= c - h]
        a_te = tr[(tr.year > c - h) & (tr.year <= c)]
        if len(a_te) < 30 or a_tr[f"y{h}"].sum() < 8 or a_te[f"y{h}"].sum() < 3:
            continue
        pm = {}
        for kind, grid in MODELS:
            try:
                pm[kind] = base_preds(a_tr, a_te, feats, h, kind, grid)
            except Exception:
                pm[kind] = np.full(len(a_te), np.nan)
        pm = pd.DataFrame(pm)
        if pm.isna().all().any():
            continue
        for agg in AGGS:
            try:
                tally[agg].append(average_precision_score(a_te[f"y{h}"].values, combine(pm, agg)))
            except Exception:
                pass
    scored = {a: np.mean(v) for a, v in tally.items() if v}
    return max(scored, key=scored.get) if scored else "mean"


def main():
    d = build_panel()
    d = enrich(d)
    rows = []
    for h in HORIZONS:
        d[f"y{h}"] = label_h(d, h)
        feats = feature_list(d) if h == HORIZONS[0] else feats
        recs, chosen = [], []
        for T in ORIGINS:
            tr = d[(d.year <= T - h) & d.at_risk]
            te = d[(d.year == T) & d.at_risk]
            if len(te) == 0 or tr[f"y{h}"].sum() < 8:
                continue
            agg = select_agg(tr, feats, h)
            chosen.append(agg)
            pm = {}
            for kind, grid in MODELS:
                try:
                    pm[kind] = base_preds(tr, te, feats, h, kind, grid)
                except Exception:
                    pm[kind] = np.full(len(te), np.nan)
            pm = pd.DataFrame(pm)
            if pm.isna().all().any():
                continue
            p = combine(pm, agg)
            for c, y, pi in zip(te.country_name.values, te[f"y{h}"].values, p):
                recs.append({"origin": T, "country": c, "y": int(y), "p": float(pi)})
        r = pd.DataFrame(recs)
        if r.empty or r.y.sum() < 5:
            continue
        auc = roc_auc_score(r.y, r.p); ap = average_precision_score(r.y, r.p)
        base = r.y.mean()
        tp = fp = fn = 0
        for T, g in r.groupby("origin"):
            top = g.sort_values("p", ascending=False).head(10)
            tp += int(top.y.sum()); fp += int((top.y == 0).sum())
            fn += int(g.y.sum() - top.y.sum())
        prec10 = tp / (tp + fp) if tp + fp else np.nan
        rec10 = tp / (tp + fn) if tp + fn else np.nan
        from collections import Counter
        rows.append({"h": h, "n": len(r), "n_pos": int(r.y.sum()),
                     "base_rate": round(base, 4), "auc_roc": round(auc, 4),
                     "auc_pr": round(ap, 4), "ap_lift": round(ap / base, 2),
                     "prec_top10": round(prec10, 4), "recall_top10": round(rec10, 4),
                     "lift_top10": round(prec10 / base, 2),
                     "aggs_chosen": dict(Counter(chosen))})
        print(f"h={h}  n={len(r):>4} pos={int(r.y.sum()):>3} base={base:.3f}  "
              f"AUC={auc:.4f}  AP={ap:.4f} ({ap/base:.2f}x)  "
              f"top10 prec={prec10:.3f} rec={rec10:.3f} ({prec10/base:.2f}x)  "
              f"aggs={dict(Counter(chosen))}")
        r.to_csv(os.path.join(OUT, f"onset_preds_h{h}.csv"), index=False)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "onset_horizon_sweep.csv"), index=False)
    print("\nWrote onset_horizon_sweep.csv")


if __name__ == "__main__":
    main()
