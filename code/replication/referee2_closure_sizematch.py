"""Referee 2, round 1: does the closure effect survive matching training-set size?

closure_placebo.py shows the closure gap grows under permuted onsets, which
rules out foresight about the onset process but cannot separate two remaining
explanations: (b) label memorization across the window boundary, and (c) the
closed arm simply training on four fewer years of rows. Both survive
permutation. This adds a third arm, the open training set subsampled at each
origin to exactly the closed arm's row count, with identical seeds in every
arm. If closed minus open_matched is close to closed minus open, the gap is
(b); if it collapses toward zero, it is (c).

Reads author code only; writes nothing under the author's directories.
"""
import os, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
ROB = "/Users/jacobcrainic/AIM4D/robustness"
sys.path.insert(0, "/Users/jacobcrainic/AIM4D"); sys.path.insert(0, ROB)
from onset_forecast_clean import build_panel, EXCLUDE_COLS
from closure_placebo import label_from, learner, H, ORIGINS

def arms(d, feats, name, seed):
    pool = d[d.at_risk]
    ys = {k: [] for k in ("closed", "open", "open_matched")}; ps = {k: [] for k in ys}
    rng = np.random.default_rng(1000 + seed)
    for T in ORIGINS:
        te = pool[pool.year == T]
        tr_c = pool[pool.year <= T - H]
        tr_o = pool[pool.year <= T - 1]
        if len(te) == 0 or tr_c["y"].sum() < 5 or te["y"].sum() < 1:
            continue
        tr_m = tr_o.iloc[rng.choice(len(tr_o), size=len(tr_c), replace=False)]
        if tr_m["y"].sum() < 5:
            continue
        for k, tr in (("closed", tr_c), ("open", tr_o), ("open_matched", tr_m)):
            sc = StandardScaler(); X = sc.fit_transform(tr[feats].fillna(0).values)
            m = learner(name, seed); m.fit(X, tr["y"].values)
            ps[k].append(m.predict_proba(sc.transform(te[feats].fillna(0).values))[:, 1]); ys[k].append(te["y"].values)
    return {k: roc_auc_score(np.concatenate(ys[k]), np.concatenate(ps[k])) for k in ys}

base = build_panel()
feats = [c for c in base.columns if c not in EXCLUDE_COLS and base[c].dtype != object
         and c != "v2x_regime" and not re.fullmatch(r"y\d+", c)]
real_map = base.dropna(subset=["onset_year"]).groupby("country_name")["onset_year"].first().to_dict()
d = base.copy(); d["y"] = label_from(real_map, d)
rows = []
for name in ("gb", "rf", "lr"):
    for seed in (0, 1, 2):
        a = arms(d, feats, name, seed); a.update(learner=name, seed=seed); rows.append(a)
        print(f"{name} s{seed}  closed {a['closed']:.3f}  open {a['open']:.3f}  open_matched {a['open_matched']:.3f}", flush=True)
r = pd.DataFrame(rows)
r["closed_minus_open"] = r.closed - r.open
r["closed_minus_open_matched"] = r.closed - r.open_matched
r["open_matched_minus_open"] = r.open_matched - r.open
r.to_csv(os.path.join(os.path.dirname(__file__), "referee2_closure_sizematch.csv"), index=False)
print("\nmean over 9 (learner,seed):")
print(f"  closed - open          = {r.closed_minus_open.mean():+.4f}   (closure_placebo observed: -0.190)")
print(f"  closed - open_matched  = {r.closed_minus_open_matched.mean():+.4f}   (label memorization share)")
print(f"  open_matched - open    = {r.open_matched_minus_open.mean():+.4f}   (sample-size share)")
print(f"  share of gap explained by memorization: {r.closed_minus_open_matched.mean()/r.closed_minus_open.mean():.0%}")
