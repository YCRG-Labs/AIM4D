"""
Which part of the evaluation design does the work.

The strict and conventional numbers differ in more than one respect at once, so
a reader cannot tell whether the fall in discrimination comes from the risk set,
the label timing, the origin structure, or the requirement that training outcome
windows close before the forecast is issued. This holds the feature matrix and
the learners fixed and varies the four design choices factorially, so each cell
differs from its neighbours in exactly one respect.

    risk    all country-years            vs at-risk democracies not in episode
    label   window including t (W_it)    vs onset strictly in t+1..t+h
    origin  single 2019 cutoff           vs rolling origins
    closure training windows may overlap vs training windows must close by T

Outputs robustness/design_factorial.csv.
"""

import itertools
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

from onset_forecast_clean import EXCLUDE_COLS, build_panel, learner

OUT = os.path.dirname(os.path.abspath(__file__))
H = 5
T0 = 2019
LAST_OBS = 2025
# An origin needs its whole outcome window inside the observed panel. With onsets
# observed through 2025 and h=5, origins past 2020 would score rows against
# outcomes that cannot yet have happened, coding them 0 by default.
ORIGINS = list(range(2005, LAST_OBS - H + 1))
SEEDS = [0, 1, 2]


def label(d, future_only):
    fut = d["onset_year"] - d["year"]
    lo = 1 if future_only else 0
    return ((fut >= lo) & (fut <= H)).astype(int)


def score(d, feats, name, seed, restrict, future_only, rolling, closure):
    d = d.copy()
    d["y"] = label(d, future_only)
    pool = d[d.at_risk] if restrict else d
    # Without closure the training window may still be open, but it must not
    # contain the rows being scored. Under rolling origins that means stopping
    # at T-1; under a fixed origin the test set is already disjoint at T+1.
    lag = H if closure else (1 if rolling else 0)
    origins = ORIGINS if rolling else [T0]
    ys, ps = [], []
    for T in origins:
        tr = pool[pool.year <= T - lag]
        te = pool[pool.year == T] if rolling else pool[pool.year > T]
        # Scored rows also need a closed outcome window, or a censored row is
        # counted as a negative.
        te = te[te.year <= LAST_OBS - H]
        if len(te) == 0 or tr["y"].sum() < 5 or te["y"].sum() < 1:
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr[feats].fillna(0).values)
        Xte = sc.transform(te[feats].fillna(0).values)
        m = learner(name, seed)
        try:
            m.fit(Xtr, tr["y"].values)
            ps.append(m.predict_proba(Xte)[:, 1])
        except Exception:
            continue
        ys.append(te["y"].values)
    if not ys:
        return None
    y, p = np.concatenate(ys), np.concatenate(ps)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return len(y), int(y.sum()), roc_auc_score(y, p), average_precision_score(y, p)


def main():
    d = build_panel()
    feats = [c for c in d.columns if c not in EXCLUDE_COLS and d[c].dtype != object
             and c != "v2x_regime"]
    print(f"features held fixed at {len(feats)}\n")
    print("note: within the at-risk pool the two label rules coincide, because\n"
          "the onset year itself is inside an episode and so never at risk.\n")

    rows = []
    grid = list(itertools.product([True, False], repeat=4))
    for restrict, future_only, rolling, closure in grid:
        for name in ["gb", "rf", "lr"]:
            got = [score(d, feats, name, s, restrict, future_only, rolling, closure)
                   for s in SEEDS]
            got = [g for g in got if g]
            if not got:
                continue
            rows.append({
                "risk_set": "at-risk" if restrict else "all",
                "label": "future-only" if future_only else "window",
                "origin": "rolling" if rolling else "fixed-2019",
                "closure": "enforced" if closure else "none",
                "learner": name,
                "n": got[0][0], "n_pos": got[0][1],
                "auc": round(float(np.mean([g[2] for g in got])), 4),
                "ap": round(float(np.mean([g[3] for g in got])), 4),
            })
            r = rows[-1]
            print(f"{r['risk_set']:<8}{r['label']:<13}{r['origin']:<12}{r['closure']:<10}"
                  f"{name:<4}n={r['n']:<6}pos={r['n_pos']:<5}AUC={r['auc']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "design_factorial.csv"), index=False)

    print("\n=== marginal effect of each design choice on AUC, averaged over the other three ===")
    base = df.groupby(["risk_set", "label", "origin", "closure", "learner"])["auc"].mean().reset_index()
    for dim, a, b in [("risk_set", "all", "at-risk"), ("label", "window", "future-only"),
                      ("origin", "fixed-2019", "rolling"), ("closure", "none", "enforced")]:
        others = [c for c in ["risk_set", "label", "origin", "closure", "learner"] if c != dim]
        pv = base.pivot_table(index=others, columns=dim, values="auc")
        if a in pv and b in pv:
            dd = (pv[b] - pv[a]).dropna()
            print(f"  {dim:<10}{a:>12} -> {b:<13}  mean dAUC {dd.mean():+.4f}   "
                  f"min {dd.min():+.4f}  max {dd.max():+.4f}  (n={len(dd)})")
    print("\nWrote design_factorial.csv")


if __name__ == "__main__":
    main()
