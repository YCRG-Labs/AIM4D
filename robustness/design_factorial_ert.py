"""The design factorial rerun on the unmodified ERT v16 outcome.

design_factorial.py varies the four evaluation-design choices on our
hand-maintained episode set. Because that set is country-keyed it cannot carry
recurrent onsets, and the framework-versus-parsimony ordering is known to turn
on the outcome definition. This reruns the identical experiment against the
published ERT release so the design marginals can be read off the outcome the
field would use.

Everything except the outcome is held fixed: the same feature matrix, the same
three learner families, the same three seeds, the same sixteen cells.

Outputs robustness/design_factorial_ert.csv.
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

from onset_forecast_clean import EXCLUDE_COLS, learner
from ert_panel import build_panel_ert, label_ert

OUT = os.path.dirname(os.path.abspath(__file__))
H = 5
T0 = 2019
LAST_OBS = 2025
ORIGINS = list(range(2005, LAST_OBS - H + 1))
SEEDS = [0, 1, 2]


def score(d, feats, name, seed, restrict, future_only, rolling, closure):
    d = d.copy()
    d.attrs["onsets"] = build_panel_ert.__wrapped_onsets__
    d["y"] = label_ert(d, H, future_only)
    pool = d[d.at_risk] if restrict else d
    lag = H if closure else (1 if rolling else 0)
    origins = ORIGINS if rolling else [T0]
    ys, ps = [], []
    for T in origins:
        tr = pool[pool.year <= T - lag]
        te = pool[pool.year == T] if rolling else pool[pool.year > T]
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
    d = build_panel_ert()
    build_panel_ert.__wrapped_onsets__ = d.attrs["onsets"]
    feats = [c for c in d.columns if c not in EXCLUDE_COLS and d[c].dtype != object
             and c not in ("v2x_regime", "n_episodes")
             and not (isinstance(c, str) and c.startswith("y") and c[1:].isdigit())]
    print(f"ERT v16 outcome; features held fixed at {len(feats)}\n")

    rows = []
    for restrict, future_only, rolling, closure in itertools.product([True, False], repeat=4):
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
    df.to_csv(os.path.join(OUT, "design_factorial_ert.csv"), index=False)

    # Marginals: average the paired comparisons in which only one dimension moves.
    keys = ["risk_set", "label", "origin", "closure"]
    print("\nmarginal effect of each dimension (paired comparisons):")
    marg = []
    for k in keys:
        others = [x for x in keys if x != k] + ["learner"]
        piv = df.pivot_table(index=others, columns=k, values="auc")
        if piv.shape[1] != 2:
            continue
        a, b = list(piv.columns)
        diff = (piv[b] - piv[a]).dropna()
        marg.append({"dimension": k, "from": a, "to": b, "n_pairs": len(diff),
                     "mean_dauc": round(float(diff.mean()), 4),
                     "min": round(float(diff.min()), 4), "max": round(float(diff.max()), 4)})
        print(f"  {k:<10}{a} -> {b:<16}mean {diff.mean():+.3f}  "
              f"range [{diff.min():+.3f}, {diff.max():+.3f}]  n={len(diff)}")
    pd.DataFrame(marg).to_csv(os.path.join(OUT, "design_factorial_ert_marginals.csv"), index=False)
    print("\nWrote design_factorial_ert.csv and design_factorial_ert_marginals.csv")


if __name__ == "__main__":
    main()
