"""
Placebo test for the closure effect.

Table~\ref{tab:factorial} attributes a 0.224 AUC drop to requiring training
outcome windows to close before the forecast origin, and the paper reads that as
information leakage. An alternative reading is that the open and closed training
arms differ in sample size and composition, and that the gap reflects those
differences rather than leakage. If the mechanism is leakage, then destroying the
link between predictors and onsets should destroy the effect: under permuted
onset labels there is no future information to leak, so the closure contrast
should collapse toward zero.

Onsets are permuted across countries, preserving the number of episodes and each
country's at-risk structure, and the closure contrast is recomputed. The real
contrast is reported alongside for reference.

Outputs robustness/closure_placebo.csv.
"""

import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onset_forecast_clean import build_panel, EXCLUDE_COLS

OUT = os.path.dirname(os.path.abspath(__file__))
H, LAST_OBS = 5, 2025
ORIGINS = list(range(2005, LAST_OBS - H + 1))
N_PERM = int(os.environ.get("AIM4D_PLACEBO_PERM", "12"))


def learner(name, seed):
    if name == "gb":
        return GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                          subsample=0.8, min_samples_leaf=20, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                      random_state=seed, n_jobs=-1)
    return LogisticRegression(C=1.0, max_iter=2000, random_state=seed)


def closure_contrast(d, feats, name, seed):
    """AUC with closure enforced minus AUC with open training windows."""
    out = {}
    for closure in (True, False):
        lag = H if closure else 1
        ys, ps = [], []
        for T in ORIGINS:
            tr = d[(d.year <= T - lag) & d.at_risk]
            te = d[(d.year == T) & d.at_risk]
            if len(te) == 0 or tr["y"].sum() < 5 or te["y"].sum() < 1:
                continue
            sc = StandardScaler()
            X = sc.fit_transform(tr[feats].fillna(0).values)
            Xt = sc.transform(te[feats].fillna(0).values)
            m = learner(name, seed)
            try:
                m.fit(X, tr["y"].values)
                ps.append(m.predict_proba(Xt)[:, 1])
                ys.append(te["y"].values)
            except Exception:
                continue
        if not ys:
            return None
        y, p = np.concatenate(ys), np.concatenate(ps)
        if y.sum() == 0 or y.sum() == len(y):
            return None
        out[closure] = roc_auc_score(y, p)
    return out[True] - out[False]


def label_from(onset_map, d):
    on = d["country_name"].map(onset_map)
    fut = on - d["year"]
    return ((fut >= 1) & (fut <= H)).astype(int)


def main():
    base = build_panel()
    feats = [c for c in base.columns if c not in EXCLUDE_COLS and base[c].dtype != object
             and c != "v2x_regime" and not re.fullmatch(r"y\d+", c)]
    real_map = base.dropna(subset=["onset_year"]).groupby("country_name")["onset_year"].first().to_dict()
    countries = sorted(base["country_name"].unique())
    print(f"{len(feats)} features, {len(real_map)} countries with an onset, "
          f"origins {ORIGINS[0]}--{ORIGINS[-1]}\n")

    rows = []
    d = base.copy()
    d["y"] = label_from(real_map, d)
    real = [closure_contrast(d, feats, n, s) for n in ("gb", "rf", "lr") for s in (0, 1)]
    real = [r for r in real if r is not None]
    print(f"observed closure contrast (enforced minus open): {np.mean(real):+.4f}")
    rows.append({"kind": "observed", "perm": -1, "mean_contrast": round(float(np.mean(real)), 4)})

    rng = np.random.default_rng(20260905)
    placebo = []
    for b in range(N_PERM):
        shuffled = list(real_map.values())
        rng.shuffle(shuffled)
        targets = rng.choice(countries, size=len(shuffled), replace=False)
        pmap = dict(zip(targets, shuffled))
        dp = base.copy()
        dp["y"] = label_from(pmap, dp)
        if dp.loc[dp.at_risk, "y"].sum() < 20:
            continue
        vals = [closure_contrast(dp, feats, n, 0) for n in ("gb", "rf", "lr")]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        m = float(np.mean(vals))
        placebo.append(m)
        rows.append({"kind": "placebo", "perm": b, "mean_contrast": round(m, 4)})
        print(f"  permutation {b+1:>2}: closure contrast {m:+.4f}")

    pl = np.array(placebo)
    print(f"\nplacebo mean {pl.mean():+.4f}   sd {pl.std():.4f}   "
          f"range [{pl.min():+.4f}, {pl.max():+.4f}]   n={len(pl)}")
    print(f"observed {np.mean(real):+.4f} lies "
          f"{(np.mean(real)-pl.mean())/max(pl.std(),1e-9):+.1f} placebo SDs from the placebo mean")
    print(f"share of permutations at least as negative as observed: "
          f"{float((pl <= np.mean(real)).mean()):.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "closure_placebo.csv"), index=False)
    print("\nWrote closure_placebo.csv")


if __name__ == "__main__":
    main()
