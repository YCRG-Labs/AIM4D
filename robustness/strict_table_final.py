"""
Final strict-design comparison table.

Three things this fixes relative to baselines_clean_design.py. Origins are
restricted to those whose outcome window closes inside the observed panel, so
no row is scored against an outcome that cannot yet have happened; with onsets
observed through 2025 that means origins through 2025-h. Every model is scored
on the same rows, taken as the intersection across models, so the columns are
comparable. And discrimination is reported with average precision and
precision-at-the-top alongside AUC, since at an 8 percent base rate ranking
alone does not describe how a watchlist would perform.

Outputs robustness/strict_table_final.csv.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onset_forecast_clean import build_panel, label_h, EXCLUDE_COLS
from baselines_clean_design import add_pitf

OUT = os.path.dirname(os.path.abspath(__file__))
H = int(os.environ.get("AIM4D_H", "5"))
LAST_OBS = 2025
ORIGINS = [t for t in range(2005, LAST_OBS - H + 1)]
N_BOOT = 2000
RNG = np.random.default_rng(20260905)


def rolling(d, feats, make_model, scale=True):
    rows = []
    for T in ORIGINS:
        tr = d[(d.year <= T - H) & d.at_risk]
        te = d[(d.year == T) & d.at_risk]
        if len(te) == 0 or tr[f"y{H}"].sum() < 5:
            continue
        X, Xt = tr[feats].fillna(0).values, te[feats].fillna(0).values
        if scale:
            sc = StandardScaler(); X = sc.fit_transform(X); Xt = sc.transform(Xt)
        m = make_model()
        try:
            m.fit(X, tr[f"y{H}"].values)
            p = m.predict_proba(Xt)[:, 1]
        except Exception:
            continue
        rows.append(pd.DataFrame({"country_name": te.country_name.values, "year": te.year.values,
                                  "y": te[f"y{H}"].values, "p": p}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def persistence(d):
    rows = []
    for T in ORIGINS:
        te = d[(d.year == T) & d.at_risk]
        if len(te) == 0 or "poly_d3" not in te.columns:
            continue
        rows.append(pd.DataFrame({"country_name": te.country_name.values, "year": te.year.values,
                                  "y": te[f"y{H}"].values, "p": -te["poly_d3"].fillna(0).values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def prec_at(y, p, k):
    if k > len(y):
        return np.nan
    return float(y[np.argsort(-p)[:k]].mean())


def metrics(df):
    y, p = df.y.values, df.p.values
    base = y.mean()
    ap = average_precision_score(y, p)
    return {"n": len(y), "n_pos": int(y.sum()), "base_rate": round(base, 4),
            "auc_roc": round(roc_auc_score(y, p), 4), "auc_pr": round(ap, 4),
            "ap_lift": round(ap / base, 2),
            "prec_top10": round(prec_at(y, p, 10), 3),
            "prec_top25": round(prec_at(y, p, 25), 3),
            "prec_top_decile": round(prec_at(y, p, max(1, len(y) // 10)), 3)}


def paired_ci(y, pa, pb, countries):
    uniq = np.unique(countries); idx = {c: np.where(countries == c)[0] for c in uniq}
    da, dp = [], []
    for _ in range(N_BOOT):
        draw = RNG.choice(uniq, size=len(uniq), replace=True)
        j = np.concatenate([idx[c] for c in draw])
        if y[j].sum() < 3 or y[j].sum() == len(j):
            continue
        da.append(roc_auc_score(y[j], pa[j]) - roc_auc_score(y[j], pb[j]))
        dp.append(average_precision_score(y[j], pa[j]) - average_precision_score(y[j], pb[j]))
    f = lambda a: (round(float(np.mean(a)), 4), round(float(np.percentile(a, 2.5)), 4),
                   round(float(np.percentile(a, 97.5)), 4))
    return f(da), f(dp)


def main():
    d = add_pitf(build_panel())
    d[f"y{H}"] = label_h(d, H)
    drop = set(EXCLUDE_COLS) | {"v2x_regime", f"y{H}"}
    allf = [c for c in d.columns if c not in drop and d[c].dtype != object]
    poly = [c for c in ["v2x_polyarchy", "poly_d1", "poly_d3", "poly_rm5"] if c in d.columns]

    print(f"h={H}  origins {ORIGINS[0]}..{ORIGINS[-1]} (outcome windows close by {LAST_OBS})")
    print(f"features {len(allf)}  at-risk rows {int(d.at_risk.sum())}\n")

    preds = {}
    preds["Persistence (3-yr polyarchy decline)"] = persistence(d)
    preds["Four polyarchy variables"] = rolling(d, poly, lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced"))
    preds["Elastic net, all features"] = rolling(d, allf, lambda: LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.1, max_iter=3000,
        class_weight="balanced"))
    preds["Gradient boosting, all features"] = rolling(d, allf, lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
        min_samples_leaf=10, random_state=0))
    preds["Random-forest ensemble"] = rolling(d, allf, lambda: RandomForestClassifier(
        n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=-1))

    # framework: rank-mean blend of the three learner families on the framework features
    parts = [preds[k] for k in ["Elastic net, all features", "Gradient boosting, all features",
                                "Random-forest ensemble"] if len(preds[k])]
    key = ["country_name", "year", "y"]
    blend = parts[0][key].copy()
    r = np.zeros(len(blend))
    for pt in parts:
        m = blend.merge(pt, on=key, how="left")
        r = r + rankdata(m["p"].fillna(m["p"].median()).values) / len(m)
    blend["p"] = r / len(parts)
    preds["Five-stage framework, rank-mean blend"] = blend

    common = None
    for v in preds.values():
        s = set(map(tuple, v[["country_name", "year"]].values))
        common = s if common is None else (common & s)
    print(f"common scored rows across all models: {len(common)}\n")

    rows = []
    aligned = {}
    for name, v in preds.items():
        v = v[[tuple(x) in common for x in v[["country_name", "year"]].values]]
        v = v.sort_values(["country_name", "year"]).reset_index(drop=True)
        aligned[name] = v
        rows.append({"model": name, **metrics(v)})
        r = rows[-1]
        print(f"{name:<40} AUC {r['auc_roc']:.3f}  AP {r['auc_pr']:.3f} "
              f"({r['ap_lift']}x)  p@10 {r['prec_top10']}  p@25 {r['prec_top25']}")

    fw = aligned["Five-stage framework, rank-mean blend"]
    pv = aligned["Four polyarchy variables"]
    (ma, la, ha), (mp, lp, hp) = paired_ci(fw.y.values, fw.p.values, pv.p.values,
                                           fw.country_name.values)
    print(f"\nframework minus four variables: dAUC {ma:+.3f} [{la:+.3f}, {ha:+.3f}]   "
          f"dAP {mp:+.3f} [{lp:+.3f}, {hp:+.3f}]")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"strict_table_final_h{H}.csv"), index=False)
    print(f"\nWrote strict_table_final_h{H}.csv")


if __name__ == "__main__":
    main()
