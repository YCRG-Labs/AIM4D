"""
Re-run the baseline ladder under the corrected design.

Table 5 compares the framework against persistence, a PITF-style logit, an
elastic net, a gradient booster on raw indicators and a V-Forecast-style
ensemble, all on the old protocol: contemporaneous features, pre-onset-window
labels, and a risk set that included countries already autocratizing. Those
numbers are not comparable to anything measured on the corrected design.

The paper's actual claim is relative, not absolute. It says the framework
matches strong baselines and earns its place through interpretable
decomposition rather than accuracy. That claim survives a fall in absolute
performance provided the baselines fall with it. This script tests exactly
that, putting every baseline on the same clean at-risk pool, the same
h-step-ahead target, and the same rolling origins.

Outputs robustness/baselines_clean_design.csv.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from onset_forecast_clean import build_panel, label_h, EXCLUDE_COLS, ORIGINS

OUT = os.path.dirname(os.path.abspath(__file__))
H = 5

PITF_VARS = ["v2x_polyarchy", "v2x_regime"]


def add_pitf(d):
    v = pd.read_csv(os.path.join(OUT, "..", "data", "vdem_v16.csv"), low_memory=False,
                    usecols=["country_name", "year", "v2x_polyarchy"])
    d = d.merge(v, on=["country_name", "year"], how="left", suffixes=("", "_p"))
    d = d.sort_values(["country_name", "year"])
    g = d.groupby("country_name", group_keys=False)
    d["poly_d3"] = g["v2x_polyarchy"].diff(3)
    d["poly_d1"] = g["v2x_polyarchy"].diff(1)
    d["poly_rm5"] = g["v2x_polyarchy"].transform(lambda x: x.rolling(5, min_periods=2).mean())
    return d


def rolling(d, feats, h, make_model, use_scaler=True):
    recs = []
    for T in ORIGINS:
        tr = d[(d.year <= T - h) & d.at_risk]
        te = d[(d.year == T) & d.at_risk]
        if len(te) == 0 or tr[f"y{h}"].sum() < 8:
            continue
        Xtr_raw = tr[feats].fillna(0).values
        Xte_raw = te[feats].fillna(0).values
        if use_scaler:
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr_raw); Xte = sc.transform(Xte_raw)
        else:
            Xtr, Xte = Xtr_raw, Xte_raw
        try:
            m = make_model()
            m.fit(Xtr, tr[f"y{h}"].values)
            p = m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba") else m.predict(Xte)
        except Exception:
            continue
        for c, y, pi in zip(te.country_name.values, te[f"y{h}"].values, p):
            recs.append({"origin": T, "country": c, "y": int(y), "p": float(pi)})
    return pd.DataFrame(recs)


def persistence(d, h):
    """Trend extrapolation: recent polyarchy decline as the only signal."""
    recs = []
    for T in ORIGINS:
        te = d[(d.year == T) & d.at_risk]
        if len(te) == 0:
            continue
        s = (-te["poly_d3"].fillna(0)).values
        for c, y, pi in zip(te.country_name.values, te[f"y{h}"].values, s):
            recs.append({"origin": T, "country": c, "y": int(y), "p": float(pi)})
    return pd.DataFrame(recs)


def report(df, name, rows):
    if df.empty or df.y.sum() < 5:
        print(f"  {name:<40} insufficient")
        return
    a = roc_auc_score(df.y, df.p); ap = average_precision_score(df.y, df.p)
    base = df.y.mean()
    print(f"  {name:<40} AUC={a:.4f}  AP={ap:.4f}  ({ap/base:.2f}x)  n={len(df)} pos={int(df.y.sum())}")
    rows.append({"model": name, "auc_roc": round(a, 4), "auc_pr": round(ap, 4),
                 "ap_lift": round(ap / base, 2), "n": len(df), "n_pos": int(df.y.sum()),
                 "base_rate": round(base, 4)})


def main():
    d = build_panel()
    d = add_pitf(d)
    d[f"y{H}"] = label_h(d, H)
    drop = set(EXCLUDE_COLS) | {"v2x_regime", "onset_year", "ep_end", "in_episode",
                                "at_risk", f"y{H}"}
    allf = [c for c in d.columns if c not in drop and d[c].dtype != object]
    pitf = [c for c in ["v2x_polyarchy", "poly_d1", "poly_d3", "poly_rm5"] if c in d.columns]

    print(f"clean design, h={H}, at-risk rows {int(d.at_risk.sum())}, "
          f"positives {int(d.loc[d.at_risk, f'y{H}'].sum())}\n")
    rows = []

    report(persistence(d, H), "Persistence (3yr polyarchy decline)", rows)
    report(rolling(d, pitf, H, lambda: LogisticRegression(max_iter=2000, class_weight="balanced")),
           "PITF-style logit (parsimonious)", rows)
    report(rolling(d, allf, H, lambda: LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.1, max_iter=3000,
        class_weight="balanced")), "Elastic net (all features)", rows)
    report(rolling(d, allf, H, lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
        min_samples_leaf=10, random_state=0)), "Gradient boosting (all features)", rows)
    report(rolling(d, allf, H, lambda: RandomForestClassifier(
        n_estimators=400, min_samples_leaf=3, class_weight="balanced",
        random_state=0, n_jobs=-1)), "Random forest (V-Forecast-style ensemble)", rows)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "baselines_clean_design.csv"), index=False)
    print("\nWrote baselines_clean_design.csv")


if __name__ == "__main__":
    main()
