"""
Does the pipeline add anything on top of polyarchy trend?

A head-to-head cannot answer this. A four-variable polyarchy model reaching
AUC 0.719 against the full pipeline's 0.729 shows the simple model captures
most of the recoverable signal; it does not show the pipeline is redundant,
because two models can score alike while ranking different countries correctly.
What settles it is whether pipeline information adds to the simple model when
both are available, and whether it adds at the top of the list where an
operator actually acts.

Four comparisons, all under the locked protocol (democratic at-risk pool,
predictors dated t or earlier, rolling origins, nothing tuned on the pooled
predictions):
  simple      polyarchy level, 1yr and 3yr change, 5yr rolling mean
  pipeline    the engineered Stage-5 feature matrix
  combined    both feature sets in one model
  blend       rank-average of the simple and pipeline scores

Reports AUC, average precision, precision at the top of the annual list, and
the correlation between the two models' rankings, which is the direct evidence
on whether they are seeing the same thing.

Outputs robustness/parsimony_complementarity.csv.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from onset_forecast_clean import build_panel, label_h, EXCLUDE_COLS, ORIGINS

OUT = os.path.dirname(os.path.abspath(__file__))
H = 5
N_BOOT = 2000
RNG = np.random.default_rng(20260904)


def add_simple(d):
    v = pd.read_csv(os.path.join(OUT, "..", "data", "vdem_v16.csv"), low_memory=False,
                    usecols=["country_name", "year", "v2x_polyarchy"])
    d = d.merge(v, on=["country_name", "year"], how="left")
    d = d.sort_values(["country_name", "year"])
    g = d.groupby("country_name", group_keys=False)
    d["poly_d1"] = g["v2x_polyarchy"].diff(1)
    d["poly_d3"] = g["v2x_polyarchy"].diff(3)
    d["poly_rm5"] = g["v2x_polyarchy"].transform(lambda x: x.rolling(5, min_periods=2).mean())
    return d


SIMPLE = ["v2x_polyarchy", "poly_d1", "poly_d3", "poly_rm5"]


def roll(d, feats, h, kind):
    recs = []
    for T in ORIGINS:
        tr = d[(d.year <= T - h) & d.at_risk]
        te = d[(d.year == T) & d.at_risk]
        if len(te) == 0 or tr[f"y{h}"].sum() < 8:
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr[feats].fillna(0).values)
        Xte = sc.transform(te[feats].fillna(0).values)
        m = (LogisticRegression(max_iter=3000, class_weight="balanced")
             if kind == "lr" else
             RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                    class_weight="balanced", random_state=0, n_jobs=-1))
        try:
            m.fit(Xtr, tr[f"y{h}"].values)
            p = m.predict_proba(Xte)[:, 1]
        except Exception:
            continue
        for c, y, pi in zip(te.country_name.values, te[f"y{h}"].values, p):
            recs.append({"origin": T, "country": c, "y": int(y), "p": float(pi)})
    return pd.DataFrame(recs)


def prec_at(df, k):
    tp = fp = 0
    for T, g in df.groupby("origin"):
        top = g.sort_values("p", ascending=False).head(k)
        tp += int(top.y.sum()); fp += int((top.y == 0).sum())
    return tp / (tp + fp) if tp + fp else np.nan


def paired_delta(y, pa, pb, countries):
    uniq = np.unique(countries)
    idx = {c: np.where(countries == c)[0] for c in uniq}
    da, dp = [], []
    for _ in range(N_BOOT):
        draw = RNG.choice(uniq, size=len(uniq), replace=True)
        i = np.concatenate([idx[c] for c in draw])
        if y[i].sum() < 3 or y[i].sum() == len(i):
            continue
        try:
            da.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
            dp.append(average_precision_score(y[i], pa[i]) - average_precision_score(y[i], pb[i]))
        except Exception:
            continue
    f = lambda a: (float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return f(da), f(dp)


def main():
    d = build_panel()
    d = add_simple(d)
    d[f"y{H}"] = label_h(d, H)
    drop = set(EXCLUDE_COLS) | {"v2x_regime", "onset_year", "ep_end", "in_episode",
                                "at_risk", f"y{H}"}
    pipe = [c for c in d.columns if c not in drop and d[c].dtype != object
            and c not in SIMPLE]
    print(f"simple {len(SIMPLE)} features | pipeline {len(pipe)} features\n")

    runs = {
        "simple (4 polyarchy vars)": roll(d, SIMPLE, H, "lr"),
        "pipeline (engineered)": roll(d, pipe, H, "rf"),
        "combined (simple + pipeline)": roll(d, SIMPLE + pipe, H, "rf"),
    }
    key = ["origin", "country", "y"]
    merged = None
    rows = []
    for name, df in runs.items():
        if df.empty:
            continue
        a = roc_auc_score(df.y, df.p); ap = average_precision_score(df.y, df.p)
        rows.append({"model": name, "auc_roc": round(a, 4), "auc_pr": round(ap, 4),
                     "prec_top5": round(prec_at(df, 5), 4), "prec_top10": round(prec_at(df, 10), 4),
                     "n": len(df), "n_pos": int(df.y.sum())})
        print(f"  {name:<32} AUC={a:.4f} AP={ap:.4f} "
              f"p@5={prec_at(df,5):.3f} p@10={prec_at(df,10):.3f}")
        m = df.rename(columns={"p": name})
        merged = m if merged is None else merged.merge(m, on=key, how="inner")

    s_col, p_col = "simple (4 polyarchy vars)", "pipeline (engineered)"
    merged["blend (rank-avg)"] = (merged.groupby("origin")[s_col].rank(pct=True)
                                  + merged.groupby("origin")[p_col].rank(pct=True)) / 2
    bl = merged[key + ["blend (rank-avg)"]].rename(columns={"blend (rank-avg)": "p"})
    a = roc_auc_score(bl.y, bl.p); ap = average_precision_score(bl.y, bl.p)
    rows.append({"model": "blend (rank-avg)", "auc_roc": round(a, 4), "auc_pr": round(ap, 4),
                 "prec_top5": round(prec_at(bl, 5), 4), "prec_top10": round(prec_at(bl, 10), 4),
                 "n": len(bl), "n_pos": int(bl.y.sum())})
    print(f"  {'blend (rank-avg)':<32} AUC={a:.4f} AP={ap:.4f} "
          f"p@5={prec_at(bl,5):.3f} p@10={prec_at(bl,10):.3f}")

    rho = spearmanr(merged[s_col], merged[p_col]).statistic
    print(f"\n  Spearman correlation between simple and pipeline rankings: {rho:.3f}")

    y = merged.y.values; cty = merged.country.values
    print("\n  paired country-clustered bootstrap, vs the simple model:")
    for other in ["combined (simple + pipeline)", "blend (rank-avg)"]:
        col = other if other in merged.columns else None
        if col is None:
            continue
        (ma, la, ha), (mp, lp, hp) = paired_delta(y, merged[col].values, merged[s_col].values, cty)
        print(f"    {other:<30} dAUC {ma:+.4f} [{la:+.4f},{ha:+.4f}]   dAP {mp:+.4f} [{lp:+.4f},{hp:+.4f}]")
        rows.append({"model": f"delta: {other} minus simple", "auc_roc": round(ma, 4),
                     "auc_pr": round(mp, 4), "prec_top5": np.nan, "prec_top10": np.nan,
                     "n": len(merged), "n_pos": int(y.sum())})

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "parsimony_complementarity.csv"), index=False)
    print("\nWrote parsimony_complementarity.csv")


if __name__ == "__main__":
    main()
