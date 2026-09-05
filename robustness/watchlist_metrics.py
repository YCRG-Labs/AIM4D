"""
What an annual watchlist built from the strict-design forecasts would look like.

Precision at 25 in the strict table pools country-years across the whole
evaluation period, so it is not the precision of a list an analyst would receive
each year, and several of its rows can belong to one onset. This reports the
quantities a user of such a list would care about: precision within each year's
list, how many distinct onsets are caught at least once, how far ahead they are
caught, and how many alerts are issued per onset detected.

Outputs robustness/watchlist_metrics.csv.
"""

import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onset_forecast_clean import build_panel, label_h, EXCLUDE_COLS
from baselines_clean_design import add_pitf

OUT = os.path.dirname(os.path.abspath(__file__))
H = 5
LAST_OBS = 2025
ORIGINS = list(range(2005, LAST_OBS - H + 1))
DEPTHS = [5, 10, 20]


def predictions(d, feats, mk, name):
    rows = []
    for T in ORIGINS:
        tr = d[(d.year <= T - H) & d.at_risk]
        te = d[(d.year == T) & d.at_risk]
        if len(te) == 0 or tr[f"y{H}"].sum() < 5:
            continue
        sc = StandardScaler()
        X = sc.fit_transform(tr[feats].fillna(0).values)
        Xt = sc.transform(te[feats].fillna(0).values)
        m = mk(); m.fit(X, tr[f"y{H}"].values)
        rows.append(pd.DataFrame({"year": T, "country_name": te.country_name.values,
                                  "y": te[f"y{H}"].values, "onset_year": te.onset_year.values,
                                  "p": m.predict_proba(Xt)[:, 1], "learner": name}))
    return pd.concat(rows, ignore_index=True)


def blend(parts):
    b = parts[0][["year", "country_name", "y", "onset_year"]].copy()
    acc = np.zeros(len(b))
    for pt in parts:
        m = b.merge(pt, on=["year", "country_name", "y", "onset_year"], how="left")
        v = m["p"].fillna(m["p"].median()).values
        r = np.zeros(len(v))
        for yr in np.unique(m["year"].values):
            sel = m["year"].values == yr
            r[sel] = rankdata(v[sel]) / sel.sum()
        acc += r
    b["p"] = acc / len(parts)
    return b


def report(b, tag, rows):
    # An episode is a (country, onset year) pair. Counting onset years alone
    # merges distinct countries that happen to decline in the same year.
    pos = b[b.y == 1]
    tot_onsets = len(set(zip(pos.country_name, pos.onset_year)))
    for k in DEPTHS:
        precs, flagged, leads, alerts = [], set(), [], 0
        for yr, g in b.groupby("year"):
            top = g.nlargest(min(k, len(g)), "p")
            alerts += len(top)
            precs.append(top.y.mean())
            for _, r in top[top.y == 1].iterrows():
                flagged.add((r.country_name, r.onset_year))
                leads.append(r.onset_year - yr)
        rows.append({"model": tag, "depth": k,
                     "mean_annual_precision": round(float(np.mean(precs)), 3),
                     "onsets_detected": len(flagged), "onsets_total": int(tot_onsets),
                     "detection_rate": round(len(flagged) / max(tot_onsets, 1), 3),
                     "median_lead_yrs": float(np.median(leads)) if leads else np.nan,
                     "alerts_issued": alerts,
                     "alerts_per_onset_detected": round(alerts / max(len(flagged), 1), 1)})
        r = rows[-1]
        print(f"  {tag:<26} top-{k:<3} annual precision {r['mean_annual_precision']:.3f}  "
              f"onsets {r['onsets_detected']}/{r['onsets_total']} ({r['detection_rate']:.0%})  "
              f"median lead {r['median_lead_yrs']}y  {r['alerts_per_onset_detected']} alerts per onset")


def main():
    d = add_pitf(build_panel())
    d[f"y{H}"] = label_h(d, H)
    feats = [c for c in d.columns if c not in EXCLUDE_COLS and d[c].dtype != object
             and c != "v2x_regime" and not re.fullmatch(r"y\d+", c)]
    poly = [c for c in ["v2x_polyarchy", "poly_d1", "poly_d3", "poly_rm5"] if c in d.columns]
    print(f"origins {ORIGINS[0]}--{ORIGINS[-1]}, {len(feats)} features\n")

    MK = {"gb": lambda: GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                   subsample=0.8, min_samples_leaf=10, random_state=0),
          "rf": lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                               class_weight="balanced", random_state=0, n_jobs=-1),
          "lr": lambda: LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.1,
                                           max_iter=3000, class_weight="balanced", random_state=0)}
    rows = []
    fw = blend([predictions(d, feats, MK[k], k) for k in MK])
    report(fw, "Five-stage framework", rows)
    p4 = predictions(d, poly, lambda: LogisticRegression(max_iter=2000, class_weight="balanced"), "lr")
    report(p4.assign(p=p4.p), "Four polyarchy variables", rows)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "watchlist_metrics.csv"), index=False)
    print("\nWrote watchlist_metrics.csv")


if __name__ == "__main__":
    main()
