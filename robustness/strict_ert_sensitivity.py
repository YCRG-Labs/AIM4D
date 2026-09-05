"""
The strict comparison under the unmodified ERT outcome definition.

The paper's episode set is a hand-maintained dictionary keyed by country. It
carries at most one episode per country, so of the 110 autocratization episodes
ERT v16 records with an onset from 1996 on, 26 belonging to the 20 countries with
more than one cannot be represented at all, and a recurrent onset after an
earlier episode ends is unlabelled even though the risk set lets the country back
in.

This rebuilds the outcome from the published ERT file with every episode kept,
labels a country-year positive when any onset falls in the forecast window, and
treats a country as in-episode whenever any episode interval covers the year.
It then reruns the strict comparison so the two outcome definitions can be read
side by side.

Outputs robustness/strict_ert_sensitivity.csv.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onset_forecast_clean import EXCLUDE_COLS
from baselines_clean_design import add_pitf
from episode_ledger import ert_episodes

OUT = os.path.dirname(os.path.abspath(__file__))
H = 5
LAST_OBS = 2025
ORIGINS = list(range(2005, LAST_OBS - H + 1))


def build_panel_ert():
    d = pd.read_csv(os.path.join(OUT, "..", "stage5_ews", "ews_signals.csv"))
    v = pd.read_csv(os.path.join(OUT, "..", "data", "vdem_v16.csv"), low_memory=False,
                    usecols=["country_name", "year", "v2x_regime"]).dropna(subset=["v2x_regime"])
    v["v2x_regime"] = v["v2x_regime"].astype(int)
    d = d.merge(v, on=["country_name", "year"], how="left")

    ep = ert_episodes()
    spans, onsets = {}, {}
    for _, r in ep.iterrows():
        spans.setdefault(r["country_name"], []).append((int(r["onset"]), int(r["end"])))
        onsets.setdefault(r["country_name"], []).append(int(r["onset"]))

    in_ep, y = np.zeros(len(d), bool), np.zeros(len(d), int)
    cn, yr = d["country_name"].values, d["year"].values
    for i in range(len(d)):
        for a, b in spans.get(cn[i], ()):
            if a <= yr[i] <= b:
                in_ep[i] = True
        for a in onsets.get(cn[i], ()):
            if yr[i] + 1 <= a <= yr[i] + H:
                y[i] = 1
    d["in_episode"] = in_ep
    d["at_risk"] = (d["v2x_regime"] >= 2) & (~d["in_episode"])
    d[f"y{H}"] = y
    d["n_episodes"] = [len(spans.get(c, ())) for c in cn]
    return d, ep


def rolling(d, feats, mk):
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
                                  "y": te[f"y{H}"].values, "p": m.predict_proba(Xt)[:, 1]}))
    return pd.concat(rows, ignore_index=True)


def main():
    d, ep = build_panel_ert()
    d = add_pitf(d)
    feats = [c for c in d.columns if c not in EXCLUDE_COLS and d[c].dtype != object
             and c not in ("v2x_regime", "n_episodes") and not re.fullmatch(r"y\d+", c)]
    poly = [c for c in ["v2x_polyarchy", "poly_d1", "poly_d3", "poly_rm5"] if c in d.columns]

    ar = d[d.at_risk]
    npos = int(d.loc[d.at_risk, f"y{H}"].sum())
    print(f"ERT v16 outcome, all {len(ep)} episodes kept")
    print(f"at-risk country-years {len(ar)}, positives {npos}, "
          f"base rate {npos/len(ar):.3f}, {len(feats)} features\n")

    MK = {"gb": lambda: GradientBoostingClassifier(n_estimators=200, max_depth=3,
              learning_rate=0.05, subsample=0.8, min_samples_leaf=10, random_state=0),
          "rf": lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
              class_weight="balanced", random_state=0, n_jobs=-1),
          "lr": lambda: LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
              C=0.1, max_iter=3000, class_weight="balanced", random_state=0)}
    parts = {k: rolling(d, feats, MK[k]) for k in MK}
    p4 = rolling(d, poly, lambda: LogisticRegression(max_iter=2000, class_weight="balanced"))

    b = parts["gb"][["year", "country_name", "y"]].copy()
    acc = np.zeros(len(b))
    for pt in parts.values():
        m = b.merge(pt, on=["year", "country_name", "y"], how="left")
        v = m["p"].fillna(m["p"].median()).values
        r = np.zeros(len(v))
        for yr in np.unique(m["year"].values):
            s = m["year"].values == yr
            r[s] = rankdata(v[s]) / s.sum()
        acc += r
    b["p"] = acc / len(parts)

    rows = []
    for name, df in [("Four polyarchy variables", p4), ("Five-stage framework", b),
                     ("Gradient boosting", parts["gb"]), ("Random forest", parts["rf"]),
                     ("Elastic net", parts["lr"])]:
        rows.append({"model": name, "n": len(df), "n_pos": int(df.y.sum()),
                     "base_rate": round(df.y.mean(), 4),
                     "auc_roc": round(roc_auc_score(df.y, df.p), 4),
                     "auc_pr": round(average_precision_score(df.y, df.p), 4)})
        r = rows[-1]
        print(f"  {name:<26} AUC {r['auc_roc']:.3f}  AP {r['auc_pr']:.3f}  "
              f"n={r['n']} pos={r['n_pos']} base={r['base_rate']:.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "strict_ert_sensitivity.csv"), index=False)

    # Row-level predictions, so the paired intervals below can be reproduced.
    preds = b.rename(columns={"p": "p_framework"}).merge(
        p4.rename(columns={"p": "p_poly4"}), on=["year", "country_name", "y"])
    preds.to_csv(os.path.join(OUT, "strict_ert_sensitivity_predictions.csv"), index=False)

    def paired(df, a, c, seed=20260905, n_boot=2000):
        rng = np.random.default_rng(seed)
        u = df["country_name"].unique()
        idx = {k: np.where(df["country_name"].values == k)[0] for k in u}
        da, dp = [], []
        for _ in range(n_boot):
            j = np.concatenate([idx[k] for k in rng.choice(u, len(u), replace=True)])
            if df["y"].values[j].sum() < 3:
                continue
            da.append(roc_auc_score(df["y"].values[j], df[a].values[j])
                      - roc_auc_score(df["y"].values[j], df[c].values[j]))
            dp.append(average_precision_score(df["y"].values[j], df[a].values[j])
                      - average_precision_score(df["y"].values[j], df[c].values[j]))
        f = lambda v: (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5))
        return f(da), f(dp)

    (ma, la, ha), (mp, lp, hp) = paired(preds, "p_framework", "p_poly4")
    print(f"\nframework minus four variables, all origins ({preds.year.min()}--{preds.year.max()}):")
    print(f"  dAUC {ma:+.3f} [{la:+.3f}, {ha:+.3f}]   dAP {mp:+.3f} [{lp:+.3f}, {hp:+.3f}]")

    # The paper's episode set skips 2005-2007 under the five-positive training rule,
    # so the two outcome definitions are not scored on the same origins. Restricting
    # to the common window separates the outcome change from the sample change.
    common = preds[preds.year >= 2008]
    (ma2, la2, ha2), (mp2, lp2, hp2) = paired(common, "p_framework", "p_poly4")
    print(f"common origins 2008--2020 only, n={len(common)} pos={int(common.y.sum())}:")
    print(f"  dAUC {ma2:+.3f} [{la2:+.3f}, {ha2:+.3f}]   dAP {mp2:+.3f} [{lp2:+.3f}, {hp2:+.3f}]")
    print("\nNote: this reuses the existing upstream EWS features and reruns only the")
    print("downstream learners. It does not rebuild the pipeline at each origin.")
    print("\nWrote strict_ert_sensitivity.csv and strict_ert_sensitivity_predictions.csv")


if __name__ == "__main__":
    main()
