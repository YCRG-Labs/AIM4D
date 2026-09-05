"""
Fix calibration and put the alert tiers on a defensible operating point.

Two problems motivate this. The deployed risk scores have a calibration slope
near 3.4, meaning they are badly under-dispersed and cannot be read as
probabilities. And the alert tiers are percentiles of a training-negative
distribution, which fixes the false-alarm rate by construction and leaves the
watchlist burden unstated and uncontrolled.

Both have standard fixes. Calibration is repaired prequentially: at each
forecast origin the recalibration map is fitted only on predictions from
strictly earlier origins, so the corrected scores remain honestly out of
sample (Platt 1999; Zadrozny and Elkan 2002; Niculescu-Mizil and Caruana 2005;
Kull, Silva Filho and Flach 2017). Operating points are then set by a stated
annual alert budget rather than by a training percentile, and reported with
precision, recall and lift over the base rate, which is what an analyst acting
on a watchlist actually needs (Saito and Rehmsmeier 2015).

Runs on the clean at-risk panel: democratic country-years not already inside an
episode, with every predictor dated t or earlier and onset scored over t+1..t+h.

Outputs robustness/recalibration.csv and robustness/operating_points.csv.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
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
BUDGETS = [5, 10, 20, 30]


def rolling_predictions(d, feats, h, seed=0):
    recs = []
    for T in ORIGINS:
        tr = d[(d.year <= T - h) & d.at_risk]
        te = d[(d.year == T) & d.at_risk]
        if len(te) == 0 or tr[f"y{h}"].sum() < 5:
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr[feats].fillna(0).values)
        Xte = sc.transform(te[feats].fillna(0).values)
        m = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                       subsample=0.8, min_samples_leaf=20, random_state=seed)
        m.fit(Xtr, tr[f"y{h}"].values)
        p = m.predict_proba(Xte)[:, 1]
        for c, y, pi in zip(te["country_name"].values, te[f"y{h}"].values, p):
            recs.append({"origin": T, "country": c, "y": int(y), "p_raw": float(pi)})
    return pd.DataFrame(recs)


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def cal_slope_intercept(p, y):
    m = LogisticRegression(C=1e12, solver="lbfgs", max_iter=5000)
    m.fit(logit(p).reshape(-1, 1), y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def boot_ci(p, y, countries):
    uniq = np.unique(countries)
    idx_by_c = {c: np.where(countries == c)[0] for c in uniq}
    S, I = [], []
    for _ in range(N_BOOT):
        draw = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_c[c] for c in draw])
        if y[idx].sum() < 3 or y[idx].sum() == len(idx):
            continue
        try:
            s, i = cal_slope_intercept(p[idx], y[idx])
            S.append(s); I.append(i)
        except Exception:
            continue
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return q(S), q(I), len(S)


def prequential_recalibrate(pred):
    out = np.full(len(pred), np.nan)
    origins = sorted(pred.origin.unique())
    for i, T in enumerate(origins):
        past = pred[pred.origin < T]
        cur = pred.origin == T
        if len(past) < 40 or past.y.sum() < 5 or past.y.sum() == len(past):
            continue
        m = LogisticRegression(C=1e12, solver="lbfgs", max_iter=5000)
        m.fit(logit(past.p_raw.values).reshape(-1, 1), past.y.values)
        out[cur.values] = m.predict_proba(logit(pred.loc[cur, "p_raw"].values).reshape(-1, 1))[:, 1]
    pred["p_cal"] = out
    return pred


def operating_points(pred, col):
    rows = []
    sub = pred.dropna(subset=[col])
    base = sub.y.mean()
    for k in BUDGETS:
        tp = fp = fn = 0
        for T, g in sub.groupby("origin"):
            g = g.sort_values(col, ascending=False)
            top = g.head(k)
            tp += int(top.y.sum())
            fp += int((top.y == 0).sum())
            fn += int(g.y.sum() - top.y.sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        rows.append({"score": col, "budget_per_year": k,
                     "precision": round(prec, 4), "recall": round(rec, 4),
                     "lift_over_base_rate": round(prec / base, 2) if base else np.nan,
                     "base_rate": round(base, 4),
                     "alerts_per_year": k, "true_hits": tp, "false_alerts": fp})
    return pd.DataFrame(rows)


def main():
    d = build_panel()
    d[f"y{H}"] = label_h(d, H)
    feats = [c for c in d.columns if c not in EXCLUDE_COLS
             and c not in ("v2x_regime", "onset_year", "ep_end", "in_episode", "at_risk", f"y{H}")
             and d[c].dtype != object]
    print(f"features {len(feats)}, horizon {H}")

    pred = rolling_predictions(d, feats, H)
    print(f"scored {len(pred)} at-risk country-years, {int(pred.y.sum())} positives, "
          f"{pred.country.nunique()} countries, origins {pred.origin.min()}-{pred.origin.max()}")

    pred = prequential_recalibrate(pred)
    ok = pred.dropna(subset=["p_cal"])
    print(f"recalibrated rows: {len(ok)} (origins with enough history)\n")

    rows = []
    for col, lab in [("p_raw", "raw rolling-origin score"), ("p_cal", "prequentially recalibrated")]:
        sub = pred.dropna(subset=[col])
        s, i = cal_slope_intercept(sub[col].values, sub.y.values)
        (slo, shi), (ilo, ihi), nb = boot_ci(sub[col].values, sub.y.values, sub.country.values)
        auc = roc_auc_score(sub.y, sub[col]); ap = average_precision_score(sub.y, sub[col])
        rows.append({"score": lab, "n": len(sub), "auc_roc": round(auc, 4), "auc_pr": round(ap, 4),
                     "cal_slope": round(s, 3), "slope_lo": round(slo, 3), "slope_hi": round(shi, 3),
                     "cal_intercept": round(i, 3), "int_lo": round(ilo, 3), "int_hi": round(ihi, 3),
                     "n_boot": nb})
        print(f"{lab:<32} slope {s:6.3f} [{slo:6.3f},{shi:6.3f}]   "
              f"intercept {i:6.3f} [{ilo:6.3f},{ihi:6.3f}]   AUC {auc:.3f}  AP {ap:.3f}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "recalibration.csv"), index=False)

    op = pd.concat([operating_points(pred, "p_raw"), operating_points(pred, "p_cal")])
    op.to_csv(os.path.join(OUT, "operating_points.csv"), index=False)
    print("\noperating points by stated annual alert budget:")
    print(op.to_string(index=False))
    print("\nWrote recalibration.csv and operating_points.csv")


if __name__ == "__main__":
    main()
