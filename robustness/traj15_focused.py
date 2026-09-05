"""
Focused test of the one arm that beat everything on average precision.

Across the whole strategy search, the single best average precision came from
trajectory-shape features over a fifteen-year window (AP 0.266 against a base
rate of 0.102), even though its AUC of 0.70 was unremarkable. That combination
matters here: at a ten percent base rate average precision is the metric an
operator feels, and AUC is not. The contrast was not significant in the sweep
that produced it, so this re-tests it directly.

Shape statistics per variable over the trailing fifteen years: level, slope,
curvature, dispersion, drawdown from the running peak, net change, the share of
years in decline, and the longest unbroken decline. Everything is computed from
years t-14..t inclusive, so nothing crosses the forecast origin.

Protocol is the locked one: democratic at-risk pool, predictors dated t or
earlier, rolling origins, aggregation chosen inside the training window.

Outputs robustness/traj15_focused.csv.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from onset_forecast_clean import build_panel, label_h, EXCLUDE_COLS, ORIGINS
from onset_forecast_optimized import THEORY

OUT = os.path.dirname(os.path.abspath(__file__))
H = 5
WIN = 15
N_BOOT = 2000
RNG = np.random.default_rng(20260904)


def shape_stats(a):
    """Trajectory descriptors for one trailing window; a is (n, WIN), oldest first."""
    n, w = a.shape
    x = np.arange(w, dtype=float)
    xc = x - x.mean()
    out = {}
    valid = ~np.isnan(a)
    filled = np.where(valid, a, np.nanmean(a, axis=1, keepdims=True))
    filled = np.nan_to_num(filled)
    out["level"] = filled[:, -1]
    out["mean"] = filled.mean(axis=1)
    out["sd"] = filled.std(axis=1)
    denom = (xc ** 2).sum()
    out["slope"] = ((filled - filled.mean(axis=1, keepdims=True)) * xc).sum(axis=1) / denom
    half = w // 2
    s1 = ((filled[:, :half] - filled[:, :half].mean(axis=1, keepdims=True))
          * (np.arange(half) - np.arange(half).mean())).sum(axis=1) / max(((np.arange(half) - np.arange(half).mean()) ** 2).sum(), 1e-9)
    s2 = ((filled[:, half:] - filled[:, half:].mean(axis=1, keepdims=True))
          * (np.arange(w - half) - np.arange(w - half).mean())).sum(axis=1) / max(((np.arange(w - half) - np.arange(w - half).mean()) ** 2).sum(), 1e-9)
    out["accel"] = s2 - s1
    out["net"] = filled[:, -1] - filled[:, 0]
    out["drawdown"] = filled[:, -1] - filled.max(axis=1)
    dif = np.diff(filled, axis=1)
    out["frac_decl"] = (dif < 0).mean(axis=1)
    runs = np.zeros(n)
    cur = np.zeros(n)
    for j in range(dif.shape[1]):
        cur = np.where(dif[:, j] < 0, cur + 1, 0)
        runs = np.maximum(runs, cur)
    out["max_decl_run"] = runs
    return out


def build_traj15(d):
    d = d.sort_values(["country_name", "year"]).reset_index(drop=True)
    v = pd.read_csv(os.path.join(OUT, "..", "data", "vdem_v16.csv"), low_memory=False,
                    usecols=["country_name", "year"] + list(THEORY.values()))
    v = v.rename(columns={x: k for k, x in THEORY.items()})
    d = d.merge(v, on=["country_name", "year"], how="left")
    d = d.sort_values(["country_name", "year"]).reset_index(drop=True)

    frames = {}
    for k in THEORY:
        mats = []
        for _, g in d.groupby("country_name", sort=False):
            s = g[k].to_numpy(float)
            m = np.full((len(s), WIN), np.nan)
            for j in range(len(s)):
                lo = max(0, j - WIN + 1)
                seg = s[lo:j + 1]
                m[j, WIN - len(seg):] = seg
            mats.append(m)
        M = np.vstack(mats)
        for nm, arr in shape_stats(M).items():
            frames[f"t15_{k}_{nm}"] = arr
    return pd.concat([d, pd.DataFrame(frames, index=d.index)], axis=1), list(frames.keys())


def roll(d, feats, kind, seed=0):
    recs = []
    for T in ORIGINS:
        tr = d[(d.year <= T - H) & d.at_risk]
        te = d[(d.year == T) & d.at_risk]
        if len(te) == 0 or tr[f"y{H}"].sum() < 8:
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr[feats].fillna(0).values)
        Xte = sc.transform(te[feats].fillna(0).values)
        m = (RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced",
                                    random_state=seed, n_jobs=-1) if kind == "rf" else
             HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                            l2_regularization=1.0, random_state=seed,
                                            early_stopping=False) if kind == "hgb" else
             LogisticRegression(max_iter=4000, class_weight="balanced", C=0.1))
        try:
            m.fit(Xtr, tr[f"y{H}"].values)
            p = m.predict_proba(Xte)[:, 1]
        except Exception:
            continue
        for c, y, pi in zip(te.country_name.values, te[f"y{H}"].values, p):
            recs.append({"origin": T, "country": c, "y": int(y), "p": float(pi)})
    return pd.DataFrame(recs)


def prec_at(df, k):
    tp = fp = 0
    for _, g in df.groupby("origin"):
        top = g.sort_values("p", ascending=False).head(k)
        tp += int(top.y.sum()); fp += int((top.y == 0).sum())
    return tp / (tp + fp) if tp + fp else np.nan


def paired(y, pa, pb, cty):
    uniq = np.unique(cty); idx = {c: np.where(cty == c)[0] for c in uniq}
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
    f = lambda a: (np.mean(a), np.percentile(a, 2.5), np.percentile(a, 97.5), np.mean(np.array(a) > 0))
    return f(da), f(dp)


def main():
    d = build_panel()
    d, T15 = build_traj15(d)
    d[f"y{H}"] = label_h(d, H)
    base = [c for c in d.columns if c not in set(EXCLUDE_COLS) |
            {"v2x_regime", "onset_year", "ep_end", "in_episode", "at_risk", f"y{H}"}
            and c not in T15 and c not in THEORY and d[c].dtype != object]
    print(f"BASE {len(base)} features | TRAJ15 {len(T15)} features "
          f"({len(THEORY)} variables x {len(T15)//len(THEORY)} shape stats over {WIN} years)\n")

    spaces = {"BASE": base, "TRAJ15": T15, "BASE+TRAJ15": base + T15}
    preds, rows = {}, []
    for sp, fl in spaces.items():
        per = {}
        for kind in ["rf", "hgb", "lr"]:
            r = roll(d, fl, kind)
            if r.empty:
                continue
            per[kind] = r
            rows.append({"space": sp, "model": kind, "auc_roc": round(roc_auc_score(r.y, r.p), 4),
                         "auc_pr": round(average_precision_score(r.y, r.p), 4),
                         "prec_top10": round(prec_at(r, 10), 4), "n": len(r), "n_pos": int(r.y.sum())})
            print(f"  {sp:<13} {kind:<4} AUC={roc_auc_score(r.y,r.p):.4f}  "
                  f"AP={average_precision_score(r.y,r.p):.4f}  p@10={prec_at(r,10):.3f}")
        if len(per) >= 2:
            k0 = list(per)[0]
            m = per[k0][["origin", "country", "y"]].copy()
            for kind, r in per.items():
                m = m.merge(r.rename(columns={"p": kind}), on=["origin", "country", "y"], how="inner")
            for kind in per:
                m[kind] = m.groupby("origin")[kind].rank(pct=True)
            m["p"] = m[list(per)].mean(axis=1)
            ens = m[["origin", "country", "y", "p"]]
            preds[sp] = ens
            rows.append({"space": sp, "model": "rank-mean ens",
                         "auc_roc": round(roc_auc_score(ens.y, ens.p), 4),
                         "auc_pr": round(average_precision_score(ens.y, ens.p), 4),
                         "prec_top10": round(prec_at(ens, 10), 4), "n": len(ens),
                         "n_pos": int(ens.y.sum())})
            print(f"  {sp:<13} ens  AUC={roc_auc_score(ens.y,ens.p):.4f}  "
                  f"AP={average_precision_score(ens.y,ens.p):.4f}  p@10={prec_at(ens,10):.3f}\n")

    if "BASE" in preds:
        b = preds["BASE"]
        for sp in ["TRAJ15", "BASE+TRAJ15"]:
            if sp not in preds:
                continue
            m = b.rename(columns={"p": "pb"}).merge(
                preds[sp].rename(columns={"p": "pa"}), on=["origin", "country", "y"])
            (ma, la, ha, pa_), (mp, lp, hp, pp_) = paired(
                m.y.values, m.pa.values, m.pb.values, m.country.values)
            print(f"  {sp} minus BASE:  dAUC {ma:+.4f} [{la:+.4f},{ha:+.4f}] p(>0)={pa_:.3f}   "
                  f"dAP {mp:+.4f} [{lp:+.4f},{hp:+.4f}] p(>0)={pp_:.3f}")
            rows.append({"space": f"delta {sp} - BASE", "model": "rank-mean ens",
                         "auc_roc": round(ma, 4), "auc_pr": round(mp, 4),
                         "prec_top10": np.nan, "n": len(m), "n_pos": int(m.y.sum())})

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "traj15_focused.csv"), index=False)
    print("\nWrote traj15_focused.csv")


if __name__ == "__main__":
    main()
