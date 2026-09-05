"""
Maximize honest forecasting skill for autocratization onset among democracies.

Same estimand and protocol as onset_forecast_clean.py: the at-risk pool is
democratic country-years not already inside an episode, the target is an onset
during t+1..t+h, and every predictor is dated t or earlier with forecasts made
from rolling origins. Nothing here touches the scored year.

What changes is the input space and the model selection. The Stage-5 feature
matrix was engineered for pre-onset-window membership, a different target, so
this adds predictors the comparative literature actually nominates for onset
risk: the protective belt of judicial and legislative constraints, rule of law,
civil-society participation and party institutionalization; polarization;
clientelism and neopatrimonial rule; media freedom; and the dynamics of each,
since a level and a three-year slide carry different information. It also adds
lagged regional and global onset counts, which are legitimate here because they
are computed only from onsets already observed at the forecast origin.

Hyperparameters are chosen by blocked time-series validation INSIDE the
training window at each origin. The scored year never participates in
selection, so the reported figures remain out of sample in the information
sense and not merely the row sense.

Outputs robustness/onset_forecast_optimized.csv and *_predictions.csv.
"""

import os
import re
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

from onset_forecast_clean import build_panel, label_h, EXCLUDE_COLS

OUT = os.path.dirname(os.path.abspath(__file__))
H = int(os.environ.get("AIM4D_H", "5"))
ORIGINS = list(range(2005, 2026))

THEORY = {
    "polyarchy": "v2x_polyarchy", "libdem": "v2x_libdem", "egaldem": "v2x_egaldem",
    "jucon": "v2x_jucon", "legcon": "v2xlg_legcon", "rule": "v2x_rule",
    "cspart": "v2x_cspart", "csprtcpt": "v2csprtcpt", "polarization": "v2cacamps",
    "party_inst": "v2xps_party", "execorr": "v2x_execorr", "client": "v2xnp_client",
    "freexp": "v2x_freexp_altinf", "media": "v2xme_altinf", "civlib": "v2x_civlib",
    "neopat": "v2x_neopat",
}

GRID_HGB = [
    {"max_depth": 3, "learning_rate": 0.05, "max_iter": 300, "l2_regularization": 1.0},
    {"max_depth": 3, "learning_rate": 0.03, "max_iter": 500, "l2_regularization": 5.0},
    {"max_depth": None, "learning_rate": 0.05, "max_iter": 300, "l2_regularization": 1.0},
    {"max_depth": 2, "learning_rate": 0.08, "max_iter": 400, "l2_regularization": 0.5},
]
GRID_LR = [{"C": c} for c in (0.01, 0.1, 1.0)]


def enrich(d):
    v = pd.read_csv(os.path.join(OUT, "..", "data", "vdem_v16.csv"), low_memory=False,
                    usecols=["country_name", "year"] + list(THEORY.values()))
    v = v.rename(columns={x: k for k, x in THEORY.items()})
    d = d.merge(v, on=["country_name", "year"], how="left")
    d = d.sort_values(["country_name", "year"])
    g = d.groupby("country_name", group_keys=False)
    new = {}
    for k in THEORY:
        s = d[k]
        new[f"{k}_d1"] = g[k].diff(1)
        new[f"{k}_d3"] = g[k].diff(3)
        new[f"{k}_d5"] = g[k].diff(5)
        new[f"{k}_rm5"] = g[k].transform(lambda x: x.rolling(5, min_periods=2).mean())
        new[f"{k}_rsd5"] = g[k].transform(lambda x: x.rolling(5, min_periods=2).std())
        new[f"{k}_min5"] = g[k].transform(lambda x: x.rolling(5, min_periods=2).min())
        new[f"{k}_drawdown"] = s - g[k].transform(lambda x: x.rolling(10, min_periods=2).max())
    d = pd.concat([d, pd.DataFrame(new, index=d.index)], axis=1)

    onset_by_year = (d.dropna(subset=["onset_year"])
                      .drop_duplicates("country_name")[["country_name", "onset_year"]])
    counts = onset_by_year.groupby("onset_year").size()
    yrs = sorted(d.year.unique())
    prior5 = {y: int(sum(counts.get(z, 0) for z in range(y - 5, y))) for y in yrs}
    prior10 = {y: int(sum(counts.get(z, 0) for z in range(y - 10, y))) for y in yrs}
    d["global_onsets_prior5"] = d.year.map(prior5)
    d["global_onsets_prior10"] = d.year.map(prior10)
    d["yrs_since_own_onset"] = np.where(d.onset_year.notna() & (d.year > d.onset_year),
                                        d.year - d.onset_year, -1)
    d["had_prior_episode"] = (d["yrs_since_own_onset"] > 0).astype(int)
    return d


def feature_list(d):
    """Every label column must be excluded, not just the one for the active horizon.

    A previous version dropped only y{H}. Any other y{h} created on the same frame
    survived into the feature matrix and predicted itself, which is why h=1 once
    returned an AUC of 0.9995.
    """
    drop = set(EXCLUDE_COLS) | {"v2x_regime", "onset_year", "ep_end", "in_episode", "at_risk"}
    drop |= {c for c in d.columns if re.fullmatch(r"y\d+", str(c))}
    return [c for c in d.columns if c not in drop and d[c].dtype != object]


def fit_predict(cfg, kind, Xtr, ytr, Xte, w=None):
    if kind == "hgb":
        m = HistGradientBoostingClassifier(random_state=0, early_stopping=False, **cfg)
    elif kind == "rf":
        m = RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                   class_weight="balanced", random_state=0, n_jobs=-1)
    else:
        m = LogisticRegression(max_iter=4000, class_weight="balanced", **cfg)
    m.fit(Xtr, ytr, sample_weight=w) if w is not None and kind != "rf" else m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def inner_select(tr, feats, h, kind, grid):
    """Blocked time-series validation strictly inside the training window."""
    yrs = np.sort(tr.year.unique())
    if len(yrs) < 12:
        return grid[0]
    cuts = [yrs[int(len(yrs) * f)] for f in (0.6, 0.75, 0.9)]
    best, best_score = grid[0], -np.inf
    for cfg in grid:
        scores = []
        for c in cuts:
            a = tr[tr.year <= c - h]
            b = tr[(tr.year > c - h) & (tr.year <= c)]
            if len(b) < 30 or a[f"y{h}"].sum() < 5 or b[f"y{h}"].sum() < 2:
                continue
            sc = StandardScaler()
            Xa = sc.fit_transform(a[feats].fillna(0).values)
            Xb = sc.transform(b[feats].fillna(0).values)
            try:
                p = fit_predict(cfg, kind, Xa, a[f"y{h}"].values, Xb)
                scores.append(average_precision_score(b[f"y{h}"].values, p))
            except Exception:
                continue
        if scores and np.mean(scores) > best_score:
            best_score, best = np.mean(scores), cfg
    return best


def run(d, feats, h, kind, grid):
    recs = []
    for T in ORIGINS:
        tr = d[(d.year <= T - h) & d.at_risk]
        te = d[(d.year == T) & d.at_risk]
        if len(te) == 0 or tr[f"y{h}"].sum() < 8:
            continue
        cfg = inner_select(tr, feats, h, kind, grid)
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr[feats].fillna(0).values)
        Xte = sc.transform(te[feats].fillna(0).values)
        w = np.where(tr[f"y{h}"].values == 1,
                     (len(tr) - tr[f"y{h}"].sum()) / max(tr[f"y{h}"].sum(), 1), 1.0)
        try:
            p = fit_predict(cfg, kind, Xtr, tr[f"y{h}"].values, Xte, w if kind == "hgb" else None)
        except Exception:
            continue
        for c, y, pi in zip(te.country_name.values, te[f"y{h}"].values, p):
            recs.append({"origin": T, "country": c, "y": int(y), "p": float(pi), "model": kind})
    return pd.DataFrame(recs)


def score(df, tag):
    if df.empty or df.y.sum() < 5:
        return None
    a = roc_auc_score(df.y, df.p); ap = average_precision_score(df.y, df.p)
    print(f"  {tag:<26} n={len(df):>5} pos={int(df.y.sum()):>3}  AUC={a:.4f}  AP={ap:.4f}  base={df.y.mean():.4f}")
    return {"model": tag, "n": len(df), "n_pos": int(df.y.sum()),
            "auc_roc": round(a, 4), "auc_pr": round(ap, 4), "base_rate": round(df.y.mean(), 4)}


def main():
    d = build_panel()
    d = enrich(d)
    d[f"y{H}"] = label_h(d, H)
    feats = feature_list(d)
    print(f"horizon h={H}; features {len(feats)} "
          f"({len([c for c in feats if any(c.startswith(k) for k in THEORY)])} theory-derived)")
    print(f"at-risk rows {int(d.at_risk.sum())}, positives {int(d.loc[d.at_risk, f'y{H}'].sum())}\n")

    rows, preds = [], []
    for kind, grid in [("hgb", GRID_HGB), ("rf", [{}]), ("lr", GRID_LR)]:
        p = run(d, feats, H, kind, grid)
        if p.empty:
            continue
        preds.append(p)
        r = score(p, kind)
        if r:
            rows.append(r)

    if preds:
        allp = pd.concat(preds)
        piv = allp.pivot_table(index=["origin", "country", "y"], columns="model",
                               values="p").reset_index()
        mcols = [c for c in piv.columns if c in ("hgb", "rf", "lr")]
        piv["p"] = piv[mcols].mean(axis=1)
        ens = piv[["origin", "country", "y", "p"]].dropna()
        r = score(ens, "ensemble (mean)")
        if r:
            rows.append(r)
        rk = piv.copy()
        for c in mcols:
            rk[c] = rk.groupby("origin")[c].rank(pct=True)
        rk["p"] = rk[mcols].mean(axis=1)
        r = score(rk[["origin", "country", "y", "p"]].dropna(), "ensemble (rank-mean)")
        if r:
            rows.append(r)
        ens.to_csv(os.path.join(OUT, "onset_forecast_optimized_predictions.csv"), index=False)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "onset_forecast_optimized.csv"), index=False)
    print("\nWrote onset_forecast_optimized.csv")


if __name__ == "__main__":
    main()
