"""External forecasting benchmarks re-estimated on the AIM4D autocratization panel.

Four comparators, each a faithful reimplementation of a published specification
re-fit on our ERT 5-year pre-onset label, never a transfer of published numbers:

  persistence    trend-extrapolation baseline (ViEWS bm_last_historical philosophy):
                 the trailing 3-year decline in v2x_polyarchy used directly as risk.
  pitf           Goldstone et al. (2010) four-variable logit: regime type
                 (Polity EXREC x PARCOMP), log-normalised infant mortality,
                 autocratizing-neighbourhood, state-led discrimination.
  elastic_net    L1/L2-penalised logit on the full 332-indicator V-Dem space
                 (the dimensionality alternative to Stage-1 factor extraction).
  vforecast      Morgan, Beger & Glynn (2019) PART recipe: unweighted ensemble of
                 elastic-net logit + random forest + gradient-boosted forest on the
                 same 332-indicator space.

Every row runs through baseline_comparison's labels, WINDOWS, and evaluate(), so
results sit directly alongside the in-house baselines and AIM4D.
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from stage5_ews.estimate import TRAIN_CUTOFF
from baseline_comparison import build_labels, evaluate, WINDOWS
from stage1_factors.extract import select_indicators, build_panel

BASE = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
REGIME_DUMMIES = ["reg_AP", "reg_Dfact", "reg_DP", "reg_DF", "reg_transition"]


def load_panel():
    """Country-year panel with the PITF inputs and persistence score merged on."""
    vdem = pd.read_csv(os.path.join(BASE, "data", "vdem_v16.csv"), low_memory=False)
    vdem = vdem[(vdem["year"] >= 1970) & (vdem["year"] <= 2025)].copy()

    macro = pd.read_csv(os.path.join(BASE, "data", "macro_covariates.csv"))
    vdem = vdem.merge(
        macro[["iso3", "year", "gdp_pc", "gdp_growth"]].rename(columns={"iso3": "country_text_id"}),
        on=["country_text_id", "year"], how="left",
    )
    vdem["log_gdp_pc"] = np.log1p(vdem["gdp_pc"].fillna(vdem["gdp_pc"].median()))

    mp = pd.read_csv(os.path.join(BASE, "data", "macro_pitf.csv"))
    vdem = vdem.merge(
        mp[["iso3", "year", "infant_mortality"]].rename(columns={"iso3": "country_text_id"}),
        on=["country_text_id", "year"], how="left",
    )

    reg = pd.read_csv(os.path.join(BASE, "data", "pitf_regime.csv"))
    vdem = vdem.merge(reg, on=["COWcode", "year"], how="left")

    gd = pd.read_csv(os.path.join(BASE, "data", "global_diffusion.csv"))
    vdem = vdem.merge(
        gd[["country_text_id", "year", "n_backsliding_neighbors"]],
        on=["country_text_id", "year"], how="left",
    )

    annual_med = vdem.groupby("year")["infant_mortality"].transform("median")
    vdem["imr_normed_ln"] = np.log(np.clip(vdem["infant_mortality"] / annual_med, 1e-3, None))

    disc_thresh = vdem.loc[vdem["year"] <= TRAIN_CUTOFF, "v2pepwrsoc"].quantile(0.25)
    vdem["state_discrimination"] = (vdem["v2pepwrsoc"] <= disc_thresh).astype(float)

    mapping = {"A/P": "reg_AP", "D/fact": "reg_Dfact", "D/P": "reg_DP",
               "D/F": "reg_DF", "transition": "reg_transition"}
    for col in REGIME_DUMMIES:
        vdem[col] = 0.0
    for cat, col in mapping.items():
        vdem.loc[vdem["pol_cat_pitf"] == cat, col] = 1.0
    vdem.loc[vdem["pol_cat_pitf"].isna(), REGIME_DUMMIES] = np.nan

    vdem = vdem.sort_values(["country_text_id", "year"])
    vdem["poly_decline_3yr"] = (
        vdem.groupby("country_text_id")["v2x_polyarchy"].shift(3) - vdem["v2x_polyarchy"]
    )

    vdem = build_labels(vdem)
    return vdem


def load_indicator_panel():
    """The 332-indicator V-Dem matrix used by Stage 1, with onset labels merged on."""
    vdem = pd.read_csv(os.path.join(BASE, "data", "vdem_v16.csv"), low_memory=False)
    indicators = select_indicators(vdem)
    panel = build_panel(vdem, indicators)
    panel = build_labels(panel)
    return panel, indicators


def cv_auc(panel, feats, make_model, label="label"):
    aucs = []
    for train_end, test_end in WINDOWS:
        tr = panel[(panel["year"] <= train_end) & panel[feats].notna().all(axis=1) & panel[label].notna()]
        te = panel[(panel["year"] > train_end) & (panel["year"] <= test_end)
                   & panel[feats].notna().all(axis=1) & panel[label].notna()]
        if tr[label].sum() < 3 or te[label].sum() == 0 or te[label].nunique() < 2:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr[feats].values)
        X_te = scaler.transform(te[feats].values)
        model = make_model()
        model.fit(X_tr, tr[label].values)
        try:
            aucs.append(roc_auc_score(te[label], model.predict_proba(X_te)[:, 1]))
        except ValueError:
            pass
    return (np.mean(aucs), np.std(aucs)) if aucs else (np.nan, np.nan)


def fit_model_row(panel, feats, make_model, name, label="label"):
    valid = panel.dropna(subset=feats + [label])
    if valid[label].sum() < 3:
        return {"model": name, "auc_roc": np.nan}
    train = valid[valid["year"] <= TRAIN_CUTOFF]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[feats].values)
    X_all = scaler.transform(valid[feats].values)
    model = make_model()
    model.fit(X_tr, train[label].values)
    y_pred = model.predict_proba(X_all)[:, 1]
    result = evaluate(valid[label].values, y_pred, name)
    result["cv_auc_mean"], result["cv_auc_std"] = cv_auc(valid, feats, make_model, label)
    return result


def score_row(panel, score_col, name, label="label"):
    valid = panel.dropna(subset=[score_col, label])
    result = evaluate(valid[label].values, valid[score_col].values, name)
    aucs = []
    for train_end, test_end in WINDOWS:
        te = valid[(valid["year"] > train_end) & (valid["year"] <= test_end)]
        if te[label].sum() > 0 and te[label].nunique() > 1:
            try:
                aucs.append(roc_auc_score(te[label], te[score_col]))
            except ValueError:
                pass
    result["cv_auc_mean"] = np.mean(aucs) if aucs else np.nan
    result["cv_auc_std"] = np.std(aucs) if aucs else np.nan
    return result


class UnweightedEnsemble:
    """V-Forecast PART ensemble: mean predicted probability across constituents."""

    def __init__(self):
        self.members = [
            LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                               C=1.0, max_iter=2000, class_weight="balanced", random_state=42),
            RandomForestClassifier(n_estimators=500, max_depth=6, class_weight="balanced",
                                   random_state=42, n_jobs=-1),
            GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                       subsample=0.8, random_state=42),
        ]

    def fit(self, X, y):
        for m in self.members:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.members], axis=0)
        return np.column_stack([1 - p, p])


def make_enet():
    return LogisticRegressionCV(penalty="elasticnet", solver="saga", l1_ratios=[0.5],
                                Cs=5, cv=3, scoring="roc_auc", max_iter=2000,
                                class_weight="balanced", random_state=42, n_jobs=-1)


def run():
    print("=" * 70)
    print("EXTERNAL FORECASTING BENCHMARKS (re-estimated on AIM4D panel)")
    print("=" * 70)

    panel = load_panel()
    print(f"\nPanel: {len(panel)} country-years, {int(panel['label'].sum())} positive labels")

    results = []

    r = score_row(panel, "poly_decline_3yr", "persistence_trend")
    results.append(r)
    print(f"  Persistence (3yr polyarchy decline): AUC={r['auc_roc']:.3f}, "
          f"CV={r.get('cv_auc_mean', np.nan):.3f}, n={r.get('n')}")

    pitf_feats = REGIME_DUMMIES + ["imr_normed_ln", "n_backsliding_neighbors", "state_discrimination"]
    r = fit_model_row(panel, pitf_feats, lambda: LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced", random_state=42), "pitf_goldstone_2010")
    results.append(r)
    print(f"  PITF logit (Goldstone 2010): AUC={r['auc_roc']:.3f}, "
          f"CV={r.get('cv_auc_mean', np.nan):.3f}, n={r.get('n')}")

    ind_panel, indicators = load_indicator_panel()
    print(f"  (indicator matrix: {len(ind_panel)} rows, {len(indicators)} indicators)")

    r = fit_model_row(ind_panel, indicators, make_enet, "elastic_net_full_vdem")
    results.append(r)
    print(f"  Elastic-net (full V-Dem): AUC={r['auc_roc']:.3f}, "
          f"CV={r.get('cv_auc_mean', np.nan):.3f}, n={r.get('n')}")

    r = fit_model_row(ind_panel, indicators, UnweightedEnsemble, "vforecast_ensemble")
    results.append(r)
    print(f"  V-Forecast ensemble (Morgan 2019): AUC={r['auc_roc']:.3f}, "
          f"CV={r.get('cv_auc_mean', np.nan):.3f}, n={r.get('n')}")

    df = pd.DataFrame(results)
    cols = ["model", "auc_roc", "auc_pr", "brier", "cv_auc_mean", "cv_auc_std", "n", "n_positive"]
    cols = [c for c in cols if c in df.columns]
    print("\n" + "=" * 70)
    print(df[cols].to_string(index=False, float_format="%.3f"))
    out = os.path.join(OUTPUT_DIR, "external_benchmarks_results.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")
    return df


if __name__ == "__main__":
    run()
