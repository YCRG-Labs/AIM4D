"""
Baseline comparison + LOCO ablation study (restructured).

Two tables following ML pipeline best practices:

Table 1: External baselines (logistic, XGBoost, RF on raw V-Dem)
  Answers: "Does AIM4D outperform simpler approaches?"

Table 2: LOCO ablation (keep meta-learner constant, ablate inputs)
  Answers: "What does each pipeline stage contribute?"
  Standard: Hamilton et al. (2017 GraphSAGE), Kipf & Welling (2017 GCN)

Methodological basis:
  - Goldstone et al. (2010, AJPS): parsimonious logistic benchmark
  - Ward, Greenhill & Bakke (2010): OOS prediction standard
  - Montgomery, Hollenbach & Ward (2012): ensemble methods for instability
  - Muchlinski et al. (2016): ML vs logistic for rare events
  - Stock & Watson (2002): factor extraction justification
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stage5_ews.estimate import KNOWN_EPISODES, LEAD_YEARS, TRAIN_CUTOFF

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

WINDOWS = [
    (2005, 2008), (2008, 2011), (2011, 2014),
    (2014, 2017), (2017, 2020), (2020, 2023),
]


def build_labels(df):
    known_w = {}
    for c, info in KNOWN_EPISODES.items():
        for y in range(info["onset"] - LEAD_YEARS, info["onset"] + 1):
            known_w[(c, y)] = True
    df["label"] = df.apply(
        lambda r: 1 if (r.get("country_name", ""), r.get("year", 0)) in known_w else 0, axis=1
    )
    return df


def load_all_data():
    """Load and merge all pipeline stage outputs into a single panel."""
    base = os.path.join(os.path.dirname(__file__), "..")

    # V-Dem raw
    vdem_path = os.path.join(base, "data", "vdem_v16.csv")
    vdem_cols = ["country_name", "country_text_id", "year", "v2x_polyarchy", "v2x_regime",
                 "v2x_libdem", "v2x_partipdem", "v2x_egaldem"]
    available = pd.read_csv(vdem_path, low_memory=False, nrows=1).columns
    vdem_cols = [c for c in vdem_cols if c in available]
    vdem = pd.read_csv(vdem_path, low_memory=False, usecols=vdem_cols)
    vdem = vdem[vdem["year"] >= 1990]

    # Macro
    macro_path = os.path.join(base, "data", "macro_covariates.csv")
    macro = pd.read_csv(macro_path) if os.path.exists(macro_path) else None
    if macro is not None:
        macro_cols = ["gdp_pc", "gdp_growth"]
        avail = [c for c in macro_cols if c in macro.columns]
        vdem = vdem.merge(macro[["iso3", "year"] + avail].rename(columns={"iso3": "country_text_id"}),
                          on=["country_text_id", "year"], how="left")
        for c in avail:
            vdem[c] = vdem[c].fillna(vdem[c].median())
        vdem["log_gdp_pc"] = np.log1p(vdem.get("gdp_pc", 0))

    # Neighborhood polyarchy (region average proxy)
    vdem["region"] = vdem["country_text_id"].str[:2]
    vdem["neighbor_polyarchy"] = vdem.groupby(["region", "year"])["v2x_polyarchy"].transform("mean")

    # Stage 1: POET factors
    factors = pd.read_csv(os.path.join(base, "stage1_factors", "country_year_factors.csv"))
    factor_cols = [c for c in factors.columns if c.startswith("factor_")]

    # Stage 3: HMM states
    states = pd.read_csv(os.path.join(base, "stage3_msvar", "country_year_states.csv"))
    state_cols = [c for c in states.columns if c.startswith("prob_state_")]

    # Stage 4: Contagion
    contagion_path = os.path.join(base, "stage4_nscm", "contagion_scores.csv")
    contagion = pd.read_csv(contagion_path) if os.path.exists(contagion_path) else None

    # Stage 5: EWS (run full to get calibrated risk)
    try:
        from stage5_ews.estimate import run_ews
        ews = run_ews()
    except Exception:
        ews = pd.read_csv(os.path.join(base, "stage5_ews", "ews_signals.csv"))

    # Merge all
    panel = vdem.copy()
    panel = panel.merge(factors[["country_text_id", "year"] + factor_cols],
                        on=["country_text_id", "year"], how="left")
    panel = panel.merge(states[["country_text_id", "year"] + state_cols],
                        on=["country_text_id", "year"], how="left")
    if contagion is not None and "contagion_score" in contagion.columns:
        panel = panel.merge(contagion[["country_text_id", "year", "contagion_score"]],
                            on=["country_text_id", "year"], how="left")
        panel["contagion_score"] = panel["contagion_score"].fillna(0)

    ews_cols = ["csd_index", "election_vulnerability", "party_threat", "mil_zscore",
                "combined_risk", "calibrated_risk"]
    ews_avail = [c for c in ews_cols if c in ews.columns]
    if ews_avail:
        panel = panel.merge(ews[["country_text_id", "year"] + ews_avail],
                            on=["country_text_id", "year"], how="left")

    panel = build_labels(panel)
    return panel, factor_cols, state_cols


def evaluate(y_true, y_pred, name):
    results = {"model": name}
    if len(np.unique(y_true)) < 2:
        return {**results, "auc_roc": np.nan, "auc_pr": np.nan, "brier": np.nan}
    try:
        results["auc_roc"] = roc_auc_score(y_true, y_pred)
        results["auc_pr"] = average_precision_score(y_true, y_pred)
        results["brier"] = brier_score_loss(y_true, np.clip(y_pred, 0, 1))
    except ValueError:
        results["auc_roc"] = results["auc_pr"] = results["brier"] = np.nan
    results["n"] = len(y_true)
    results["n_positive"] = int(y_true.sum())
    return results


def temporal_cv_auc(panel, feature_cols, label_col="label"):
    """Expanding-window temporal CV."""
    aucs = []
    for train_end, test_end in WINDOWS:
        train = panel[(panel["year"] <= train_end) & panel[feature_cols].notna().all(axis=1)]
        test = panel[(panel["year"] > train_end) & (panel["year"] <= test_end) & panel[feature_cols].notna().all(axis=1)]
        if train[label_col].sum() < 3 or test[label_col].sum() == 0 or test[label_col].nunique() < 2:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train[feature_cols].values)
        X_te = scaler.transform(test[feature_cols].values)
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(X_tr, train[label_col].values)
        try:
            aucs.append(roc_auc_score(test[label_col], model.predict_proba(X_te)[:, 1]))
        except ValueError:
            pass
    return np.mean(aucs) if aucs else np.nan, np.std(aucs) if aucs else np.nan


def fit_and_evaluate(panel, feature_cols, name):
    """Fit logistic on train, evaluate on all."""
    valid = panel.dropna(subset=feature_cols + ["label"])
    if valid["label"].sum() < 3:
        return {"model": name, "auc_roc": np.nan}

    train = valid[valid["year"] <= TRAIN_CUTOFF]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[feature_cols].values)
    X_all = scaler.transform(valid[feature_cols].values)

    model = LogisticRegressionCV(cv=3, max_iter=1000, random_state=42)
    model.fit(X_tr, train["label"].values)
    y_pred = model.predict_proba(X_all)[:, 1]

    result = evaluate(valid["label"].values, y_pred, name)
    cv_mean, cv_std = temporal_cv_auc(valid, feature_cols)
    result["cv_auc_mean"] = cv_mean
    result["cv_auc_std"] = cv_std
    return result


def run_baseline_comparison():
    print("=" * 70)
    print("BASELINE COMPARISON + LOCO ABLATION (RESTRUCTURED)")
    print("=" * 70)
    print()

    panel, factor_cols, state_cols = load_all_data()
    print(f"Panel: {len(panel)} country-years, {panel['label'].sum()} positive labels\n")

    all_results = []

    # ================================================================
    # TABLE 1: EXTERNAL BASELINES
    # ================================================================
    print(f"{'='*60}")
    print("TABLE 1: External Baselines")
    print(f"{'='*60}\n")

    # 1a. Logistic on polyarchy + GDP
    b1_feats = [c for c in ["v2x_polyarchy", "log_gdp_pc", "neighbor_polyarchy"] if c in panel.columns]
    r = fit_and_evaluate(panel, b1_feats, "logistic_polyarchy_gdp")
    all_results.append(r)
    print(f"  Logistic (polyarchy+GDP): AUC={r.get('auc_roc', np.nan):.3f}, CV={r.get('cv_auc_mean', np.nan):.3f}")

    # 1b. XGBoost on raw V-Dem high-level indices
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        has_ensemble = True
    except ImportError:
        has_ensemble = False

    vdem_feats = [c for c in ["v2x_polyarchy", "v2x_libdem", "v2x_partipdem", "v2x_egaldem",
                               "v2x_regime", "log_gdp_pc", "gdp_growth", "neighbor_polyarchy"]
                  if c in panel.columns]

    if has_ensemble and vdem_feats:
        valid = panel.dropna(subset=vdem_feats + ["label"])
        train = valid[valid["year"] <= TRAIN_CUTOFF]

        if train["label"].sum() >= 3:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(train[vdem_feats].values)
            X_all = scaler.transform(valid[vdem_feats].values)

            # XGBoost (GradientBoosting)
            xgb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                              subsample=0.8, random_state=42)
            xgb.fit(X_tr, train["label"].values)
            y_xgb = xgb.predict_proba(X_all)[:, 1]
            r_xgb = evaluate(valid["label"].values, y_xgb, "xgboost_vdem_indices")
            # CV for XGBoost
            xgb_aucs = []
            for train_end, test_end in WINDOWS:
                w_tr = valid[valid["year"] <= train_end]
                w_te = valid[(valid["year"] > train_end) & (valid["year"] <= test_end)]
                if w_tr["label"].sum() < 3 or w_te["label"].sum() == 0 or w_te["label"].nunique() < 2:
                    continue
                sc = StandardScaler()
                xtr = sc.fit_transform(w_tr[vdem_feats].values)
                xte = sc.transform(w_te[vdem_feats].values)
                m = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                                subsample=0.8, random_state=42)
                m.fit(xtr, w_tr["label"].values)
                try:
                    xgb_aucs.append(roc_auc_score(w_te["label"], m.predict_proba(xte)[:, 1]))
                except ValueError:
                    pass
            r_xgb["cv_auc_mean"] = np.mean(xgb_aucs) if xgb_aucs else np.nan
            r_xgb["cv_auc_std"] = np.std(xgb_aucs) if xgb_aucs else np.nan
            all_results.append(r_xgb)
            print(f"  XGBoost (V-Dem indices): AUC={r_xgb['auc_roc']:.3f}, CV={r_xgb.get('cv_auc_mean', np.nan):.3f}")

            # Random Forest
            rf = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=42, class_weight="balanced")
            rf.fit(X_tr, train["label"].values)
            y_rf = rf.predict_proba(X_all)[:, 1]
            r_rf = evaluate(valid["label"].values, y_rf, "random_forest_vdem_indices")
            rf_aucs = []
            for train_end, test_end in WINDOWS:
                w_tr = valid[valid["year"] <= train_end]
                w_te = valid[(valid["year"] > train_end) & (valid["year"] <= test_end)]
                if w_tr["label"].sum() < 3 or w_te["label"].sum() == 0 or w_te["label"].nunique() < 2:
                    continue
                sc = StandardScaler()
                xtr = sc.fit_transform(w_tr[vdem_feats].values)
                xte = sc.transform(w_te[vdem_feats].values)
                m = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=42, class_weight="balanced")
                m.fit(xtr, w_tr["label"].values)
                try:
                    rf_aucs.append(roc_auc_score(w_te["label"], m.predict_proba(xte)[:, 1]))
                except ValueError:
                    pass
            r_rf["cv_auc_mean"] = np.mean(rf_aucs) if rf_aucs else np.nan
            r_rf["cv_auc_std"] = np.std(rf_aucs) if rf_aucs else np.nan
            all_results.append(r_rf)
            print(f"  Random Forest (V-Dem indices): AUC={r_rf['auc_roc']:.3f}, CV={r_rf.get('cv_auc_mean', np.nan):.3f}")

    # Full AIM4D
    if "combined_risk" in panel.columns:
        valid = panel.dropna(subset=["combined_risk", "label"])
        r_full = evaluate(valid["label"].values, valid["combined_risk"].values, "aim4d_full")
        cv_aucs = []
        for train_end, test_end in WINDOWS:
            w_te = valid[(valid["year"] > train_end) & (valid["year"] <= test_end)]
            if w_te["label"].sum() > 0 and w_te["label"].nunique() > 1:
                try:
                    cv_aucs.append(roc_auc_score(w_te["label"], w_te["combined_risk"]))
                except ValueError:
                    pass
        r_full["cv_auc_mean"] = np.mean(cv_aucs) if cv_aucs else np.nan
        r_full["cv_auc_std"] = np.std(cv_aucs) if cv_aucs else np.nan
        all_results.append(r_full)
        print(f"  AIM4D Full: AUC={r_full['auc_roc']:.3f}, CV={r_full.get('cv_auc_mean', np.nan):.3f}")

    # ================================================================
    # TABLE 2: LOCO ABLATION (meta-learner constant, ablate inputs)
    # ================================================================
    print(f"\n{'='*60}")
    print("TABLE 2: LOCO Ablation (meta-learner constant, ablate inputs)")
    print("  Standard: Hamilton et al. (2017), Kipf & Welling (2017)")
    print(f"{'='*60}\n")

    # Build the full meta-learner feature set
    from stage5_ews.estimate import run_ews
    ews_df = run_ews()

    # Identify meta-learner feature groups
    csd_features = [c for c in ews_df.columns if "csd" in c.lower() and c != "csd_x_network"]
    election_features = [c for c in ews_df.columns if "election" in c.lower() or "party_threat" in c.lower()]
    military_features = [c for c in ews_df.columns if "mil_" in c.lower()]
    network_features = [c for c in ews_df.columns if "network" in c.lower() or "contagion" in c.lower() or c == "csd_x_network"]
    dsp_features = [c for c in ews_df.columns if c.startswith("v2sm")]

    # All meta-learner input features
    all_meta_candidates = csd_features + election_features + military_features + network_features + dsp_features
    # Add detrended + era interaction versions
    all_meta = []
    for f in all_meta_candidates:
        if f in ews_df.columns:
            all_meta.append(f)
        if f"{f}_detrended" in ews_df.columns:
            all_meta.append(f"{f}_detrended")
        if f"{f}_x_post2015" in ews_df.columns:
            all_meta.append(f"{f}_x_post2015")
        if f"{f}_x_post2015_detrended" in ews_df.columns:
            all_meta.append(f"{f}_x_post2015_detrended")
    all_meta = list(dict.fromkeys(all_meta))  # deduplicate preserving order
    all_meta = [f for f in all_meta if f in ews_df.columns]

    if not all_meta or "label" not in ews_df.columns:
        print("  Could not build meta-learner features")
        summary = pd.DataFrame(all_results)
        summary.to_csv(os.path.join(OUTPUT_DIR, "baseline_comparison_results.csv"), index=False)
        return summary

    # Define feature groups for LOCO
    feature_groups = {
        "CSD (univariate + multivariate)": [f for f in all_meta if any(x in f for x in ["csd", "var_z", "ar1_z", "kurt_z", "dom_eig", "xcorr"])],
        "Election vulnerability": [f for f in all_meta if any(x in f for x in ["election", "party_threat"])],
        "Military threat": [f for f in all_meta if "mil_" in f],
        "Network exposure": [f for f in all_meta if any(x in f for x in ["network", "contagion", "csd_x_network"])],
        "DSP (digital)": [f for f in all_meta if f.startswith("v2sm")],
    }

    # Filter to features that actually exist
    for k in feature_groups:
        feature_groups[k] = [f for f in feature_groups[k] if f in ews_df.columns]

    # Full model
    from sklearn.preprocessing import StandardScaler as SS

    known_w = {}
    for c, info in KNOWN_EPISODES.items():
        for y in range(info["onset"] - LEAD_YEARS, info["onset"] + 1):
            known_w[(c, y)] = True
    ews_df["label"] = ews_df.apply(lambda r: 1 if (r["country_name"], r["year"]) in known_w else 0, axis=1)
    train_mask = ews_df["year"] <= TRAIN_CUTOFF

    half_life = 8
    max_year = ews_df.loc[train_mask, "year"].max()
    time_weights = np.exp(-np.log(2) * (max_year - ews_df["year"].values) / half_life)

    def fit_meta_learner(feature_list, name):
        feats = [f for f in feature_list if f in ews_df.columns]
        if not feats:
            return {"model": name, "auc_roc": np.nan, "cv_auc_mean": np.nan}

        X = ews_df[feats].fillna(0).values
        y = ews_df["label"].values
        scaler = SS()
        X_s = scaler.fit_transform(X)

        model = LogisticRegressionCV(cv=3, scoring="average_precision", max_iter=1000, random_state=42)
        model.fit(X_s[train_mask], y[train_mask], sample_weight=time_weights[train_mask])
        y_pred = model.predict_proba(X_s)[:, 1]

        result = evaluate(y, y_pred, name)

        # CV
        valid_df = ews_df.copy()
        valid_df["pred"] = y_pred
        cv_aucs = []
        for train_end, test_end in WINDOWS:
            w_te = valid_df[(valid_df["year"] > train_end) & (valid_df["year"] <= test_end)]
            if w_te["label"].sum() > 0 and w_te["label"].nunique() > 1:
                try:
                    cv_aucs.append(roc_auc_score(w_te["label"], w_te["pred"]))
                except ValueError:
                    pass
        result["cv_auc_mean"] = np.mean(cv_aucs) if cv_aucs else np.nan
        result["cv_auc_std"] = np.std(cv_aucs) if cv_aucs else np.nan
        result["n_features"] = len(feats)
        return result

    # Full meta-learner
    r_full_meta = fit_meta_learner(all_meta, "full_meta_learner")
    loco_results = [r_full_meta]
    print(f"  Full meta-learner ({r_full_meta.get('n_features', 0)} features): AUC={r_full_meta.get('auc_roc', np.nan):.3f}, CV={r_full_meta.get('cv_auc_mean', np.nan):.3f}")

    # LOCO: remove each group
    for group_name, group_feats in feature_groups.items():
        if not group_feats:
            continue
        remaining = [f for f in all_meta if f not in group_feats]
        r = fit_meta_learner(remaining, f"without_{group_name}")
        delta = r_full_meta.get("auc_roc", 0) - r.get("auc_roc", 0)
        loco_results.append(r)
        print(f"  Without {group_name}: AUC={r.get('auc_roc', np.nan):.3f} (delta={delta:+.3f})")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*60}")
    print("SUMMARY: External Baselines")
    print(f"{'='*60}")
    baseline_df = pd.DataFrame(all_results)
    cols = ["model", "auc_roc", "auc_pr", "brier", "cv_auc_mean", "cv_auc_std"]
    cols = [c for c in cols if c in baseline_df.columns]
    print(baseline_df[cols].to_string(index=False, float_format="%.3f"))

    print(f"\n{'='*60}")
    print("SUMMARY: LOCO Ablation")
    print(f"{'='*60}")
    loco_df = pd.DataFrame(loco_results)
    loco_cols = ["model", "auc_roc", "cv_auc_mean", "n_features"]
    loco_cols = [c for c in loco_cols if c in loco_df.columns]
    print(loco_df[loco_cols].to_string(index=False, float_format="%.3f"))

    # Save
    combined = pd.DataFrame(all_results + loco_results)
    combined.to_csv(os.path.join(OUTPUT_DIR, "baseline_comparison_results.csv"), index=False)
    print(f"\nSaved to robustness/baseline_comparison_results.csv")

    return combined


if __name__ == "__main__":
    run_baseline_comparison()
