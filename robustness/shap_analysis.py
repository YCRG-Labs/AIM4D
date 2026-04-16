"""
SHAP interpretability analysis for the stacked ensemble meta-learner.

Computes grouped SHAP values showing each pipeline stage's contribution
to risk predictions, addressing the "black box" concern.

Methodological basis:
  - Lundberg & Lee (2017, NeurIPS): SHAP values
  - Covert, Lundberg & Lee (2021, AISTATS): grouped SHAP
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_shap_analysis():
    print("=" * 70)
    print("SHAP Interpretability Analysis")
    print("=" * 70)

    # Run full stage 5 to get models and data
    from stage5_ews.estimate import run_ews
    ews_df = run_ews()

    # We need to access the fitted models — re-extract from the module
    # Since models aren't returned, we re-fit on the same data
    from stage5_ews.estimate import KNOWN_EPISODES, LEAD_YEARS, TRAIN_CUTOFF
    from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold

    # Rebuild features (same as in run_ews)
    base_features = [c for c in ews_df.columns if any(c.startswith(p) for p in
                     ["csd_index", "mv_csd", "election_", "party_threat", "mil_zscore",
                      "network_", "csd_x_", "v2sm", "protest_", "conflict_", "repression_",
                      "v2jun", "v2xlg", "v2x_ju", "v2exres"])]

    # Filter to numeric, non-label columns
    exclude = ["label", "label_soft", "year", "combined_risk", "calibrated_risk",
               "alert_tier", "combined_alert", "combined_alert_legacy",
               "raw_alert", "ews_alert", "election_alert", "dem_vulnerability_alert",
               "military_threat_alert", "mv_csd_alert", "eig_trend_sig", "xcorr_trend_sig"]
    meta_candidates = [c for c in ews_df.columns
                       if c not in exclude
                       and c not in ["country_name", "country_text_id", "region"]
                       and ews_df[c].dtype in [np.float64, np.int64, float, int]
                       and "_lag" in c or "_delta" in c or "_pctile" in c
                       or "_detrended" in c or "_x_post" in c or "_zscore" in c
                       or c in base_features]

    # Use the same feature set as the meta-learner by checking what's available
    all_meta = [c for c in meta_candidates if c in ews_df.columns and not ews_df[c].isna().all()]

    if len(all_meta) < 5:
        print("  Insufficient features for SHAP analysis")
        return

    print(f"  Features for SHAP: {len(all_meta)}")

    # Build labels
    label_decay = 2.0
    known_w_soft = {}
    for c, info in KNOWN_EPISODES.items():
        for y in range(info["onset"] - LEAD_YEARS, info["onset"] + 1):
            dist = max(0, info["onset"] - y)
            known_w_soft[(c, y)] = np.exp(-dist / label_decay)
    ews_df["label_soft"] = ews_df.apply(
        lambda r: known_w_soft.get((r["country_name"], r["year"]), 0.0), axis=1
    )
    ews_df["label"] = (ews_df["label_soft"] > 0.05).astype(int)

    X = ews_df[all_meta].fillna(0).values
    y = ews_df["label"].values
    train_mask = ews_df["year"] <= TRAIN_CUTOFF

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elastic net feature selection
    half_life = 8
    max_year = ews_df.loc[train_mask, "year"].max()
    time_weights = np.exp(-np.log(2) * (max_year - ews_df["year"].values) / half_life)

    enet = SGDClassifier(loss="log_loss", penalty="elasticnet", l1_ratio=0.5,
                         alpha=0.001, max_iter=2000, random_state=42, class_weight="balanced")
    enet.fit(X_scaled[train_mask], y[train_mask], sample_weight=time_weights[train_mask])
    selected_mask = np.abs(enet.coef_[0]) > 1e-4
    selected_features = [f for f, s in zip(all_meta, selected_mask) if s]
    X_selected = X_scaled[:, selected_mask]
    print(f"  Selected features: {len(selected_features)}")

    # Fit LR + GB
    lr = LogisticRegressionCV(cv=3, scoring="average_precision", max_iter=1000, random_state=42)
    lr.fit(X_selected[train_mask], y[train_mask], sample_weight=time_weights[train_mask])

    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, min_samples_leaf=20, random_state=42)
    gb.fit(X_selected[train_mask], y[train_mask], sample_weight=time_weights[train_mask])

    # Stacking weights (simplified — use 0.2/0.8 as found in main run)
    w_lr, w_gb = 0.2, 0.8

    # Define feature groups
    feature_groups = {
        "CSD (univariate)": [f for f in selected_features if "csd_index" in f and "mv_" not in f and "x_" not in f],
        "CSD (multivariate)": [f for f in selected_features if "mv_csd" in f],
        "Election vulnerability": [f for f in selected_features if "election" in f or "party_threat" in f],
        "Military threat": [f for f in selected_features if "mil_" in f],
        "Network exposure": [f for f in selected_features if "network" in f or "contagion" in f or f == "csd_x_network" or f == "csd_x_network_detrended"],
        "GDELT events": [f for f in selected_features if "protest" in f or "conflict" in f or "repression" in f],
        "DSP (digital)": [f for f in selected_features if f.startswith("v2sm")],
        "Institutional erosion": [f for f in selected_features if "v2jun" in f or "v2xlg" in f or "v2x_ju" in f or "v2exres" in f],
        "Interactions": [f for f in selected_features if "csd_x_election" in f or "csd_x_military" in f],
    }

    # Remove empty groups
    feature_groups = {k: v for k, v in feature_groups.items() if v}

    print(f"\n  Feature groups:")
    for grp, members in feature_groups.items():
        print(f"    {grp}: {len(members)} features")

    # Option B: weighted SHAP (faster, exact for linear blend)
    # TreeSHAP for GB, LinearExplainer for LR
    try:
        import shap

        print(f"\n  Computing SHAP values (TreeSHAP for GB, Linear for LR)...")

        # GB SHAP (exact via TreeSHAP)
        gb_explainer = shap.TreeExplainer(gb)
        gb_shap = gb_explainer.shap_values(X_selected)
        if isinstance(gb_shap, list):
            gb_shap = gb_shap[1]  # class 1 SHAP values

        # LR SHAP (exact via LinearExplainer)
        lr_explainer = shap.LinearExplainer(lr, X_selected[train_mask])
        lr_shap = lr_explainer.shap_values(X_selected)

        # Weighted combination
        ensemble_shap = w_lr * lr_shap + w_gb * gb_shap

        print(f"  SHAP values computed: {ensemble_shap.shape}")

        # Grouped importance
        name_to_idx = {name: i for i, name in enumerate(selected_features)}
        print(f"\n  Grouped SHAP importance (mean |SHAP| summed within group):")
        group_results = []
        for grp, members in sorted(feature_groups.items()):
            col_idx = [name_to_idx[m] for m in members if m in name_to_idx]
            if col_idx:
                importance = np.abs(ensemble_shap[:, col_idx]).sum(axis=1).mean()
                group_results.append({"group": grp, "importance": importance, "n_features": len(col_idx)})
                print(f"    {grp}: {importance:.4f} ({len(col_idx)} features)")

        # Top individual features
        feat_importance = np.abs(ensemble_shap).mean(axis=0)
        top_idx = np.argsort(feat_importance)[::-1][:15]
        print(f"\n  Top 15 individual features by mean |SHAP|:")
        for i in top_idx:
            print(f"    {selected_features[i]}: {feat_importance[i]:.4f}")

        # Save results
        group_df = pd.DataFrame(group_results).sort_values("importance", ascending=False)
        group_df.to_csv(os.path.join(OUTPUT_DIR, "shap_grouped_importance.csv"), index=False)

        feat_df = pd.DataFrame({
            "feature": selected_features,
            "mean_abs_shap": feat_importance,
        }).sort_values("mean_abs_shap", ascending=False)
        feat_df.to_csv(os.path.join(OUTPUT_DIR, "shap_feature_importance.csv"), index=False)

        print(f"\n  Saved to robustness/shap_grouped_importance.csv")
        print(f"  Saved to robustness/shap_feature_importance.csv")

        return group_df, feat_df

    except ImportError:
        print("  shap library not installed. Install with: pip install shap")
        return None, None


if __name__ == "__main__":
    run_shap_analysis()
