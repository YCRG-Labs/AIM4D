"""
OOS permutation feature importance for the AIM4D meta-learner.

Replaces in-sample SHAP / GB-feature-importance with shuffle-based importance
computed on the held-out (year > TRAIN_CUTOFF, post-onset excluded) slice.

For each feature:
  1. Compute baseline OOS AUC-ROC and AUC-PR
  2. Shuffle the feature's OOS values
  3. Recompute OOS AUC / AUC-PR
  4. Delta = baseline - shuffled (positive means feature matters)
  5. Repeat N_PERMS times for stability, average

Why OOS not in-sample? In-sample importances can reward features that overfit;
OOS permutation tells you what genuinely transports.

Why permutation not SHAP? SHAP measures attribution within a model; permutation
measures information loss in the predictive signal. The latter is what reviewers
actually care about for forecasting claims.

Output: robustness/permutation_importance_oos.csv with columns
  feature, mean_delta_auc, std_delta_auc, mean_delta_auc_pr, std_delta_auc_pr, rank
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from stage5_ews.estimate import KNOWN_EPISODES, TRAIN_CUTOFF, POS_WEIGHT, LEAD_YEARS, lead_for

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "permutation_importance_oos.csv")
EWS_PATH = os.path.join(REPO, "stage5_ews", "ews_signals.csv")

N_PERMS = 10
RNG = np.random.default_rng(42)

EXCLUDE_COLS = {
    "country_name", "country_text_id", "year", "label", "label_soft",
    "combined_risk", "calibrated_risk", "smoothed_risk", "alert_tier",
    "combined_alert", "combined_alert_legacy", "ews_alert", "raw_alert",
    "election_alert", "dem_vulnerability_alert", "military_threat_alert",
    "mv_csd_alert", "n_factors", "is_postonset", "is_episode_pre",
    "years_to_onset",
}


def build_label_columns(df):
    """Reconstruct the binary label + post-onset mask from KNOWN_EPISODES."""
    preonset = set()
    postonset = set()
    for c, info in KNOWN_EPISODES.items():
        onset = info["onset"]
        lead = lead_for(info)
        for y in range(onset - lead, onset + 1):
            preonset.add((c, y))
        for y in range(onset + 1, onset + 6):
            postonset.add((c, y))
    df["label"] = df.apply(
        lambda r: 1 if (r["country_name"], r["year"]) in preonset else 0, axis=1
    )
    df["is_postonset"] = df.apply(
        lambda r: (r["country_name"], r["year"]) in postonset, axis=1
    )
    return df


def main():
    if not os.path.exists(EWS_PATH):
        sys.exit(f"Missing {EWS_PATH} — run stage5_ews/estimate.py first")

    ews = pd.read_csv(EWS_PATH)
    if "label" not in ews.columns or "is_postonset" not in ews.columns:
        print("  Rebuilding label / is_postonset columns from KNOWN_EPISODES")
        ews = build_label_columns(ews)

    ews = ews[ews["year"] >= 2000].reset_index(drop=True)

    features = [c for c in ews.columns
                if c not in EXCLUDE_COLS
                and pd.api.types.is_numeric_dtype(ews[c])]
    print(f"Loaded {len(ews)} country-years, {len(features)} candidate features")

    X = ews[features].fillna(0).values
    y = ews["label"].astype(int).values
    train_mask = (ews["year"] <= TRAIN_CUTOFF).values & (~ews["is_postonset"].fillna(False).values)
    oos_mask = (ews["year"] > TRAIN_CUTOFF).values & (~ews["is_postonset"].fillna(False).values)

    scaler = StandardScaler()
    scaler.fit(X[train_mask])
    Xs = scaler.transform(X)

    max_year = int(ews.loc[train_mask, "year"].max())
    time_weights = np.exp(-np.log(2) * (max_year - ews["year"].values) / 8.0)
    train_w = time_weights * np.where(y == 1, POS_WEIGHT, 1.0)

    print("Fitting stacked ensemble on train (post-onset excluded)...")
    lr = LogisticRegressionCV(cv=3, scoring="average_precision", max_iter=1000, random_state=42)
    lr.fit(Xs[train_mask], y[train_mask], sample_weight=train_w[train_mask])

    N_GB = 20
    gb_preds_oos = []
    gb_preds_full = []
    for seed in range(N_GB):
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20, random_state=seed,
        )
        gb.fit(Xs[train_mask], y[train_mask], sample_weight=train_w[train_mask])
        gb_preds_full.append(gb.predict_proba(Xs)[:, 1])
        gb_preds_oos.append(gb.predict_proba(Xs)[:, 1])
    gb_pred = np.mean(gb_preds_full, axis=0)

    rf = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=10,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(Xs[train_mask], y[train_mask], sample_weight=train_w[train_mask])
    rf_pred = rf.predict_proba(Xs)[:, 1]

    et = ExtraTreesClassifier(n_estimators=500, max_depth=10, min_samples_leaf=10,
                               class_weight="balanced", random_state=42, n_jobs=-1)
    et.fit(Xs[train_mask], y[train_mask], sample_weight=train_w[train_mask])
    et_pred = et.predict_proba(Xs)[:, 1]

    try:
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier(
            iterations=1500, depth=5, learning_rate=0.03, l2_leaf_reg=5,
            bootstrap_type="Bayesian", bagging_temperature=1.0,
            auto_class_weights="SqrtBalanced", random_seed=42,
            verbose=0, allow_writing_files=False,
        )
        cb.fit(Xs[train_mask], y[train_mask], sample_weight=train_w[train_mask])
        cb_pred = cb.predict_proba(Xs)[:, 1]
        have_cb = True
    except Exception as e:
        print(f"  CatBoost skipped: {e}")
        cb_pred = None
        have_cb = False

    lr_pred = lr.predict_proba(Xs)[:, 1]

    if have_cb:
        ens = 0.14 * lr_pred + 0.56 * gb_pred + 0.10 * rf_pred + 0.10 * et_pred + 0.10 * cb_pred
    else:
        ens = 0.20 * lr_pred + 0.65 * gb_pred + 0.075 * rf_pred + 0.075 * et_pred

    baseline_auc = roc_auc_score(y[oos_mask], ens[oos_mask])
    baseline_ap = average_precision_score(y[oos_mask], ens[oos_mask])
    print(f"\nBaseline OOS AUC: {baseline_auc:.4f}  AUC-PR: {baseline_ap:.4f}")
    print(f"\nPermutation importance with {N_PERMS} shuffles per feature...")

    trained_gb_models = []
    for seed in range(N_GB):
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20, random_state=seed,
        )
        gb.fit(Xs[train_mask], y[train_mask], sample_weight=train_w[train_mask])
        trained_gb_models.append(gb)

    def oos_ens_pred(X_input):
        lr_o = lr.predict_proba(X_input)[:, 1]
        gb_o = np.mean([m.predict_proba(X_input)[:, 1] for m in trained_gb_models], axis=0)
        rf_o = rf.predict_proba(X_input)[:, 1]
        et_o = et.predict_proba(X_input)[:, 1]
        if have_cb:
            cb_o = cb.predict_proba(X_input)[:, 1]
            return 0.14 * lr_o + 0.56 * gb_o + 0.10 * rf_o + 0.10 * et_o + 0.10 * cb_o
        return 0.20 * lr_o + 0.65 * gb_o + 0.075 * rf_o + 0.075 * et_o

    oos_idx = np.where(oos_mask)[0]
    y_oos = y[oos_mask]

    rows = []
    for fi, feat in enumerate(features):
        deltas_auc = []
        deltas_ap = []
        for k in range(N_PERMS):
            rng = np.random.default_rng(1000 * fi + k)
            X_perm = Xs[oos_idx].copy()
            col = X_perm[:, fi].copy()
            rng.shuffle(col)
            X_perm[:, fi] = col
            preds = oos_ens_pred(X_perm)
            try:
                auc_p = roc_auc_score(y_oos, preds)
                ap_p = average_precision_score(y_oos, preds)
            except ValueError:
                continue
            deltas_auc.append(baseline_auc - auc_p)
            deltas_ap.append(baseline_ap - ap_p)
        if deltas_auc:
            rows.append({
                "feature": feat,
                "mean_delta_auc": float(np.mean(deltas_auc)),
                "std_delta_auc": float(np.std(deltas_auc, ddof=1)) if len(deltas_auc) > 1 else 0.0,
                "mean_delta_auc_pr": float(np.mean(deltas_ap)),
                "std_delta_auc_pr": float(np.std(deltas_ap, ddof=1)) if len(deltas_ap) > 1 else 0.0,
            })
        if (fi + 1) % 25 == 0:
            print(f"  {fi+1}/{len(features)} features done")

    df = pd.DataFrame(rows).sort_values("mean_delta_auc", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df.to_csv(OUT, index=False)

    print("\n" + "=" * 78)
    print("TOP 20 OOS PERMUTATION IMPORTANCES (delta OOS AUC when shuffled)")
    print("=" * 78)
    print(f"{'#':>3s}  {'feature':<45s}  {'Δ AUC':>10s}  {'± std':>8s}  {'Δ AUC-PR':>10s}")
    for _, r in df.head(20).iterrows():
        print(f"{int(r['rank']):>3d}  {r['feature'][:45]:<45s}  "
              f"{r['mean_delta_auc']:>+10.5f}  {r['std_delta_auc']:>8.5f}  "
              f"{r['mean_delta_auc_pr']:>+10.5f}")
    print(f"\nWrote {OUT}")
    print(f"Baseline OOS AUC: {baseline_auc:.4f}  AUC-PR: {baseline_ap:.4f}")


if __name__ == "__main__":
    main()
