"""
DSP feature ablation.

Tests whether the Digital Society Project variables (v2smgovdom, v2smfordom,
v2smgovfilprc, v2smgovsmmon, v2smpardom and their derived terms) carry
distinct predictive signal beyond the rest of the AIM4D feature set.

Reproduces the Stage 5 stacked ensemble (LR + GB, 0.20/0.80 weights) on
the engineered feature matrix saved in ews_signals.csv, with and without
all DSP-derived columns. Reports AUC-ROC, AUC-PR, OOS AUC, and per-episode
detection at the watch tier.

Outputs robustness/dsp_ablation.csv.
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

OUT = os.path.dirname(os.path.abspath(__file__))
TRAIN_CUTOFF = 2019
LEAD_YEARS = 5
RANDOM_STATE = 42

DSP_PREFIXES = ("v2smgovdom", "v2smfordom", "v2smgovfilprc", "v2smgovsmmon", "v2smpardom")

EXCLUDE_COLS = {
    "country_name", "country_text_id", "year", "label", "label_soft",
    "combined_risk", "calibrated_risk", "alert_tier",
    "combined_alert", "combined_alert_legacy", "ews_alert", "raw_alert",
    "election_alert", "dem_vulnerability_alert", "military_threat_alert",
    "mv_csd_alert", "n_factors",
}


def candidate_features(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def is_dsp(col):
    return any(col.startswith(p) for p in DSP_PREFIXES)


def stage5_ensemble(X, y, sample_weight, train_mask):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(Xs[train_mask], y[train_mask], sample_weight=sample_weight[train_mask])
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=20, random_state=RANDOM_STATE,
    )
    gb.fit(Xs[train_mask], y[train_mask], sample_weight=sample_weight[train_mask])

    p_lr = lr.predict_proba(Xs)[:, 1]
    p_gb = gb.predict_proba(Xs)[:, 1]
    return 0.2 * p_lr + 0.8 * p_gb


def evaluate(risk, df, y):
    valid = ~np.isnan(risk)
    auc = roc_auc_score(y[valid], risk[valid])
    ap = average_precision_score(y[valid], risk[valid])

    oos = (df["year"] > TRAIN_CUTOFF).values & valid
    if "is_postonset" in df.columns:
        oos = oos & (~df["is_postonset"].fillna(False).values)
    if oos.sum() > 10 and y[oos].sum() > 1:
        auc_oos = roc_auc_score(y[oos], risk[oos])
        ap_oos = average_precision_score(y[oos], risk[oos])
    else:
        auc_oos = ap_oos = np.nan

    train_risk = risk[(df["year"] <= TRAIN_CUTOFF).values & valid]
    thresh_watch = np.quantile(train_risk, 0.80)

    try:
        from stage5_ews.estimate import KNOWN_EPISODES
    except Exception:
        KNOWN_EPISODES = {}

    detected = 0
    total = 0
    for country, info in KNOWN_EPISODES.items():
        onset = info["onset"]
        mask = ((df["country_name"] == country)
                & (df["year"] >= onset - LEAD_YEARS)
                & (df["year"] < onset)).values
        if mask.sum() == 0:
            continue
        total += 1
        if risk[mask].max() >= thresh_watch:
            detected += 1

    return {
        "auc_roc": float(auc),
        "auc_pr": float(ap),
        "auc_roc_oos_2017": float(auc_oos) if not np.isnan(auc_oos) else np.nan,
        "auc_pr_oos_2017": float(ap_oos) if not np.isnan(ap_oos) else np.nan,
        "watch_detected": detected,
        "watch_total": total,
        "watch_sensitivity": detected / total if total else np.nan,
    }


def main():
    ews_path = os.path.join(OUT, "..", "stage5_ews", "ews_signals.csv")
    df = pd.read_csv(ews_path)
    if "label" not in df.columns or "combined_risk" not in df.columns:
        raise RuntimeError("ews_signals.csv missing label or combined_risk; rerun stage 5")

    df = df.dropna(subset=["label"])
    y = df["label"].astype(int).values

    max_year = df["year"].max()
    sample_weight = np.exp(-np.log(2) * (max_year - df["year"].values) / 7.0)

    train_mask = (df["year"] <= TRAIN_CUTOFF).values
    if "is_postonset" in df.columns:
        train_mask = train_mask & (~df["is_postonset"].fillna(False).values)

    full_features = candidate_features(df)
    dsp_features = [c for c in full_features if is_dsp(c)]
    nondsp_features = [c for c in full_features if not is_dsp(c)]

    print(f"Total candidate features: {len(full_features)}")
    print(f"  DSP-derived: {len(dsp_features)}")
    print(f"  Non-DSP:     {len(nondsp_features)}\n")

    configs = [
        ("full", full_features),
        ("ablate_dsp", nondsp_features),
        ("dsp_only", dsp_features),
    ]

    rows = []
    for name, features in configs:
        if not features:
            continue
        X = df[features].fillna(0).values
        risk = stage5_ensemble(X, y, sample_weight, train_mask)
        m = evaluate(risk, df, y)
        m["configuration"] = name
        m["n_features"] = len(features)
        print(f"  {name:<12}  n={len(features):>3}  "
              f"AUC={m['auc_roc']:.3f}  AUC-PR={m['auc_pr']:.3f}  "
              f"OOS AUC={m['auc_roc_oos_2017']:.3f}  "
              f"watch={m['watch_detected']}/{m['watch_total']}")
        rows.append(m)

    if len(rows) >= 2:
        full = next(r for r in rows if r["configuration"] == "full")
        abl = next(r for r in rows if r["configuration"] == "ablate_dsp")
        d_auc = abl["auc_roc"] - full["auc_roc"]
        d_pr = abl["auc_pr"] - full["auc_pr"]
        d_oos = abl["auc_roc_oos_2017"] - full["auc_roc_oos_2017"]
        d_watch = abl["watch_sensitivity"] - full["watch_sensitivity"]

        print(f"\n  Δ AUC-ROC (ablate − full): {d_auc:+.3f}")
        print(f"  Δ AUC-PR  (ablate − full): {d_pr:+.3f}")
        print(f"  Δ OOS AUC (ablate − full): {d_oos:+.3f}")
        print(f"  Δ watch   (ablate − full): {d_watch:+.1%}")

        verdict = ("DSP carries strong distinct signal" if d_auc <= -0.03
                   else "DSP modestly load-bearing" if d_auc <= -0.01
                   else "DSP redundant with other features")
        print(f"\n  Verdict: {verdict}")

        rows.append({
            "configuration": "delta_full_minus_ablate",
            "n_features": full["n_features"] - abl["n_features"],
            "auc_roc": d_auc, "auc_pr": d_pr,
            "auc_roc_oos_2017": d_oos, "auc_pr_oos_2017": np.nan,
            "watch_detected": full["watch_detected"] - abl["watch_detected"],
            "watch_total": full["watch_total"],
            "watch_sensitivity": d_watch,
        })

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT, "dsp_ablation.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
