"""
Symmetric channel ablation.

Runs the DSP ablation protocol of dsp_ablation.py over three feature blocks
rather than one, so that leave-one-block-out and block-only results are
directly comparable on identical data, sample weights, folds and seeds:

    digital control (DSP)   v2smgovdom, v2smfordom, v2smgovfilprc,
                            v2smgovsmmon, v2smpardom and derived terms
    mobilization            v2ca* (pro-, anti- and general mobilization)
    latent factor dynamics  factor_*, f1_*, *_c22_*

The asymmetry this addresses: dsp_ablation.py establishes that removing the
DSP block costs little, but never tests whether removing the mobilization
block costs any more. In a correlated 229-feature space that comparison is
the whole content of a redundancy claim.

Block sizes are unequal, so block-only scores are read alongside n_features.

Outputs robustness/channel_ablation.csv.
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


def is_mob(col):
    return col.lower().startswith("v2ca")


def is_factor(col):
    c = col.lower()
    return c.startswith("factor_") or c.startswith("f1_") or "_c22_" in c


def is_mob_dem(col):
    return col.lower().startswith("v2cademmob")


def is_mob_aut(col):
    return col.lower().startswith("v2caautmob")


def is_mob_gen(col):
    c = col.lower()
    return c.startswith("v2cagenmob") or c.startswith("v2caconmob")


BLOCKS = {
    "dsp": is_dsp,
    "mob": is_mob,
    "factor": is_factor,
    "mobdem": is_mob_dem,
    "mobaut": is_mob_aut,
    "mobgen": is_mob_gen,
}


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
        "auc_roc_oos": float(auc_oos) if not np.isnan(auc_oos) else np.nan,
        "auc_pr_oos": float(ap_oos) if not np.isnan(ap_oos) else np.nan,
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

    print(f"Total candidate features: {len(full_features)}")
    block_features = {}
    for name, pred in BLOCKS.items():
        block_features[name] = [c for c in full_features if pred(c)]
        print(f"  {name:<8} block: {len(block_features[name]):>3} features")
    print()

    configs = [("full", full_features)]
    for name, feats in block_features.items():
        configs.append((f"ablate_{name}", [c for c in full_features if c not in set(feats)]))
        configs.append((f"{name}_only", feats))

    rows = []
    for name, features in configs:
        if not features:
            continue
        X = df[features].fillna(0).values
        risk = stage5_ensemble(X, y, sample_weight, train_mask)
        m = evaluate(risk, df, y)
        m["configuration"] = name
        m["n_features"] = len(features)
        print(f"  {name:<14}  n={len(features):>3}  "
              f"AUC={m['auc_roc']:.3f}  AUC-PR={m['auc_pr']:.3f}  "
              f"OOS AUC={m['auc_roc_oos']:.3f}  OOS AUC-PR={m['auc_pr_oos']:.3f}  "
              f"watch={m['watch_detected']}/{m['watch_total']}")
        rows.append(m)

    full = next(r for r in rows if r["configuration"] == "full")
    print("\n  leave-one-block-out deltas (ablate minus full), OOS:")
    for name in BLOCKS:
        try:
            abl = next(r for r in rows if r["configuration"] == f"ablate_{name}")
        except StopIteration:
            continue
        d_auc = abl["auc_roc_oos"] - full["auc_roc_oos"]
        d_pr = abl["auc_pr_oos"] - full["auc_pr_oos"]
        rows.append({
            "configuration": f"delta_ablate_{name}_minus_full",
            "n_features": abl["n_features"] - full["n_features"],
            "auc_roc": abl["auc_roc"] - full["auc_roc"],
            "auc_pr": abl["auc_pr"] - full["auc_pr"],
            "auc_roc_oos": d_auc,
            "auc_pr_oos": d_pr,
            "watch_detected": abl["watch_detected"] - full["watch_detected"],
            "watch_total": full["watch_total"],
            "watch_sensitivity": abl["watch_sensitivity"] - full["watch_sensitivity"],
        })
        print(f"    {name:<8}  d_OOS_AUC={d_auc:+.4f}   d_OOS_AUC_PR={d_pr:+.4f}")

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT, "channel_ablation.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
