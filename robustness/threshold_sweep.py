"""
Threshold sensitivity analysis + Thailand near-miss investigation.

Sweeps the EWS alert threshold to show results aren't threshold-dependent,
generates precision/recall/F1 curves, calibration analysis, and investigates
the Thailand LOEO near-miss (0.007 below threshold).

Methodological basis:
  - Saito & Rehmsmeier (2015): PR curves for imbalanced data
  - Cranmer & Desmarais (2017): threshold selection in polsci prediction
  - Niculescu-Mizil & Caruana (2005): calibration plots
  - Vickers & Elkin (2006): decision curve analysis

Reports:
  - Precision/recall/F1 at multiple thresholds
  - ROC + PR curve data
  - Calibration analysis (observed vs predicted risk)
  - Thailand-specific near-miss analysis
  - Multi-threshold stability table
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stage5_ews.estimate import KNOWN_EPISODES, LEAD_YEARS, TRAIN_CUTOFF

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_ews():
    """Load EWS signals — run full stage 5 to get all columns (meta-learner, election, military)."""
    try:
        from stage5_ews.estimate import run_ews
        print("Running full stage 5 to get enriched EWS data...")
        df = run_ews()
        return df
    except Exception as e:
        print(f"Could not run full stage 5 ({e}), falling back to saved CSV")
        path = os.path.join(os.path.dirname(__file__), "..", "stage5_ews", "ews_signals.csv")
        df = pd.read_csv(path)
        return df


def compute_labels(df):
    """Assign binary labels based on known episodes."""
    known_w = {}
    for c, info in KNOWN_EPISODES.items():
        for y in range(info["onset"] - LEAD_YEARS, info["onset"] + 1):
            known_w[(c, y)] = True
    df["label"] = df.apply(
        lambda r: 1 if (r["country_name"], r["year"]) in known_w else 0, axis=1
    )
    return df


def threshold_sweep(df, risk_col="combined_risk", n_thresholds=50):
    """Sweep thresholds and compute precision/recall/F1 at each."""
    valid = df.dropna(subset=[risk_col])
    if valid["label"].sum() == 0:
        return pd.DataFrame()

    thresholds = np.linspace(
        valid[risk_col].quantile(0.50),
        valid[risk_col].quantile(0.995),
        n_thresholds,
    )

    total_positive = valid["label"].sum()
    rows = []

    for thresh in thresholds:
        flagged = valid[valid[risk_col] >= thresh]
        tp = flagged["label"].sum()
        fp = len(flagged) - tp
        fn = total_positive - tp
        tn = len(valid) - len(flagged) - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        rows.append({
            "threshold": thresh,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_flagged": len(flagged),
        })

    return pd.DataFrame(rows)


def detection_by_threshold(df, risk_col="combined_risk", n_thresholds=20):
    """For each threshold, which episodes are detected?"""
    valid = df.dropna(subset=[risk_col])
    thresholds = np.linspace(
        valid[risk_col].quantile(0.70),
        valid[risk_col].quantile(0.99),
        n_thresholds,
    )

    episode_rows = []
    for country, info in KNOWN_EPISODES.items():
        onset = info["onset"]
        pre = valid[(valid["country_name"] == country) &
                    (valid["year"] >= onset - LEAD_YEARS) & (valid["year"] < onset)]
        if len(pre) == 0:
            continue
        max_risk = pre[risk_col].max()

        for thresh in thresholds:
            detected = max_risk >= thresh
            episode_rows.append({
                "country": country,
                "onset": onset,
                "type": info["type"],
                "max_risk": max_risk,
                "threshold": thresh,
                "detected": detected,
            })

    return pd.DataFrame(episode_rows)


def calibration_analysis(df, risk_col="combined_risk", n_bins=10):
    """Calibration: how well do predicted risks match observed frequencies?"""
    valid = df.dropna(subset=[risk_col])
    if valid["label"].sum() == 0:
        return pd.DataFrame()

    valid["bin"] = pd.qcut(valid[risk_col], n_bins, duplicates="drop")
    cal = valid.groupby("bin").agg(
        mean_predicted=(risk_col, "mean"),
        observed_freq=("label", "mean"),
        n=("label", "count"),
        n_positive=("label", "sum"),
    ).reset_index()

    return cal


def thailand_analysis(df, risk_col="combined_risk"):
    """Deep dive into Thailand's LOEO near-miss."""
    print(f"\n{'='*50}")
    print("THAILAND NEAR-MISS ANALYSIS")
    print(f"{'='*50}")

    thai = df[df["country_name"] == "Thailand"].sort_values("year")
    if len(thai) == 0:
        print("  Thailand not found in data")
        return

    onset = KNOWN_EPISODES.get("Thailand", {}).get("onset", 2014)
    pre = thai[(thai["year"] >= onset - LEAD_YEARS) & (thai["year"] < onset)]

    print(f"\n  Episode: coup, onset {onset}")
    print(f"  Pre-onset window ({onset - LEAD_YEARS}-{onset - 1}):")

    risk_cols = [c for c in ["csd_index", "combined_risk", "calibrated_risk",
                              "election_vulnerability", "mil_zscore"] if c in thai.columns]

    for _, row in pre.iterrows():
        vals = ", ".join(f"{c}={row[c]:.3f}" for c in risk_cols if not pd.isna(row[c]))
        alerts = []
        if row.get("ews_alert", False): alerts.append("CSD")
        if row.get("election_alert", False): alerts.append("ELEC")
        if row.get("military_threat_alert", False): alerts.append("MIL")
        alert_str = f" [{'+'.join(alerts)}]" if alerts else ""
        print(f"    {int(row['year'])}: {vals}{alert_str}")

    if risk_col in pre.columns:
        max_risk = pre[risk_col].max()
        all_risks = df[df["year"] <= TRAIN_CUTOFF][risk_col].dropna()
        pctile = (all_risks < max_risk).mean() * 100

        print(f"\n  Max pre-onset risk: {max_risk:.4f}")
        print(f"  Percentile in training data: {pctile:.1f}th")
        print(f"  P95 threshold: {all_risks.quantile(0.95):.4f}")
        print(f"  Gap to P95: {all_risks.quantile(0.95) - max_risk:.4f}")

        for p in [95, 94, 93, 92, 90]:
            t = all_risks.quantile(p / 100)
            detected = max_risk >= t
            print(f"  At P{p} ({t:.4f}): {'DETECTED' if detected else 'MISSED'}")

    print(f"\n  Comparison to other coups:")
    for country, info in KNOWN_EPISODES.items():
        if info["type"] != "coup":
            continue
        ep_pre = df[(df["country_name"] == country) &
                    (df["year"] >= info["onset"] - LEAD_YEARS) &
                    (df["year"] < info["onset"])]
        if len(ep_pre) > 0 and risk_col in ep_pre.columns:
            mr = ep_pre[risk_col].max()
            print(f"    {country} ({info['onset']}): max risk = {mr:.4f}")

    print(f"\n  Assessment: Thailand miss is {'a threshold artifact (within 1 percentile)' if pctile > 93 else 'a genuine detection failure'}")


def multi_stage_threshold_sensitivity(df):
    """
    Test sensitivity to the two most consequential thresholds:
    CSD z-score threshold and meta-learner percentile threshold.
    """
    print(f"\n{'='*50}")
    print("MULTI-STAGE THRESHOLD SENSITIVITY")
    print(f"{'='*50}")

    csd_thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    risk_percentiles = [90, 92, 94, 95, 96, 98]

    if "combined_risk" not in df.columns or "csd_index" not in df.columns:
        print("  Missing required columns")
        return pd.DataFrame()

    known_w = {}
    for c, info in KNOWN_EPISODES.items():
        for y in range(info["onset"] - LEAD_YEARS, info["onset"] + 1):
            known_w[(c, y)] = True

    rows = []
    for csd_t in csd_thresholds:
        for risk_p in risk_percentiles:
            risk_thresh = df[df["year"] <= TRAIN_CUTOFF]["combined_risk"].quantile(risk_p / 100)

            alert = (df["csd_index"] > csd_t) | (df["combined_risk"] > risk_thresh)

            hits, total = 0, 0
            for country, info in KNOWN_EPISODES.items():
                onset = info["onset"]
                pre = df[(df["country_name"] == country) &
                         (df["year"] >= onset - LEAD_YEARS) & (df["year"] < onset)]
                if len(pre) == 0:
                    continue
                total += 1
                if alert[pre.index].any():
                    hits += 1

            tp = df[alert]["label"].sum() if "label" in df.columns else 0
            fp = alert.sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = hits / total if total > 0 else 0
            f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0

            rows.append({
                "csd_threshold": csd_t,
                "risk_percentile": risk_p,
                "risk_threshold": risk_thresh,
                "detection_rate": recall,
                "detections": f"{hits}/{total}",
                "precision": prec,
                "f1": f1,
                "n_alerts": alert.sum(),
            })

    result = pd.DataFrame(rows)

    print(f"\n  Detection rate (rows=CSD threshold, cols=risk percentile):")
    pivot = result.pivot_table(values="detection_rate", index="csd_threshold",
                                columns="risk_percentile", aggfunc="first")
    print(pivot.to_string(float_format="%.2f"))

    print(f"\n  F1 score:")
    pivot_f1 = result.pivot_table(values="f1", index="csd_threshold",
                                   columns="risk_percentile", aggfunc="first")
    print(pivot_f1.to_string(float_format="%.3f"))

    return result


def run_threshold_sweep():
    print("=" * 70)
    print("ROBUSTNESS CHECK: Threshold Sensitivity Analysis")
    print("=" * 70)
    print()
    print("Methodological basis:")
    print("  Saito & Rehmsmeier (2015) PR curves; Cranmer & Desmarais (2017) threshold;")
    print("  Niculescu-Mizil & Caruana (2005) calibration")
    print()

    df = load_ews()
    df = compute_labels(df)
    print(f"Loaded {len(df)} country-years, {df['label'].sum()} positive labels")

    risk_col = "combined_risk"
    if risk_col not in df.columns:
        risk_col = "csd_index"
    print(f"Using risk column: {risk_col}")

    print(f"\n{'='*50}")
    print("THRESHOLD SWEEP")
    print(f"{'='*50}")
    sweep = threshold_sweep(df, risk_col)
    if len(sweep) > 0:
        print(f"\n  {'Threshold':>10}  {'Precision':>9}  {'Recall':>6}  {'F1':>5}  {'FPR':>5}  {'Flagged':>7}")
        print(f"  {'-'*50}")
        for _, row in sweep.iloc[::max(1, len(sweep) // 10)].iterrows():
            print(f"  {row['threshold']:10.3f}  {row['precision']:9.3f}  {row['recall']:6.3f}  "
                  f"{row['f1']:5.3f}  {row['fpr']:5.3f}  {int(row['n_flagged']):7d}")

        best = sweep.loc[sweep["f1"].idxmax()]
        print(f"\n  Best F1: {best['f1']:.3f} at threshold {best['threshold']:.3f}")
        print(f"    Precision={best['precision']:.3f}, Recall={best['recall']:.3f}")

        good_range = sweep[sweep["f1"] >= best["f1"] * 0.9]
        if len(good_range) > 1:
            print(f"  Thresholds within 90% of best F1: [{good_range['threshold'].min():.3f}, {good_range['threshold'].max():.3f}]")
            print(f"  {'STABLE: wide optimal range' if (good_range['threshold'].max() - good_range['threshold'].min()) > 0.5 else 'NARROW: threshold-sensitive'}")

    print(f"\n{'='*50}")
    print("EPISODE DETECTION BY THRESHOLD")
    print(f"{'='*50}")
    ep_table = detection_by_threshold(df, risk_col)
    if len(ep_table) > 0:
        print(f"\n  Episode robustness (threshold at which detection fails):")
        for country in sorted(ep_table["country"].unique()):
            ep = ep_table[ep_table["country"] == country]
            detected = ep[ep["detected"]]
            if len(detected) > 0:
                max_thresh = detected["threshold"].max()
                max_risk = ep["max_risk"].iloc[0]
                print(f"    {country} ({ep['type'].iloc[0]}, {ep['onset'].iloc[0]}): "
                      f"max_risk={max_risk:.3f}, fails at threshold>{max_thresh:.3f}")
            else:
                print(f"    {country}: never detected at any tested threshold")

    print(f"\n{'='*50}")
    print("CALIBRATION ANALYSIS")
    print(f"{'='*50}")
    cal = calibration_analysis(df, risk_col)
    if len(cal) > 0:
        print(f"\n  {'Predicted':>10}  {'Observed':>10}  {'N':>6}  {'Positive':>8}")
        print(f"  {'-'*40}")
        for _, row in cal.iterrows():
            print(f"  {row['mean_predicted']:10.4f}  {row['observed_freq']:10.4f}  "
                  f"{int(row['n']):6d}  {int(row['n_positive']):8d}")

        cal_error = (cal["mean_predicted"] - cal["observed_freq"]).abs().mean()
        print(f"\n  Mean absolute calibration error: {cal_error:.4f}")
        print(f"  {'WELL CALIBRATED (<0.05)' if cal_error < 0.05 else 'MODERATE (0.05-0.10)' if cal_error < 0.10 else 'POORLY CALIBRATED (>0.10)'}")

    thailand_analysis(df, risk_col)

    ms_result = multi_stage_threshold_sensitivity(df)

    if len(sweep) > 0:
        sweep.to_csv(os.path.join(OUTPUT_DIR, "threshold_sweep.csv"), index=False)
    if len(ep_table) > 0:
        ep_table.to_csv(os.path.join(OUTPUT_DIR, "episode_detection_by_threshold.csv"), index=False)
    if len(cal) > 0:
        cal.to_csv(os.path.join(OUTPUT_DIR, "calibration_analysis.csv"), index=False)
    if len(ms_result) > 0:
        ms_result.to_csv(os.path.join(OUTPUT_DIR, "multistage_threshold_sensitivity.csv"), index=False)

    print(f"\nSaved all threshold analysis results to robustness/")

    return sweep


if __name__ == "__main__":
    run_threshold_sweep()
