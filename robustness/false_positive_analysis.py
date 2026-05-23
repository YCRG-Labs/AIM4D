"""
False positive classification and analysis.

Classifies every false positive alert as:
  1. Temporal near-miss: episode occurred within 3 years after the alert window
  2. Near-miss: country experienced polyarchy decline (>0.05) but no formal episode
  3. True false positive: no deterioration

Methodological basis:
  - Hegre et al. (2019, ViEWS): FP discussion in conflict forecasting
  - Ward & Beger (2017): near-misses as useful predictions
  - Cederman & Weidmann (2017): recall vs precision tradeoff for policy
  - Greenhill, Ward & Sacks (2011, AJPS): separation plots for rare events

Reports:
  - Full FP table with classification, polyarchy change, risk score
  - Breakdown by FP type
  - Stable democracy false alarm rate
  - Regional distribution of FPs
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stage5_ews.estimate import KNOWN_EPISODES, LEAD_YEARS, TRAIN_CUTOFF

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

STABLE_DEMOCRACIES = [
    "Denmark", "Sweden", "Norway", "Switzerland", "Finland",
    "Germany", "Canada", "New Zealand", "Uruguay", "Belgium",
    "Iceland", "Australia", "Ireland", "Netherlands", "Austria",
    "Portugal", "Spain", "Italy", "France", "United Kingdom",
    "Japan", "South Korea", "Costa Rica", "Chile",
]


def load_data():
    """Load EWS signals and V-Dem polyarchy data."""
    vdem_path = os.path.join(os.path.dirname(__file__), "..", "data", "vdem_v16.csv")

    try:
        from stage5_ews.estimate import run_ews
        print("Running full stage 5 to get enriched EWS data...")
        ews = run_ews()
    except Exception as e:
        print(f"Could not run full stage 5 ({e}), falling back to saved CSV")
        ews_path = os.path.join(os.path.dirname(__file__), "..", "stage5_ews", "ews_signals.csv")
        ews = pd.read_csv(ews_path)

    vdem_cols = ["country_name", "country_text_id", "year", "v2x_polyarchy", "v2x_regime"]
    available = pd.read_csv(vdem_path, low_memory=False, nrows=1).columns
    vdem_cols = [c for c in vdem_cols if c in available]
    vdem = pd.read_csv(vdem_path, low_memory=False, usecols=vdem_cols)

    return ews, vdem


def identify_fps(ews, vdem):
    """Identify all false positive alerts and classify them."""
    known_w = {}
    for c, info in KNOWN_EPISODES.items():
        for y in range(info["onset"] - LEAD_YEARS, info["onset"] + 1):
            known_w[(c, y)] = True

    alert_col = "combined_alert" if "combined_alert" in ews.columns else "ews_alert"
    risk_col = "combined_risk" if "combined_risk" in ews.columns else "csd_index"

    alerts = ews[ews[alert_col] == True].copy()
    alerts["is_tp"] = alerts.apply(
        lambda r: (r["country_name"], r["year"]) in known_w, axis=1
    )
    fps = alerts[~alerts["is_tp"]].copy()

    print(f"Total alerts: {len(alerts)}")
    print(f"True positives: {alerts['is_tp'].sum()}")
    print(f"False positives: {len(fps)}")

    vdem_sub = vdem[["country_name", "year", "v2x_polyarchy"]].copy()

    fp_rows = []
    for _, fp_row in fps.iterrows():
        country = fp_row["country_name"]
        year = fp_row["year"]

        poly_now = vdem_sub[(vdem_sub["country_name"] == country) &
                            (vdem_sub["year"] == year)]["v2x_polyarchy"]
        poly_after = vdem_sub[(vdem_sub["country_name"] == country) &
                              (vdem_sub["year"] >= year) &
                              (vdem_sub["year"] <= year + 5)]["v2x_polyarchy"]

        poly_val = poly_now.iloc[0] if len(poly_now) > 0 else np.nan
        poly_min_after = poly_after.min() if len(poly_after) > 0 else np.nan
        poly_change = poly_min_after - poly_val if not np.isnan(poly_val) and not np.isnan(poly_min_after) else np.nan

        temporal_near_miss = False
        for ep_country, ep_info in KNOWN_EPISODES.items():
            if ep_country == country and year < ep_info["onset"] <= year + 3:
                temporal_near_miss = True
                break

        if temporal_near_miss:
            fp_type = "temporal_near_miss"
        elif not np.isnan(poly_change) and poly_change < -0.05:
            fp_type = "near_miss"
        elif country in STABLE_DEMOCRACIES:
            fp_type = "stable_democracy_fp"
        else:
            fp_type = "true_fp"

        risk = fp_row.get(risk_col, fp_row.get("csd_index", np.nan))

        fp_rows.append({
            "country_name": country,
            "country_text_id": fp_row.get("country_text_id", ""),
            "year": int(year),
            "risk_score": risk,
            "csd_index": fp_row.get("csd_index", np.nan),
            "polyarchy_at_alert": poly_val,
            "polyarchy_change_5yr": poly_change,
            "fp_type": fp_type,
            "is_stable_democracy": country in STABLE_DEMOCRACIES,
        })

    return pd.DataFrame(fp_rows)


def run_false_positive_analysis():
    print("=" * 70)
    print("ROBUSTNESS CHECK: False Positive Analysis")
    print("=" * 70)
    print()
    print("Methodological basis:")
    print("  Hegre et al. (2019) FP in conflict forecasting;")
    print("  Ward & Beger (2017) near-misses as useful predictions;")
    print("  Greenhill et al. (2011) separation plots")
    print()

    ews, vdem = load_data()
    print(f"EWS data: {len(ews)} country-years")
    print(f"V-Dem data: {len(vdem)} country-years")

    fp_table = identify_fps(ews, vdem)

    if len(fp_table) == 0:
        print("\nNo false positives found.")
        return pd.DataFrame()

    print(f"\n{'='*50}")
    print("FALSE POSITIVE CLASSIFICATION")
    print(f"{'='*50}")

    type_counts = fp_table["fp_type"].value_counts()
    total_fp = len(fp_table)
    for fp_type, count in type_counts.items():
        pct = count / total_fp * 100
        print(f"  {fp_type}: {count} ({pct:.1f}%)")

    temporal_nm = fp_table[fp_table["fp_type"] == "temporal_near_miss"]
    if len(temporal_nm) > 0:
        print(f"\n  Temporal near-misses (episode within 3yr after alert):")
        for _, row in temporal_nm.drop_duplicates("country_name").iterrows():
            print(f"    {row['country_name']} ({int(row['year'])}): "
                  f"polyarchy change = {row['polyarchy_change_5yr']:.3f}")

    near_miss = fp_table[fp_table["fp_type"] == "near_miss"]
    if len(near_miss) > 0:
        print(f"\n  Near-misses (polyarchy declined >0.05 within 5yr):")
        for _, row in near_miss.drop_duplicates("country_name").head(15).iterrows():
            print(f"    {row['country_name']} ({int(row['year'])}): "
                  f"polyarchy {row['polyarchy_at_alert']:.3f} -> "
                  f"change {row['polyarchy_change_5yr']:.3f}, "
                  f"risk={row['risk_score']:.2f}")

    true_fp = fp_table[fp_table["fp_type"] == "true_fp"]
    if len(true_fp) > 0:
        print(f"\n  True false positives (no deterioration):")
        for _, row in true_fp.drop_duplicates("country_name").head(15).iterrows():
            poly_str = f"polyarchy {row['polyarchy_at_alert']:.3f}" if not np.isnan(row['polyarchy_at_alert']) else "polyarchy N/A"
            change_str = f"change {row['polyarchy_change_5yr']:.3f}" if not np.isnan(row['polyarchy_change_5yr']) else "change N/A"
            print(f"    {row['country_name']} ({int(row['year'])}): "
                  f"{poly_str}, {change_str}, risk={row['risk_score']:.2f}")

    stable_fp = fp_table[fp_table["is_stable_democracy"]]
    print(f"\n  Stable democracy false alarms: {len(stable_fp)}")
    if len(stable_fp) > 0:
        for _, row in stable_fp.drop_duplicates("country_name").iterrows():
            print(f"    {row['country_name']} ({int(row['year'])}): risk={row['risk_score']:.2f}")

    n_temporal_nm = len(fp_table[fp_table["fp_type"] == "temporal_near_miss"])
    n_near_miss = len(fp_table[fp_table["fp_type"] == "near_miss"])
    n_true_fp = len(fp_table[fp_table["fp_type"].isin(["true_fp", "stable_democracy_fp"])])

    alert_col = "combined_alert" if "combined_alert" in ews.columns else "ews_alert"
    known_w = {}
    for c, info in KNOWN_EPISODES.items():
        for y in range(info["onset"] - LEAD_YEARS, info["onset"] + 1):
            known_w[(c, y)] = True
    alerts = ews[ews[alert_col] == True]
    n_tp = alerts.apply(lambda r: (r["country_name"], r["year"]) in known_w, axis=1).sum()

    standard_precision = n_tp / (n_tp + total_fp) if (n_tp + total_fp) > 0 else 0
    adjusted_precision = (n_tp + n_temporal_nm + n_near_miss) / (n_tp + total_fp) if (n_tp + total_fp) > 0 else 0

    print(f"\n{'='*50}")
    print("PRECISION ANALYSIS")
    print(f"{'='*50}")
    print(f"  Standard precision (TP / all alerts): {standard_precision:.1%}")
    print(f"  Adjusted precision (TP+near-misses / all alerts): {adjusted_precision:.1%}")
    print(f"    Near-misses reclassified as useful: {n_temporal_nm + n_near_miss}")
    print(f"    Genuine false alarms: {n_true_fp}")

    print(f"\n{'='*50}")
    print("REGIONAL DISTRIBUTION OF FPs")
    print(f"{'='*50}")

    region_map = {}
    for _, row in fp_table.iterrows():
        iso = row.get("country_text_id", "")
        if len(iso) >= 2:
            region_map[row["country_name"]] = iso[:2]

    fp_table["region_code"] = fp_table["country_name"].map(region_map)
    region_dist = fp_table.groupby("region_code").agg(
        n_fps=("country_name", "count"),
        n_countries=("country_name", "nunique"),
    ).sort_values("n_fps", ascending=False)

    for region, row in region_dist.head(10).iterrows():
        print(f"  {region}: {row['n_fps']} FP alerts across {row['n_countries']} countries")

    print(f"\n{'='*50}")
    print("RISK SCORE DISTRIBUTION: TPs vs FPs")
    print(f"{'='*50}")

    risk_col = "risk_score"
    if risk_col in fp_table.columns:
        fp_risk = fp_table[risk_col].dropna()
        tp_alerts = alerts[alerts.apply(lambda r: (r["country_name"], r["year"]) in known_w, axis=1)]
        tp_risk_col = "combined_risk" if "combined_risk" in tp_alerts.columns else "csd_index"
        tp_risk = tp_alerts[tp_risk_col].dropna() if tp_risk_col in tp_alerts.columns else pd.Series()

        if len(fp_risk) > 0:
            print(f"  FP risk scores: median={fp_risk.median():.3f}, mean={fp_risk.mean():.3f}, "
                  f"range=[{fp_risk.min():.3f}, {fp_risk.max():.3f}]")
        if len(tp_risk) > 0:
            print(f"  TP risk scores: median={tp_risk.median():.3f}, mean={tp_risk.mean():.3f}, "
                  f"range=[{tp_risk.min():.3f}, {tp_risk.max():.3f}]")
        if len(fp_risk) > 0 and len(tp_risk) > 0:
            from scipy.stats import mannwhitneyu
            try:
                stat, p = mannwhitneyu(tp_risk, fp_risk, alternative="greater")
                print(f"  Mann-Whitney U test (TPs > FPs): U={stat:.0f}, p={p:.4f}")
                print(f"  {'WELL SEPARATED (p<0.01)' if p < 0.01 else 'MODERATE SEPARATION' if p < 0.05 else 'POOR SEPARATION'}")
            except Exception:
                pass

    fp_table.to_csv(os.path.join(OUTPUT_DIR, "false_positive_table.csv"), index=False)

    summary = pd.DataFrame([{
        "total_alerts": n_tp + total_fp,
        "true_positives": n_tp,
        "false_positives": total_fp,
        "temporal_near_misses": n_temporal_nm,
        "near_misses": n_near_miss,
        "true_false_positives": n_true_fp,
        "stable_democracy_fps": len(stable_fp),
        "standard_precision": standard_precision,
        "adjusted_precision": adjusted_precision,
    }])
    summary.to_csv(os.path.join(OUTPUT_DIR, "false_positive_summary.csv"), index=False)

    print(f"\nSaved to robustness/false_positive_table.csv")
    print(f"Saved to robustness/false_positive_summary.csv")

    return fp_table


if __name__ == "__main__":
    run_false_positive_analysis()
