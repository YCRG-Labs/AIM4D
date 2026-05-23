"""
Cross-validation against Polity V regime scores.

Validates that AIM4D predictions correlate with non-V-Dem outcome measures,
addressing the V-Dem circularity concern.

Tests: Does AIM4D combined_risk predict Polity score declines?
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_polity_validation():
    print("=" * 70)
    print("CROSS-VALIDATION: AIM4D vs Polity V Outcomes")
    print("=" * 70)
    print()

    vdem_path = os.path.join(os.path.dirname(__file__), "..", "data", "vdem_v16.csv")
    polity = pd.read_csv(vdem_path, low_memory=False,
                          usecols=["country_name", "country_text_id", "year", "e_polity2"])
    polity = polity.dropna(subset=["e_polity2"])
    polity["e_polity2"] = polity["e_polity2"].astype(float)

    polity["polity_change_3yr"] = polity.groupby("country_text_id")["e_polity2"].diff(3)
    polity["polity_change_5yr"] = polity.groupby("country_text_id")["e_polity2"].diff(5)
    polity["polity_decline_3yr"] = (polity["polity_change_3yr"] < -3).astype(int)
    polity["polity_decline_5yr"] = (polity["polity_change_5yr"] < -3).astype(int)

    print(f"Polity data: {len(polity)} country-years")
    print(f"  3yr declines (>3 points): {polity['polity_decline_3yr'].sum()}")
    print(f"  5yr declines (>3 points): {polity['polity_decline_5yr'].sum()}")

    try:
        from stage5_ews.estimate import run_ews
        ews = run_ews()
    except Exception:
        ews = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "stage5_ews", "ews_signals.csv"))

    if "combined_risk" not in ews.columns:
        print("  No combined_risk in EWS data")
        return

    merged = ews[["country_text_id", "year", "combined_risk"]].merge(
        polity[["country_text_id", "year", "polity_decline_3yr", "polity_decline_5yr"]],
        on=["country_text_id", "year"], how="inner"
    )
    merged = merged.dropna()
    print(f"  Merged: {len(merged)} country-years")

    from sklearn.metrics import roc_auc_score

    for horizon, col in [("3yr", "polity_decline_3yr"), ("5yr", "polity_decline_5yr")]:
        valid = merged[merged[col].notna() & merged["combined_risk"].notna()]
        if valid[col].sum() > 0 and valid[col].nunique() > 1:
            auc = roc_auc_score(valid[col], valid["combined_risk"])
            n_pos = valid[col].sum()
            print(f"\n  Polity {horizon} decline (>3pt drop):")
            print(f"    AUC-ROC: {auc:.3f} (n_positive={n_pos})")
            print(f"    {'VALIDATES' if auc > 0.65 else 'WEAK' if auc > 0.55 else 'FAILS'}: AIM4D risk predicts non-V-Dem outcome")

    from scipy.stats import spearmanr
    for horizon in ["3yr", "5yr"]:
        col = f"polity_change_{horizon}"
        if col in merged.columns:
            valid = merged[[col, "combined_risk"]].dropna()
            if len(valid) > 10:
                rho, p = spearmanr(valid["combined_risk"], -valid[col])
                print(f"\n  Spearman correlation (risk vs Polity {horizon} decline magnitude):")
                print(f"    rho = {rho:.3f}, p = {p:.4f}")

    results = {"metric": [], "value": []}
    for horizon, col in [("3yr", "polity_decline_3yr"), ("5yr", "polity_decline_5yr")]:
        valid = merged[merged[col].notna() & merged["combined_risk"].notna()]
        if valid[col].sum() > 0 and valid[col].nunique() > 1:
            auc = roc_auc_score(valid[col], valid["combined_risk"])
            results["metric"].append(f"auc_polity_{horizon}")
            results["value"].append(auc)

    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "polity_validation.csv"), index=False)
    print(f"\n  Saved to robustness/polity_validation.csv")


if __name__ == "__main__":
    run_polity_validation()
