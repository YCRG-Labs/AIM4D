"""
Alternate-label robustness: evaluate the model's risk scores against
non-V-Dem definitions of democratic decline.

V-Dem is the field-standard but reviewers will ask: does the model predict
*democratic decline* or just *V-Dem's coding choices*? We evaluate the same
calibrated_risk scores against:

  - Freedom House (e_fh_pr): 3yr / 5yr declines at 2pt and 3pt thresholds
  - Polity-VI (e_polity2): 3yr / 5yr declines at 2pt and 3pt thresholds

Output: robustness/alternate_labels.csv — one row per (source, window, threshold).

The script reuses the bootstrap_auc helper from bootstrap_cis.py to put
country-cluster 95% CIs on every comparison.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_cis import bootstrap_auc

EWS_PATH = os.path.join(REPO, "stage5_ews", "ews_signals.csv")
VDEM_PATH = os.path.join(REPO, "data", "vdem_v16.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alternate_labels.csv")


def build_decline_labels(vdem_path, col, source_name, windows=(3, 5), thresholds=(2, 3)):
    """Build decline labels at multiple window/threshold combos."""
    df = pd.read_csv(vdem_path, usecols=["country_text_id", "year", col], low_memory=False).dropna()
    df[col] = df[col].astype(float)
    df = df.sort_values(["country_text_id", "year"]).reset_index(drop=True)
    out = {}
    for w in windows:
        diff = df.groupby("country_text_id")[col].diff(w)
        for t in thresholds:
            if source_name == "FH":
                lbl = (diff >= t).astype(int)
            else:
                lbl = (diff <= -t).astype(int)
            key = f"{source_name}_{w}yr_{t}pt"
            tmp = df[["country_text_id", "year"]].copy()
            tmp[key] = lbl
            out[key] = tmp
    return out


def main():
    if not os.path.exists(EWS_PATH):
        sys.exit(f"Missing {EWS_PATH} — run stage5_ews/estimate.py first")
    if not os.path.exists(VDEM_PATH):
        sys.exit(f"Missing {VDEM_PATH}")

    ews = pd.read_csv(EWS_PATH)
    if "calibrated_risk" not in ews.columns:
        sys.exit("ews_signals.csv missing calibrated_risk column — rerun Stage 5.")

    valid = ews.dropna(subset=["calibrated_risk", "country_text_id", "year"]).copy()
    if "is_postonset" in valid.columns:
        valid = valid[~valid["is_postonset"].fillna(False)]

    rows = []

    print("Building Freedom House decline labels...")
    fh_labels = build_decline_labels(VDEM_PATH, "e_fh_pr", "FH")
    print("Building Polity-VI decline labels...")
    polity_labels = build_decline_labels(VDEM_PATH, "e_polity2", "Polity")

    print("\n" + "=" * 78)
    print("ALTERNATE-LABEL EVALUATION (model risk vs non-V-Dem decline definitions)")
    print("=" * 78)
    print(f"{'Label spec':<28s}  {'n':>6s}  {'n+':>4s}  {'AUC':>6s}  {'95% CI':>20s}  {'AUC-PR':>7s}")

    for source, label_dict in [("FH", fh_labels), ("Polity", polity_labels)]:
        for key, df in label_dict.items():
            merged = valid.merge(df, on=["country_text_id", "year"], how="inner").dropna(subset=[key])
            if len(merged) < 50 or merged[key].sum() < 2:
                continue
            y = merged[key].astype(int).values
            s = merged["calibrated_risk"].values
            clusters = merged["country_text_id"].values
            try:
                auc = roc_auc_score(y, s)
                ap = average_precision_score(y, s)
                auc_b, lo, hi = bootstrap_auc(y, s, fn=roc_auc_score, n_boot=1000, clusters=clusters)
                ap_b, lop, hip = bootstrap_auc(y, s, fn=average_precision_score, n_boot=1000, clusters=clusters)
            except ValueError:
                continue
            row = {
                "label_source": source,
                "label_spec": key,
                "n": len(merged),
                "n_positive": int(y.sum()),
                "base_rate": float(y.mean()),
                "auc_roc": auc, "auc_roc_ci_low": lo, "auc_roc_ci_high": hi,
                "auc_pr": ap, "auc_pr_ci_low": lop, "auc_pr_ci_high": hip,
            }
            rows.append(row)
            print(f"{key:<28s}  {len(merged):>6d}  {int(y.sum()):>4d}  "
                  f"{auc:>6.3f}  [{lo:.3f}, {hi:.3f}]  {ap:>7.3f}")

    if "label" in valid.columns:
        y = valid["label"].astype(int).values
        s = valid["calibrated_risk"].values
        clusters = valid["country_text_id"].values
        if y.sum() >= 2:
            auc = roc_auc_score(y, s)
            ap = average_precision_score(y, s)
            auc_b, lo, hi = bootstrap_auc(y, s, fn=roc_auc_score, n_boot=1000, clusters=clusters)
            ap_b, lop, hip = bootstrap_auc(y, s, fn=average_precision_score, n_boot=1000, clusters=clusters)
            rows.append({
                "label_source": "V-Dem",
                "label_spec": "V-Dem_ERT_5yr_pre-onset",
                "n": len(valid), "n_positive": int(y.sum()), "base_rate": float(y.mean()),
                "auc_roc": auc, "auc_roc_ci_low": lo, "auc_roc_ci_high": hi,
                "auc_pr": ap, "auc_pr_ci_low": lop, "auc_pr_ci_high": hip,
            })
            print(f"{'V-Dem ERT (train labels)':<28s}  {len(valid):>6d}  {int(y.sum()):>4d}  "
                  f"{auc:>6.3f}  [{lo:.3f}, {hi:.3f}]  {ap:>7.3f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
