"""
Lead-time decomposition of OOS forecasting performance.

For each OOS episode (onset > TRAIN_CUTOFF), we have pre-onset country-years at
leads of 1, 2, 3, 4, 5 years before onset. Bucket the OOS panel by lead-to-onset
and compute AUC and AUC-PR within each bucket against the same non-episode
negatives.

Answers the reviewer question: "At a 1-year lead does the model do 0.96 AUC and
degrade to 0.85 by year 5, or is it the same across the horizon?" The expected
shape is graceful degradation, which itself is a paper finding.

Output: robustness/lead_time_auc.csv with one row per lead-bucket.

Run AFTER Stage 5 (depends on stage5_ews/ews_signals.csv with calibrated_risk
and is_postonset columns).
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from stage5_ews.estimate import KNOWN_EPISODES, TRAIN_CUTOFF, lead_for

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lead_time_auc.csv")
EWS_PATH = os.path.join(REPO, "stage5_ews", "ews_signals.csv")


def cluster_bootstrap_ci(y, s, clusters, fn, n_boot=1000, rng=None):
    rng = rng or np.random.default_rng(42)
    unique = np.unique(clusters)
    by_c = {c: np.where(clusters == c)[0] for c in unique}
    out = []
    for _ in range(n_boot):
        draw = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_c[c] for c in draw])
        yy, ss = y[idx], s[idx]
        if yy.sum() < 2 or yy.sum() == len(yy):
            continue
        out.append(fn(yy, ss))
    if not out:
        return np.nan, np.nan
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    if not os.path.exists(EWS_PATH):
        sys.exit(f"Missing {EWS_PATH} — run stage5_ews/estimate.py first")

    ews = pd.read_csv(EWS_PATH)
    required = {"country_name", "year", "calibrated_risk"}
    missing = required - set(ews.columns)
    if missing:
        sys.exit(f"ews_signals.csv missing columns: {missing}. Rerun stage5_ews/estimate.py.")

    ews["years_to_onset"] = np.nan
    ews["is_episode_pre"] = False
    for country, info in KNOWN_EPISODES.items():
        onset = info["onset"]
        lead = lead_for(info)
        mask = (ews["country_name"] == country) & \
               (ews["year"] >= onset - lead) & \
               (ews["year"] < onset)
        ews.loc[mask, "years_to_onset"] = onset - ews.loc[mask, "year"]
        ews.loc[mask, "is_episode_pre"] = True

    oos = ews[ews["year"] > TRAIN_CUTOFF].copy()

    if "is_postonset" in oos.columns:
        oos = oos[~oos["is_postonset"].fillna(False)]

    print(f"OOS panel: n={len(oos)} country-years, "
          f"{oos['is_episode_pre'].sum()} episode-pre rows, "
          f"{oos['country_name'].nunique()} countries")

    rows = []

    y_all = oos["is_episode_pre"].astype(int).values
    s_all = oos["calibrated_risk"].values
    if y_all.sum() >= 2:
        auc = roc_auc_score(y_all, s_all)
        ap = average_precision_score(y_all, s_all)
        lo, hi = cluster_bootstrap_ci(y_all, s_all, oos["country_name"].values, roc_auc_score)
        lop, hip = cluster_bootstrap_ci(y_all, s_all, oos["country_name"].values, average_precision_score)
        rows.append({
            "lead_years": "1-5_pooled",
            "n_total": len(oos),
            "n_positive": int(y_all.sum()),
            "auc_roc": auc, "auc_roc_ci_low": lo, "auc_roc_ci_high": hi,
            "auc_pr": ap, "auc_pr_ci_low": lop, "auc_pr_ci_high": hip,
        })

    negatives = oos[~oos["is_episode_pre"]]
    for lead in [1, 2, 3, 4, 5]:
        positives = oos[oos["years_to_onset"] == lead]
        if len(positives) < 2:
            continue
        bucket = pd.concat([positives, negatives], ignore_index=True)
        y = bucket["is_episode_pre"].astype(int).values
        s = bucket["calibrated_risk"].values
        if y.sum() < 2 or y.sum() == len(y):
            continue
        auc = roc_auc_score(y, s)
        ap = average_precision_score(y, s)
        lo, hi = cluster_bootstrap_ci(y, s, bucket["country_name"].values, roc_auc_score)
        lop, hip = cluster_bootstrap_ci(y, s, bucket["country_name"].values, average_precision_score)
        rows.append({
            "lead_years": lead,
            "n_total": len(bucket),
            "n_positive": int(y.sum()),
            "auc_roc": auc, "auc_roc_ci_low": lo, "auc_roc_ci_high": hi,
            "auc_pr": ap, "auc_pr_ci_low": lop, "auc_pr_ci_high": hip,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("\n" + "=" * 78)
    print("OOS LEAD-TIME AUC DECOMPOSITION (year > %d, post-onset excluded)" % TRAIN_CUTOFF)
    print("=" * 78)
    print(f"{'Lead':>10s}  {'n':>5s}  {'n+':>4s}  {'AUC':>6s}  {'AUC 95% CI':>20s}  "
          f"{'AUC-PR':>7s}  {'AUC-PR 95% CI':>20s}")
    for _, r in df.iterrows():
        print(f"{str(r['lead_years']):>10s}  {r['n_total']:>5d}  {r['n_positive']:>4d}  "
              f"{r['auc_roc']:>6.3f}  [{r['auc_roc_ci_low']:.3f}, {r['auc_roc_ci_high']:.3f}]  "
              f"{r['auc_pr']:>7.3f}  [{r['auc_pr_ci_low']:.3f}, {r['auc_pr_ci_high']:.3f}]")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
