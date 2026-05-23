"""
Robustness check: DSP imputation strategy.

The Stage 5 default restricts the panel to year >= 2000 (DSP coverage window
per Mechkova et al. DSP-WP1) and country-forward-fills within. A reviewer may
ask: have you tested alternative imputation that doesn't drop 26% of rows?

We rerun Stage 5 under two configurations:
  - ffill_2000 (default)
  - median_full (keep all years, train-period country-median fill)

Output: robustness/dsp_imputation_robustness.csv with OOS metrics for each.
"""

import os
import subprocess
import sys
import re
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsp_imputation_robustness.csv")


def run_stage5(strategy):
    env = os.environ.copy()
    env["AIM4D_DSP_STRATEGY"] = strategy
    print(f"\n{'='*70}\nRunning Stage 5 with AIM4D_DSP_STRATEGY={strategy}\n{'='*70}")
    proc = subprocess.run(
        ["python3", "-u", os.path.join(REPO, "stage5_ews", "estimate.py")],
        env=env, cwd=REPO, capture_output=True, text=True,
    )
    return proc.stdout + "\n" + proc.stderr


def parse_metrics(log):
    out = {}
    m = re.search(r"AUC-ROC \(OOS\):\s+([\d.]+)", log)
    if m: out["oos_auc"] = float(m.group(1))
    m = re.search(r"AUC-PR \(OOS\):\s+([\d.]+)", log)
    if m: out["oos_auc_pr"] = float(m.group(1))
    m = re.search(r"Base rate.*?AUC-ROC:\s+([\d.]+)", log, re.DOTALL)
    if m: out["in_sample_auc"] = float(m.group(1))
    m = re.search(r"Base rate.*?AUC-PR:\s+([\d.]+)", log, re.DOTALL)
    if m: out["in_sample_auc_pr"] = float(m.group(1))
    for tier, label in [(r"Watch \(P80\)", "loeo_watch"),
                        (r"Warning \(P95\)", "loeo_warning"),
                        (r"Alert \(P98\)", "loeo_alert")]:
        m = re.search(tier + r":\s+(\d+)/(\d+)", log)
        if m:
            out[label] = f"{m.group(1)}/{m.group(2)}"
    m = re.search(r"Restricted to year >= 2000.*->\s+(\d+) country-years", log)
    if m: out["n_country_years_after_dsp_window"] = int(m.group(1))
    m = re.search(r"median_full.*n=(\d+)", log)
    if m: out["n_country_years_after_dsp_window"] = int(m.group(1))
    return out


def main():
    rows = []
    for strategy in ["ffill_2000", "median_full"]:
        log = run_stage5(strategy)
        m = parse_metrics(log)
        m["strategy"] = strategy
        rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("\n" + "=" * 70)
    print("DSP IMPUTATION ROBUSTNESS")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
