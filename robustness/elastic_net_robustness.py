"""
Robustness check: feature selection (elastic-net) vs all features.

The Stage 5 default uses all features. Reviewers may ask: does the model still
work with explicit feature selection? We rerun Stage 5 twice — once with all
features, once with elastic-net pruning — and report OOS AUC, AUC-PR, and LOEO
sensitivity for each.

Run AFTER an initial Stage 5 pass. Re-runs Stage 5 once with AIM4D_USE_ENET=1.

Output: robustness/elastic_net_robustness.csv comparing the two configurations.
"""

import os
import subprocess
import sys
import shutil
import re
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EWS = os.path.join(REPO, "stage5_ews", "ews_signals.csv")
LOEO = os.path.join(REPO, "stage5_ews", "loeo_results.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elastic_net_robustness.csv")


def run_stage5(env_overrides, label):
    env = os.environ.copy()
    env.update(env_overrides)
    print(f"\n{'='*70}\nRunning Stage 5 with config: {label}\nenv: {env_overrides}\n{'='*70}")
    proc = subprocess.run(
        ["python3", "-u", os.path.join(REPO, "stage5_ews", "estimate.py")],
        env=env, cwd=REPO, capture_output=True, text=True,
    )
    return proc.stdout + "\n" + proc.stderr


def parse_metrics(log):
    """Extract OOS AUC/AUC-PR, LOEO sensitivity by tier, and feature count from a Stage 5 log."""
    out = {}
    m = re.search(r"AUC-ROC \(OOS\):\s+([\d.]+)", log)
    if m:
        out["oos_auc"] = float(m.group(1))
    m = re.search(r"AUC-PR \(OOS\):\s+([\d.]+)", log)
    if m:
        out["oos_auc_pr"] = float(m.group(1))
    m = re.search(r"Base rate.*?AUC-ROC:\s+([\d.]+)", log, re.DOTALL)
    if m:
        out["in_sample_auc"] = float(m.group(1))
    m = re.search(r"Base rate.*?AUC-PR:\s+([\d.]+)", log, re.DOTALL)
    if m:
        out["in_sample_auc_pr"] = float(m.group(1))
    for tier, label in [("Watch \\(P80\\)", "loeo_watch"),
                        ("Warning \\(P95\\)", "loeo_warning"),
                        ("Alert \\(P98\\)", "loeo_alert")]:
        m = re.search(tier + r":\s+(\d+)/(\d+)", log)
        if m:
            out[label] = f"{m.group(1)}/{m.group(2)}"
            out[label + "_frac"] = int(m.group(1)) / int(m.group(2))
    m = re.search(r"Using all (\d+) features", log)
    if m:
        out["n_features_used"] = int(m.group(1))
    m = re.search(r"pruning to (\d+)/(\d+) features", log)
    if m:
        out["n_features_used"] = int(m.group(1))
    return out


def main():
    rows = []

    log = run_stage5({}, "baseline_all_features")
    m = parse_metrics(log)
    m["config"] = "all_features"
    rows.append(m)

    log = run_stage5({"AIM4D_USE_ENET": "1"}, "elastic_net_pruned")
    m = parse_metrics(log)
    m["config"] = "elastic_net_pruned"
    rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("\n" + "=" * 70)
    print("ELASTIC-NET FEATURE SELECTION ROBUSTNESS")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
