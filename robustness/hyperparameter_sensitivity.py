"""
Hyperparameter sensitivity sweep (one-at-a-time).

For each of:
  - POSTONSET_EXCL_YEARS (default 5; try 3, 7)
  - BASELINE_END (default 2005; try 2003, 2007)
  - LEAD_YEARS (default 5; try 4, 6)
  - WATCH_PCTL (default 0.80; try 0.75, 0.85)
  - WARNING_PCTL (default 0.95; try 0.93, 0.97)
  - ALERT_PCTL (default 0.98; try 0.97, 0.99)

we rerun Stage 5 holding all other hyperparameters at default and report
OOS AUC, OOS AUC-PR, LOEO sensitivity, and detection counts.

A robust model should show <0.02 AUC swing across ±1-2 step changes in each
parameter. Larger swings indicate the headline result depends on the specific
hyperparameter choice — a red flag.

Output: robustness/hyperparameter_sensitivity.csv with one row per sweep point.
Run time: ~5 min per config × 18 configs = ~90 min on Brev.
"""

import os
import subprocess
import re
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hyperparameter_sensitivity.csv")

SWEEPS = [
    ("postonset_excl_years", "AIM4D_POSTONSET", 5, [3, 7]),
    ("baseline_end",         "AIM4D_BASELINE_END", 2005, [2003, 2007]),
    ("lead_years",           "AIM4D_LEAD_YEARS", 5, [4, 6]),
    ("watch_pctl",           "AIM4D_WATCH_PCTL", 0.80, [0.75, 0.85]),
    ("warning_pctl",         "AIM4D_WARNING_PCTL", 0.95, [0.93, 0.97]),
    ("alert_pctl",           "AIM4D_ALERT_PCTL", 0.98, [0.97, 0.99]),
]


def run_stage5(env_overrides):
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})
    print(f"\n{'='*70}\nRunning Stage 5 with: {env_overrides}\n{'='*70}")
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
                        (r"Alert \(P98\)", "loeo_alert"),
                        (r"Watch \(top 20%\)", "in_sample_watch"),
                        (r"Warning \(top 5%\)", "in_sample_warning"),
                        (r"Alert \(top 2%\)", "in_sample_alert")]:
        m = re.search(tier + r":\s+(\d+)/(\d+)", log)
        if m: out[label] = f"{m.group(1)}/{m.group(2)}"
    return out


def main():
    rows = []

    print(f"\n*** BASELINE (all defaults) ***")
    log = run_stage5({})
    m = parse_metrics(log)
    m["sweep_name"] = "baseline"
    m["param_value"] = "default"
    rows.append(m)

    for name, env_var, default, alts in SWEEPS:
        for val in alts:
            print(f"\n*** {name} = {val} (default {default}) ***")
            log = run_stage5({env_var: val})
            m = parse_metrics(log)
            m["sweep_name"] = name
            m["param_value"] = val
            rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("\n" + "=" * 78)
    print("HYPERPARAMETER SENSITIVITY")
    print("=" * 78)
    cols = ["sweep_name", "param_value", "oos_auc", "oos_auc_pr",
            "loeo_watch", "loeo_warning", "loeo_alert"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
