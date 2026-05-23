"""
Tuning sweep: test each Stage 5 tuning knob in isolation.

Five configurations, each one re-runs Stage 5 only (upstream unchanged)
and captures OOS AUC, OOS AUC-PR, LOEO sensitivity, and DSP ablation Δ.

  baseline       — no improvements (defaults: coup-lead=5, pos-weight=1, smooth=1)
  coup_lead_3    — coup pre-onset window = 3 yrs (backsliding still 5)
  pos_weight_3   — positive labels weighted 3x in GB training
  smooth_3       — rolling 3-year max of risk score before tier assignment
  all_three      — all three together (the original failing combo)

Each config:
  1. Sets env vars (AIM4D_COUP_LEAD / AIM4D_POS_WEIGHT / AIM4D_SMOOTH)
  2. Runs stage5_ews/estimate.py
  3. Parses key metrics from stdout
  4. Runs robustness/dsp_ablation.py with the resulting ews_signals.csv
  5. Records a row in robustness/tuning_sweep.csv

Total runtime: ~25-35 min (5 configs x ~5 min/config).
"""

import os
import re
import subprocess
import sys
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__))


CONFIGS = [
    ("baseline",     {}),
    ("coup_lead_3",  {"AIM4D_COUP_LEAD": "3"}),
    ("pos_weight_3", {"AIM4D_POS_WEIGHT": "3.0"}),
    ("smooth_3",     {"AIM4D_SMOOTH": "3"}),
    ("all_three",    {"AIM4D_COUP_LEAD": "3", "AIM4D_POS_WEIGHT": "3.0", "AIM4D_SMOOTH": "3"}),
]


PATTERNS = {
    "auc_in":       r"AUC-ROC:\s+([0-9.]+)\s*$",
    "auc_pr_in":    r"AUC-PR:\s+([0-9.]+)\s*$",
    "auc_oos":      r"AUC-ROC \(OOS\):\s+([0-9.]+)",
    "auc_pr_oos":   r"AUC-PR \(OOS\):\s+([0-9.]+)",
    "watch_total":  r"Watch \(top 20%\):\s+(\d+)/(\d+)",
    "warning_total":r"Warning \(top 5%\):\s+(\d+)/(\d+)",
    "alert_total":  r"Alert \(top 2%\):\s+(\d+)/(\d+)",
    "loeo_watch":   r"Watch \(P80\):\s+(\d+)/(\d+)",
    "loeo_coup":    r"coup:\s+(\d+)/(\d+)",
    "loeo_backsl":  r"backsliding:\s+(\d+)/(\d+)",
    "oos_detect":   r"Episodes \(onset>\d+\):\s+\d+,\s+detected:\s+(\d+)/(\d+)",
    "cv_mean":      r"Mean AUC across windows:\s+([0-9.]+)\s+\+/-",
}


def parse_stage5(log):
    out = {}
    for key, pat in PATTERNS.items():
        m = re.search(pat, log, re.MULTILINE)
        if not m:
            continue
        if m.lastindex and m.lastindex >= 2:
            a, b = m.group(1), m.group(2)
            out[key] = f"{a}/{b}"
            out[key + "_frac"] = float(a) / float(b) if float(b) else 0.0
        else:
            out[key] = float(m.group(1))
    return out


def parse_dsp(log):
    out = {}
    m = re.search(r"Δ AUC-ROC \(ablate − full\):\s+([+\-0-9.]+)", log)
    if m:
        out["dsp_delta_auc"] = float(m.group(1))
    m = re.search(r"Δ OOS AUC \(ablate − full\):\s+([+\-0-9.]+)", log)
    if m:
        out["dsp_delta_oos_auc"] = float(m.group(1))
    m = re.search(r"Verdict:\s+(.+)", log)
    if m:
        out["dsp_verdict"] = m.group(1).strip()
    return out


def run_config(name, env_overrides):
    print(f"\n{'='*70}\n  CONFIG: {name}    env: {env_overrides}\n{'='*70}", flush=True)

    env = os.environ.copy()
    for k in ["AIM4D_COUP_LEAD", "AIM4D_POS_WEIGHT", "AIM4D_SMOOTH"]:
        env.pop(k, None)
    env.update(env_overrides)

    s5 = subprocess.run(["python3", "-u", os.path.join(REPO, "stage5_ews/estimate.py")],
                        env=env, cwd=REPO, capture_output=True, text=True)
    if s5.returncode != 0:
        print(f"  STAGE 5 FAILED ({s5.returncode}):\n{s5.stderr[-500:]}")
        return {"config": name, "error": "stage5"}

    s5_metrics = parse_stage5(s5.stdout)

    dsp = subprocess.run(["python3", "-u", os.path.join(REPO, "robustness/dsp_ablation.py")],
                         env=env, cwd=REPO, capture_output=True, text=True)
    dsp_metrics = parse_dsp(dsp.stdout) if dsp.returncode == 0 else {}

    row = {"config": name, **env_overrides, **s5_metrics, **dsp_metrics}
    print(f"  -> AUC OOS = {row.get('auc_oos','?')}  "
          f"AUC-PR OOS = {row.get('auc_pr_oos','?')}  "
          f"LOEO = {row.get('loeo_watch','?')}  "
          f"DSP Δ OOS = {row.get('dsp_delta_oos_auc','?')}", flush=True)
    return row


def main():
    rows = []
    for name, env in CONFIGS:
        r = run_config(name, env)
        rows.append(r)

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT, "tuning_sweep.csv")
    df.to_csv(out_path, index=False)

    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    keep = ["config", "auc_in", "auc_pr_in", "auc_oos", "auc_pr_oos",
            "watch_total", "warning_total", "alert_total",
            "loeo_watch", "loeo_backsl", "loeo_coup",
            "oos_detect", "cv_mean", "dsp_delta_auc", "dsp_delta_oos_auc", "dsp_verdict"]
    keep = [c for c in keep if c in df.columns]
    print(df[keep].to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
