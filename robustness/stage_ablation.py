"""Leave-one-stage-out ablation on the headline out-of-sample metric.

The reviewer asks whether removing a pipeline stage actually costs out-of-sample
discrimination, so that the five-stage architecture is costed against the degrees
of freedom it introduces. We answer this directly for Stage 4 (the Network SCM /
contagion module), which is the gateway through which Stages 2 (time-varying
betas) and 3 (regime-state HMM) reach the Stage-5 forecaster: their only path to
the final risk score is the Stage-4 contagion feature, so dropping Stage 4 costs
the entire mechanism block at once. Stage 1 (factor extraction) underlies every
downstream feature and is not separately removable; Stage 5 is the forecaster
itself.

For each variant we re-fit the Stage-5 meta-ensemble and read the strict OOS
(year > cutoff) AUC-PR and AUC-ROC. We repeat across random-seed offsets so the
full-minus-ablated gap can be judged against seed-to-seed noise (the standard
ablation-reporting convention: report the delta with its seed spread). Runs use
AIM4D_NO_WRITE so the registered ews_signals.csv is never overwritten.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE5 = os.path.join(REPO, "stage5_ews", "estimate.py")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage_ablation_results.csv")

VARIANTS = [
    ("full", {}),
    ("drop_stage4", {"AIM4D_ABLATE_GROUP": "stage4"}),
]
SEED_OFFSETS = [int(s) for s in os.environ.get("AIM4D_ABLATE_OFFSETS", "0,100,200").split(",")]


def run_one(env_extra):
    env = dict(os.environ)
    env["AIM4D_NO_WRITE"] = "1"
    env["AIM4D_SKIP_LOEO"] = "1"
    env.update(env_extra)
    proc = subprocess.run([sys.executable, STAGE5], env=env, cwd=REPO,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-3000:] + "\n" + proc.stderr[-3000:])
        raise RuntimeError(f"stage5 failed for {env_extra}")
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("ABLATE_JSON "):
            payload = json.loads(line[len("ABLATE_JSON "):])
    if payload is None:
        raise RuntimeError(f"no ABLATE_JSON emitted for {env_extra}")
    return payload


def main():
    rows = []
    for label, env_extra in VARIANTS:
        for off in SEED_OFFSETS:
            e = dict(env_extra)
            e["AIM4D_SEED_OFFSET"] = str(off)
            print(f"running variant={label} seed_offset={off} ...", flush=True)
            r = run_one(e)
            r["variant"] = label
            rows.append(r)
            print(f"  -> OOS AUC-PR={r['oos_pr']:.3f}  AUC-ROC={r['oos_roc']:.3f}  "
                  f"n_features={r['n_features']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    full = df[df["variant"] == "full"]
    base_rate = full["oos_base_rate"].mean()

    print("\n" + "=" * 70)
    print(f"LEAVE-ONE-STAGE-OUT ABLATION (strict OOS, base rate {base_rate:.3f})")
    print("=" * 70)
    print(f"{'variant':<14}{'AUC-PR':>16}{'AUC-ROC':>16}{'dPR':>10}{'dROC':>10}")
    pr_full = full["oos_pr"].mean()
    roc_full = full["oos_roc"].mean()
    for label, _ in VARIANTS:
        g = df[df["variant"] == label]
        pr_m, pr_s = g["oos_pr"].mean(), g["oos_pr"].std(ddof=0)
        roc_m, roc_s = g["oos_roc"].mean(), g["oos_roc"].std(ddof=0)
        dpr = pr_m - pr_full
        droc = roc_m - roc_full
        print(f"{label:<14}{pr_m:>8.3f}+/-{pr_s:<5.3f}{roc_m:>8.3f}+/-{roc_s:<5.3f}"
              f"{dpr:>+10.3f}{droc:>+10.3f}")
    print(f"\nSaved to {OUT}")
    print("Interpretation: a |delta| smaller than the seed spread means the stage "
          "adds no out-of-sample discrimination over the rest of the pipeline.")


if __name__ == "__main__":
    main()
