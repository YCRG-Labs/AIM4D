"""
Robustness check: feature selection (elastic-net) vs all features.

The Stage 5 default uses all features. Reviewers may ask: does the model still
work with explicit feature selection? The all-features row is scored from the
canonical Stage 5 outputs; the pruned configuration reruns Stage 5 with
AIM4D_USE_ENET=1 inside an isolated git worktree so the canonical outputs are
never touched. Metrics are computed directly from each run's artifacts on the
strict 2019 protocol rather than parsed from logs. Stage 5 also persists the
elastic-net diagnostic coefficients to stage5_ews/enet_coefficients.csv.

Output: robustness/elastic_net_robustness.csv comparing the two configurations.
"""

import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _refit_worktree import REPO, add_worktree, refit_env, remove_worktree
from expanding_window_cv import evaluate_fold

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "elastic_net_robustness.csv")


def metrics_for(repo, config):
    r = evaluate_fold(2019, 2026, repo=repo)
    loeo = pd.read_csv(os.path.join(repo, "stage5_ews", "loeo_results.csv"))
    row = {
        "config": config,
        "oos_auc": r["auc"],
        "oos_auc_pr": r["ap"],
        "n_pos": r["n_pos"],
        "loeo_watch": int(loeo["tier"].isin(["watch", "warning", "alert"]).sum()),
        "loeo_total": len(loeo),
    }
    coef_path = os.path.join(repo, "stage5_ews", "enet_coefficients.csv")
    if os.path.exists(coef_path):
        coef = pd.read_csv(coef_path)
        row["n_features_total"] = len(coef)
        row["n_features_enet_selected"] = int((coef["coefficient"].abs() > 1e-4).sum())
    return row


def main():
    rows = [metrics_for(REPO, "all_features")]

    worktree = add_worktree("enet_robustness")
    try:
        env = refit_env(AIM4D_CUTOFF=2019, AIM4D_EXCLUDE_COUNTRY=None,
                        AIM4D_USE_ENET=1)
        rc = subprocess.call(
            [sys.executable, os.path.join(worktree, "stage5_ews", "estimate.py")],
            env=env, cwd=worktree)
        if rc != 0:
            raise SystemExit(f"Stage 5 ENET run failed with rc={rc}")
        rows.append(metrics_for(worktree, "elastic_net_pruned"))

        coef_src = os.path.join(worktree, "stage5_ews", "enet_coefficients.csv")
        coef_dst = os.path.join(REPO, "stage5_ews", "enet_coefficients.csv")
        if os.path.exists(coef_src) and not os.path.exists(coef_dst):
            pd.read_csv(coef_src).to_csv(coef_dst, index=False)
    finally:
        remove_worktree(worktree)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("\n" + "=" * 70)
    print("ELASTIC-NET FEATURE SELECTION ROBUSTNESS")
    print("=" * 70)
    print(df.to_string(index=False))
    delta = rows[1]["oos_auc"] - rows[0]["oos_auc"]
    print(f"\nPruning changes strict-2019 OOS AUC by {delta:+.3f} "
          f"relative to all features")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
