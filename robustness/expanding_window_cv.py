"""
Task E: real expanding-window cross-validation with full-pipeline refit.

For each cutoff year in [2008, 2011, 2014, 2017], refit all five stages
(POET factors, Kalman/DCC betas, MS-VAR HMM, INE-TARNet, EWS meta-learner) on
data <= cutoff, then evaluate OOS AUC and AUC-PR on the next 3-year window.

This is the only honest version of expanding-window CV for the AIM4D pipeline.
The version inside Stage 5's run loop only slices a single in-sample model's
predictions and does not refit.

Folds run concurrently, each in its own detached git worktree (data/
symlinked from the canonical checkout), so refits never touch the canonical
stage outputs. Concurrency defaults to a RAM/CPU-derived worker count;
override with AIM4D_PAR. Heaviest step per fold is the Stage 3 HMM.

Outputs:
  robustness/expanding_window_cv.csv  — per-fold AUC, AUC-PR, n_pos, episodes
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from _refit_worktree import (REPO, add_worktree, default_workers, refit_env,
                             remove_worktree, run_stages, warn_if_dirty)

OUT = os.path.dirname(os.path.abspath(__file__))

CUTOFFS = [2008, 2011, 2014, 2017]
LEAD = 5


def run_fold(cutoff):
    test_end = cutoff + 3
    print(f"Fold train<={cutoff}: starting refit in worktree", flush=True)
    worktree = add_worktree(f"ewcv_{cutoff}")
    try:
        env = refit_env(AIM4D_CUTOFF=cutoff, AIM4D_EXCLUDE_COUNTRY=None)
        rc = run_stages(worktree, env, label=f" fold={cutoff}")
        if rc != 0:
            return {"cutoff": cutoff, "test_end": test_end,
                    "auc": np.nan, "ap": np.nan, "error": f"pipeline rc={rc}"}
        return evaluate_fold(cutoff, test_end, repo=worktree)
    finally:
        remove_worktree(worktree)


def evaluate_fold(cutoff, test_window_end, repo=REPO):
    """Read ews_signals.csv, score OOS on (cutoff, test_window_end]."""
    sys.path.insert(0, REPO)
    from stage5_ews.estimate import KNOWN_EPISODES
    preonset, postonset = set(), set()
    for c, info in KNOWN_EPISODES.items():
        o = info["onset"]
        for y in range(o - LEAD, o + 1):
            preonset.add((c, y))
        for y in range(o + 1, o + 6):
            postonset.add((c, y))

    ews = pd.read_csv(os.path.join(repo, "stage5_ews/ews_signals.csv"))
    ews["lbl"] = ews.apply(lambda r: 1 if (r["country_name"], r["year"]) in preonset else 0, axis=1)
    ews["pos"] = ews.apply(lambda r: (r["country_name"], r["year"]) in postonset, axis=1)

    oos = ews[(ews["year"] > cutoff) & (ews["year"] <= test_window_end)
              & (~ews["pos"]) & ews["combined_risk"].notna()].copy()
    if oos["lbl"].sum() < 2:
        return {"cutoff": cutoff, "test_end": test_window_end,
                "auc": np.nan, "ap": np.nan, "n_pos": int(oos["lbl"].sum()),
                "n": len(oos), "episodes_in_window": 0}

    auc = roc_auc_score(oos["lbl"], oos["combined_risk"])
    ap = average_precision_score(oos["lbl"], oos["combined_risk"])
    eps_in_window = sum(1 for c, info in KNOWN_EPISODES.items()
                        if cutoff < info["onset"] <= test_window_end)
    return {"cutoff": cutoff, "test_end": test_window_end,
            "auc": float(auc), "ap": float(ap),
            "n_pos": int(oos["lbl"].sum()), "n": len(oos),
            "episodes_in_window": eps_in_window}


def main():
    warn_if_dirty()
    workers = min(default_workers(), len(CUTOFFS))
    print(f"Running {len(CUTOFFS)} folds across {workers} concurrent worktrees "
          f"(AIM4D_THREADS={os.environ.get('AIM4D_THREADS', '4')})", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run_fold, CUTOFFS))

    for r in rows:
        if "error" in r:
            print(f"  fold {r['cutoff']}: FAILED ({r['error']})")
        else:
            print(f"  fold {r['cutoff']}: AUC={r['auc']:.3f}, AP={r['ap']:.3f}, "
                  f"n_pos={r['n_pos']}, episodes={r['episodes_in_window']}", flush=True)

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT, "expanding_window_cv.csv")
    df.to_csv(out_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"Summary")
    print(f"{'=' * 70}")
    print(df.to_string(index=False))
    valid = df[df["auc"].notna()]
    if len(valid):
        print(f"\nMean AUC across folds:    {valid['auc'].mean():.3f} +/- {valid['auc'].std():.3f}")
        print(f"Mean AUC-PR across folds: {valid['ap'].mean():.3f} +/- {valid['ap'].std():.3f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
