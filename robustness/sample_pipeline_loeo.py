"""
Task F: 15-episode full-pipeline leave-one-out validation.

For each sample episode (varied across era and type), refit the entire
pipeline with that country's data EXCLUDED from the training subsets at
Stages 1, 3, and 5. The country still appears in the panel for prediction
(loadings/HMM/meta-learner are applied to it after training without it).

The cheap LOEO in Stage 5 only refits the meta-learner; this script bounds
the upstream contamination by running full-pipeline LOEOs and comparing.

Episodes run concurrently, each in its own detached git worktree (data/
symlinked from the canonical checkout), so refits never touch the canonical
stage outputs and no restore pass is needed. Concurrency defaults to a
RAM/CPU-derived worker count; override with AIM4D_PAR.

Outputs:
  robustness/sample_pipeline_loeo.csv  — per-episode max risk, detection tier,
    compared against the meta-only LOEO recorded in stage5_ews/loeo_results.csv
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

from _refit_worktree import (REPO, add_worktree, default_workers, refit_env,
                             remove_worktree, run_stages, warn_if_dirty)

OUT = os.path.dirname(os.path.abspath(__file__))

SAMPLE_EPISODES = [
    ("Venezuela", 2002, "backsliding"),
    ("Bangladesh", 2007, "backsliding"),
    ("Fiji", 2006, "coup"),
    ("Niger", 2009, "coup"),
    ("Hungary", 2010, "backsliding"),
    ("Türkiye", 2013, "backsliding"),
    ("Poland", 2015, "backsliding"),
    ("Thailand", 2014, "coup"),
    ("Egypt", 2013, "coup"),
    ("Tunisia", 2021, "backsliding"),
    ("Nigeria", 2021, "backsliding"),
    ("Brazil", 2019, "backsliding"),
    ("Burma/Myanmar", 2021, "coup"),
    ("Mali", 2020, "coup"),
    ("Sudan", 2021, "coup"),
]

_smoke_limit = int(os.environ.get("AIM4D_SMOKE_LIMIT", "0"))
if _smoke_limit > 0:
    SAMPLE_EPISODES = SAMPLE_EPISODES[:_smoke_limit]
    print(f"[SMOKE] limited to {len(SAMPLE_EPISODES)} episodes")

LEAD = 5


def run_episode(item):
    country, onset, ep_type = item
    print(f"Full-pipeline LOEO {country} ({onset}, {ep_type}): starting refit "
          f"in worktree", flush=True)
    worktree = add_worktree(f"loeo_{country}")
    try:
        env = refit_env(AIM4D_CUTOFF=2019, AIM4D_EXCLUDE_COUNTRY=country)
        rc = run_stages(worktree, env, label=f" episode={country}")
        if rc != 0:
            return {"country": country, "onset": onset, "type": ep_type,
                    "error": f"pipeline rc={rc}"}
        max_risk, best_tier, all_tiers = collect_predictions(country, onset,
                                                             repo=worktree)
        return {"country": country, "onset": onset, "type": ep_type,
                "max_risk": max_risk, "best_tier": best_tier,
                "all_tiers": all_tiers}
    finally:
        remove_worktree(worktree)


def collect_predictions(country, onset, repo=REPO):
    """Read ews_signals.csv and return the country's pre-onset risk and tier."""
    ews = pd.read_csv(os.path.join(repo, "stage5_ews/ews_signals.csv"))
    pre = ews[(ews["country_name"] == country)
              & (ews["year"] >= onset - LEAD)
              & (ews["year"] < onset)]
    if len(pre) == 0:
        return None, None, None
    pre = pre.dropna(subset=["combined_risk"])
    if len(pre) == 0:
        return None, None, None
    max_risk = float(pre["combined_risk"].max())
    tiers = pre["alert_tier"].value_counts().to_dict()
    best_tier = "none"
    for t in ["alert", "warning", "watch"]:
        if t in tiers:
            best_tier = t
            break
    return max_risk, best_tier, dict(tiers)


def load_meta_only_loeo():
    """Read Stage 5's meta-only LOEO results for comparison."""
    path = os.path.join(REPO, "stage5_ews/loeo_results.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {r["country"]: (float(r["max_risk"]), r["tier"])
            for _, r in df.iterrows()}


def main():
    try:
        import hmmlearn
    except ImportError:
        sys.exit(
            "ERROR: hmmlearn not installed (required by stage3_msvar). Install with:\n"
            "  pip install --user hmmlearn\n"
            "or:  pip install --user -r requirements.txt"
        )
    meta_only = load_meta_only_loeo()
    warn_if_dirty()

    workers = min(default_workers(), len(SAMPLE_EPISODES))
    print(f"Running {len(SAMPLE_EPISODES)} episodes across {workers} concurrent "
          f"worktrees (AIM4D_THREADS={os.environ.get('AIM4D_THREADS', '4')})",
          flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_episode, SAMPLE_EPISODES))

    rows = []
    for res in results:
        if "error" in res:
            rows.append({"country": res["country"], "onset": res["onset"],
                         "type": res["type"], "error": res["error"]})
            print(f"  {res['country']}: FAILED ({res['error']})")
            continue

        country, onset = res["country"], res["onset"]
        max_risk, best_tier, all_tiers = res["max_risk"], res["best_tier"], res["all_tiers"]
        meta_risk, meta_tier = meta_only.get(country, (np.nan, "n/a"))

        delta = max_risk - meta_risk if (max_risk is not None and not np.isnan(meta_risk)) else np.nan

        row = {
            "country": country, "onset": onset, "type": res["type"],
            "full_pipeline_max_risk": max_risk,
            "full_pipeline_tier": best_tier,
            "meta_only_max_risk": meta_risk,
            "meta_only_tier": meta_tier,
            "delta_risk": delta,
            "tier_breakdown": str(all_tiers),
        }
        rows.append(row)
        if max_risk is None:
            print(f"  {country}: no scorable pre-onset rows", flush=True)
            continue
        print(f"  {country}: full-pipeline max_risk={max_risk:.4f} tier={best_tier}", flush=True)
        print(f"     meta-only      max_risk={meta_risk:.4f} tier={meta_tier}")
        print(f"     delta (full - meta) = {delta:+.4f}")

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT, "sample_pipeline_loeo.csv")
    df.to_csv(out_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"Summary")
    print(f"{'=' * 70}")
    print(df.to_string(index=False))
    valid = df[df["delta_risk"].notna()] if "delta_risk" in df.columns else df.iloc[0:0]
    if len(valid):
        print(f"\nMean (full - meta) risk delta: {valid['delta_risk'].mean():+.4f}")
        print(f"Std:                           {valid['delta_risk'].std():.4f}")
        print(f"Max abs:                       {valid['delta_risk'].abs().max():.4f}")
        print(f"\nInterpretation: small delta means the meta-only LOEO in the paper")
        print(f"is a good proxy for full-pipeline LOEO. Large delta means upstream")
        print(f"contamination matters and meta-only LOEO is optimistic.")
    print(f"\nWrote {out_path}")
    print("Refits ran in isolated worktrees; canonical stage outputs untouched, "
          "no restore pass needed.")


if __name__ == "__main__":
    main()
