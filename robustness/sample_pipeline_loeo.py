"""
Task F: 5-episode full-pipeline leave-one-out validation.

For each of 5 sample episodes (varied across era and type), refit the entire
pipeline with that country's data EXCLUDED from the training subsets at
Stages 1, 3, and 5. The country still appears in the panel for prediction
(loadings/HMM/meta-learner are applied to it after training without it).

The cheap LOEO in Stage 5 only refits the meta-learner; this script bounds
the upstream contamination by running ~5 full-pipeline LOEOs and comparing.

Run time: ~30-45 min per episode * 5 episodes = 2.5-4 hr total.
Heaviest step is the Stage 3 HMM (60 random restarts) which gets rerun each
episode.

Outputs:
  robustness/sample_pipeline_loeo.csv  — per-episode max risk, detection tier,
    compared against the meta-only LOEO recorded in stage5_ews/loeo_results.csv
"""

import os
import subprocess
import sys
import pandas as pd
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def run_full_pipeline(exclude_country):
    """Refit stages 1-5 with one country excluded from training subsets."""
    env = os.environ.copy()
    env["AIM4D_EXCLUDE_COUNTRY"] = exclude_country
    env["AIM4D_CUTOFF"] = "2019"
    for stage in ["stage1_factors/extract.py", "stage2_betas/estimate.py",
                  "stage3_msvar/estimate.py", "stage4_nscm/estimate.py",
                  "stage5_ews/estimate.py"]:
        path = os.path.join(REPO, stage)
        rc = subprocess.call(["python3", path], env=env, cwd=REPO)
        if rc != 0:
            print(f"  [FAIL] {stage} returned {rc}", flush=True)
            return rc
    return 0


def collect_predictions(country, onset):
    """Read ews_signals.csv and return the country's pre-onset risk and tier."""
    ews = pd.read_csv(os.path.join(REPO, "stage5_ews/ews_signals.csv"))
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
    rows = []

    for country, onset, ep_type in SAMPLE_EPISODES:
        print(f"\n{'=' * 70}")
        print(f"Full-pipeline LOEO: {country} ({onset}, {ep_type})")
        print(f"{'=' * 70}", flush=True)

        rc = run_full_pipeline(country)
        if rc != 0:
            rows.append({"country": country, "onset": onset, "type": ep_type,
                         "error": f"pipeline rc={rc}"})
            continue

        max_risk, best_tier, all_tiers = collect_predictions(country, onset)
        meta_risk, meta_tier = meta_only.get(country, (np.nan, "n/a"))

        delta = max_risk - meta_risk if (max_risk is not None and not np.isnan(meta_risk)) else np.nan

        row = {
            "country": country, "onset": onset, "type": ep_type,
            "full_pipeline_max_risk": max_risk,
            "full_pipeline_tier": best_tier,
            "meta_only_max_risk": meta_risk,
            "meta_only_tier": meta_tier,
            "delta_risk": delta,
            "tier_breakdown": str(all_tiers),
        }
        rows.append(row)
        print(f"  -> full-pipeline LOEO max_risk={max_risk:.4f} tier={best_tier}", flush=True)
        print(f"     meta-only      LOEO max_risk={meta_risk:.4f} tier={meta_tier}")
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

    print(f"\nRestoring canonical pipeline state (no country excluded)...")
    env = os.environ.copy()
    env.pop("AIM4D_EXCLUDE_COUNTRY", None)
    env["AIM4D_CUTOFF"] = "2019"
    for stage in ["stage1_factors/extract.py", "stage2_betas/estimate.py",
                  "stage3_msvar/estimate.py", "stage4_nscm/estimate.py",
                  "stage5_ews/estimate.py"]:
        subprocess.call(["python3", os.path.join(REPO, stage)], env=env, cwd=REPO)


if __name__ == "__main__":
    main()
