"""
Master script: runs all robustness checks and produces a consolidated summary.

Usage:
  python robustness/run_all.py              # Run all checks
  python robustness/run_all.py --fast       # Skip slow checks (K sensitivity, network variants)
  python robustness/run_all.py --only X     # Run only check X

Individual checks can also be run standalone:
  python robustness/threshold_sweep.py
  python robustness/false_positive_analysis.py
  python robustness/baseline_comparison.py
  python robustness/k_sensitivity.py        # ~60 min (re-runs full pipeline 3x)
  python robustness/hmm_states.py           # ~40 min (re-runs HMM 4x)
  python robustness/network_variants.py     # ~20 min (re-trains GNN 4x)

Estimated total runtime: ~2-3 hours on a modern machine.
Fast mode (--fast): ~15 min (threshold + FP + baseline only).
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKS = {
    "threshold": {
        "module": "threshold_sweep",
        "function": "run_threshold_sweep",
        "description": "Threshold sensitivity + Thailand analysis",
        "fast": True,
        "est_time": "2-3 min",
    },
    "false_positive": {
        "module": "false_positive_analysis",
        "function": "run_false_positive_analysis",
        "description": "False positive classification",
        "fast": True,
        "est_time": "1-2 min",
    },
    "baseline": {
        "module": "baseline_comparison",
        "function": "run_baseline_comparison",
        "description": "Baseline comparison + staged ablation",
        "fast": True,
        "est_time": "5-10 min",
    },
    "hmm_states": {
        "module": "hmm_states",
        "function": "run_hmm_states",
        "description": "HMM state count sensitivity (S=3,4,5,6)",
        "fast": False,
        "est_time": "30-40 min",
    },
    "k_sensitivity": {
        "module": "k_sensitivity",
        "function": "run_k_sensitivity",
        "description": "Factor count sensitivity (K=3,4,5)",
        "fast": False,
        "est_time": "45-60 min",
    },
    "network": {
        "module": "network_variants",
        "function": "run_network_variants",
        "description": "Network definition robustness",
        "fast": False,
        "est_time": "15-20 min",
    },
}


def run_check(name, config):
    """Import and run a single robustness check."""
    print(f"\n{'#' * 70}")
    print(f"# RUNNING: {name} — {config['description']}")
    print(f"# Estimated time: {config['est_time']}")
    print(f"{'#' * 70}\n")

    start = time.time()

    try:
        mod = __import__(config["module"])
        func = getattr(mod, config["function"])
        result = func()
        elapsed = time.time() - start
        print(f"\n  Completed {name} in {elapsed:.0f}s")
        return {"check": name, "status": "SUCCESS", "time_s": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  FAILED {name} after {elapsed:.0f}s: {e}")
        import traceback
        traceback.print_exc()
        return {"check": name, "status": f"FAILED: {e}", "time_s": elapsed}


def run_all(fast=False, only=None):
    print("=" * 70)
    print("AIM4D ROBUSTNESS CHECK SUITE")
    print("=" * 70)
    print(f"\nMode: {'FAST (threshold + FP + baseline)' if fast else 'FULL (all checks)'}")
    if only:
        print(f"Running only: {only}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    results = []
    total_start = time.time()

    for name, config in CHECKS.items():
        if only and name != only:
            continue
        if fast and not config["fast"]:
            print(f"  Skipping {name} (slow, use --full to include)")
            continue

        result = run_check(name, config)
        results.append(result)

    total_time = time.time() - total_start

    print(f"\n{'=' * 70}")
    print("ROBUSTNESS SUITE SUMMARY")
    print(f"{'=' * 70}")
    print(f"\nTotal time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"\n{'Check':<20} {'Status':<15} {'Time':>8}")
    print(f"{'-'*45}")
    for r in results:
        status = r["status"][:12]
        print(f"  {r['check']:<18} {status:<13} {r['time_s']:>6.0f}s")

    n_success = sum(1 for r in results if r["status"] == "SUCCESS")
    n_total = len(results)
    print(f"\n  {n_success}/{n_total} checks passed")

    print(f"\nOutput files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".csv"):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
            print(f"  {f} ({size / 1024:.1f} KB)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIM4D Robustness Check Suite")
    parser.add_argument("--fast", action="store_true",
                       help="Skip slow checks (K sensitivity, HMM states, network variants)")
    parser.add_argument("--only", type=str, choices=list(CHECKS.keys()),
                       help="Run only a specific check")
    args = parser.parse_args()

    run_all(fast=args.fast, only=args.only)
