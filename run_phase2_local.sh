#!/usr/bin/env bash
# Phase 2: regenerate every paper-relevant robustness artifact against the
# fresh canonical pipeline outputs, cheap scripts first, heavy refit scripts
# last. prospective_drivers.py is deliberately excluded: Table 9 is the
# registered 2026-2031 forecast and its frozen CSV must not be regenerated.
set -uo pipefail
cd "$(dirname "$0")"

LOG="run_phase2_local.log"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

export AIM4D_THREADS="${AIM4D_THREADS:-4}"
PY="${AIM4D_PYTHON:-python3.11}"

echo "Phase 2 started $(date -u +%FT%TZ)"
FAILED=()

run() {
  echo
  echo "================================================================"
  echo "ROBUSTNESS: $1  ($(date -u +%H:%M:%SZ))"
  echo "================================================================"
  if ! $PY -u "robustness/$1"; then
    echo "  [FAIL] $1"
    FAILED+=("$1")
  fi
}

run permutation_importance_oos.py
run rashomon_importance.py
run lead_time_auc.py
run detection_lead_times.py
run temporal_holdout.py
run novel_precision.py
run benchmark_finishers.py
run benchmark_stress_test.py
run polity_validation.py
run external_benchmarks.py
run false_positive_analysis.py
run threshold_sweep.py
run mobilization_precedence.py
run dsp_imputation_robustness.py
run sanity_checks.py
run factor_robustness.py
run csd_hardening.py
run csd_levels_fix.py
run contagion_placebo.py
run contagion_fix.py
run contagion_dyadic.py
run contagion_blocsplit.py
run causal_real_data.py
run representation_value.py
run stage_ablation.py
run network_variants.py
run hmm_states.py
run k_sensitivity.py
run vdem_uncertainty.py

echo
echo "================================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "PHASE 2 DONE, all scripts succeeded. Total: $(( SECONDS / 60 )) min"
else
  echo "PHASE 2 DONE with ${#FAILED[@]} failures: ${FAILED[*]}"
  echo "Total: $(( SECONDS / 60 )) min"
fi
echo "================================================================"
