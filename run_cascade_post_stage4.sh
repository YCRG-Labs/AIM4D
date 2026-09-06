#!/usr/bin/env bash
# Rerun every script that backs a manuscript number and depends on
# stage5_ews/ews_signals.csv, after the Stage 4 fixes (causal temporal edges,
# EXCLUDE_COUNTRY honoured, econ-similarity on raw GDP) and the Stage 5 RNG seed.
#
# prospective_drivers.py is deliberately absent: its output is the frozen
# registered forecast and must not be regenerated.
#
# Tiers are ordered cheap -> expensive so headline numbers land first. Tier 3
# scripts each already use multiple workers, so they run one at a time.
set -u
cd "$(dirname "$0")"
PY="${AIM4D_PYTHON:-/Users/jacobcrainic/AIM4D/.venv_ablation/bin/python}"
LOG=logs/cascade_$(date +%Y%m%d_%H%M); mkdir -p "$LOG"
run() { local s="$1"; shift
  echo "[$(date +%H:%M:%S)] start $s" | tee -a "$LOG/_index.txt"
  ( cd robustness && "$@" $PY -u "$s" > "../$LOG/${s%.py}.log" 2>&1 ); rc=$?
  echo "[$(date +%H:%M:%S)] done  $s rc=$rc" | tee -a "$LOG/_index.txt"; }

echo "=== TIER 1: cheap, sequential ===" | tee -a "$LOG/_index.txt"
for s in netdep_total_variation.py strict_table_final.py strict_ert_sensitivity.py bootstrap_cis.py \
         baseline_common_rows.py reliability_bins.py channel_ablation.py dsp_ablation.py \
         mobilization_only_baseline.py mobilization_precedence.py lead_time_auc.py stage_ablation.py \
         permutation_importance_oos.py rashomon_importance.py watchlist_metrics.py alert_burden_calibration.py; do
  [ -f "robustness/$s" ] && run "$s" || echo "skip missing $s" | tee -a "$LOG/_index.txt"
done

echo "=== TIER 2: medium, two at a time ===" | tee -a "$LOG/_index.txt"
run design_factorial.py & run onset_forecast_clean.py & wait
run design_factorial_ert.py & run onset_forecast_ert.py & wait
AIM4D_PLACEBO_PERM=50 run closure_placebo.py & run vdem_uncertainty.py & wait

echo "=== TIER 3: long, sequential (each is internally parallel) ===" | tee -a "$LOG/_index.txt"
AIM4D_ORIGINS="2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020" run strict_endtoend_refit.py
AIM4D_ORIGINS="2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020" run strict_endtoend_refit_ert.py
run strict_endtoend_ert_comparison.py
run expanding_window_cv.py
run contagion_seed_sweep.py
run network_seed_sweep.py
run netdep_tv_seed_sweep.py
run sample_pipeline_loeo.py

echo "=== CASCADE COMPLETE ===" | tee -a "$LOG/_index.txt"
grep -c "rc=0" "$LOG/_index.txt" | xargs -I{} echo "{} scripts exited 0" | tee -a "$LOG/_index.txt"
grep "rc=[1-9]" "$LOG/_index.txt" | tee -a "$LOG/_index.txt" || true
