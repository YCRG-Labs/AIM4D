#!/usr/bin/env bash
# Full AIM4D pipeline + robustness suite, macOS-friendly.
# Canonical stages run sequentially; Task E folds and Task F episodes run
# concurrently in isolated git worktrees (AIM4D_PAR overrides worker count,
# AIM4D_THREADS caps per-process threads).
#
# Usage:
#   bash run_all_local.sh
#   AIM4D_PAR=3 AIM4D_THREADS=3 bash run_all_local.sh
set -euo pipefail
cd "$(dirname "$0")"

LOG="run_all_local.log"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

export AIM4D_THREADS="${AIM4D_THREADS:-4}"
PY="${AIM4D_PYTHON:-python3.11}"

CPUS=$(sysctl -n hw.ncpu 2>/dev/null || nproc)
RAM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
echo "================================================================"
echo "AIM4D full local run started $(date -u +%FT%TZ)"
echo "Host: $(hostname)  CPUs: ${CPUS}  RAM: ${RAM_GB}Gi  Python: $($PY --version 2>&1)"
echo "AIM4D_THREADS=${AIM4D_THREADS}  AIM4D_PAR=${AIM4D_PAR:-auto}"
echo "================================================================"

for f in data/vdem_v16.csv data/contiguity/DirectContiguity320 data/atop \
         data/macro_covariates.csv data/gdelt_country_year.csv; do
  if [[ ! -e $f ]]; then
    echo "[ERROR] Missing $f — stage prerequisites must exist before a concurrent run." >&2
    exit 1
  fi
done

if [[ -n "$(git status --porcelain | grep '\.py$' || true)" ]]; then
  echo "[WARN] Uncommitted .py changes exist; Task E/F worktrees run HEAD only:"
  git status --porcelain | grep '\.py$' || true
fi

echo
echo "================================================================"
echo "CANONICAL PIPELINE (cutoff=2019)"
echo "================================================================"
$PY -u stage1_factors/extract.py
$PY -u stage2_betas/estimate.py
$PY -u stage3_msvar/estimate.py
$PY -u stage4_nscm/estimate.py
$PY -u stage5_ews/estimate.py

echo
echo "================================================================"
echo "FAST ROBUSTNESS"
echo "================================================================"
$PY -u robustness/dsp_ablation.py
$PY -u robustness/bootstrap_cis.py
$PY -u robustness/contagion_seed_sweep.py
$PY -u robustness/network_seed_sweep.py
$PY -u robustness/contagion_galton.py
$PY -u robustness/elastic_net_robustness.py || echo "  (elastic_net_robustness failed — non-blocking)"

if [[ "${FAST_ONLY:-0}" == "1" ]]; then
  echo
  echo "FAST_ONLY=1 set — skipping Task E and Task F"
  echo "Total wall-time: $(( SECONDS / 60 )) min"
  exit 0
fi

echo
echo "================================================================"
echo "TASK E: expanding-window CV (4 folds, parallel worktrees)"
echo "================================================================"
$PY -u robustness/expanding_window_cv.py

echo
echo "================================================================"
echo "TASK F: full-pipeline LOEO (15 episodes, parallel worktrees)"
echo "================================================================"
$PY -u robustness/sample_pipeline_loeo.py

echo
echo "================================================================"
echo "DONE. Total wall-time: $(( SECONDS / 60 )) min"
echo "Log: $LOG"
echo "================================================================"
