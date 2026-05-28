#!/usr/bin/env bash
# Drive every experiment needed for the reviewer revision, sequentially on one GPU.
# All outputs land in results/ with descriptive suffixes; logs in logs/.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
mkdir -p logs

run() {
  echo "=================================================================="
  echo ">>> $* :: $(date)"
  echo "=================================================================="
  python -m src.experiments.run_real_experiments "$@"
}

# 1) Main runs: both models, all experiments (fresh JSON with new baselines).
run --model both --experiment all 2>&1 | tee logs/main_all.log

# 2) Llama multi-step finer-layer-bin remedy (6 role bins instead of 3).
run --model llama --experiment multistep --n-layer-role 6 --suffix _finebins 2>&1 | tee logs/llama_finebins.log

# 3) Discretisation sensitivity (+-20%) on the headline IOI task (Gemma).
run --model gemma --experiment ioi --kl-threshold-scale 0.8 --suffix _sens_kl0.8 2>&1 | tee logs/sens_kl0.8.log
run --model gemma --experiment ioi --kl-threshold-scale 1.2 --suffix _sens_kl1.2 2>&1 | tee logs/sens_kl1.2.log
run --model gemma --experiment ioi --act-threshold-scale 0.8 --suffix _sens_act0.8 2>&1 | tee logs/sens_act0.8.log
run --model gemma --experiment ioi --act-threshold-scale 1.2 --suffix _sens_act1.2 2>&1 | tee logs/sens_act1.2.log

echo "ALL REVISION EXPERIMENTS COMPLETE :: $(date)"
