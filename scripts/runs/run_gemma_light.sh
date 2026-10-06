#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
TASK=${1:-}
[[ -n $TASK ]] || { printf 'Usage: bash scripts/runs/run_gemma_light.sh TASK [--dry-run]\n'; exit 2; }
shift
case "$TASK" in
    sokoban|countdown|frozen_lake|webshop|lean|sudoku) LAUNCHER="run_${TASK}_light.sh" ;;
    sokoban_ablations|countdown_ablations) LAUNCHER="${TASK}.sh" ;;
    sokoban_maxrl|sokoban_tstar) LAUNCHER="run_${TASK}.sh" ;;
    *) printf 'Unknown Gemma task: %s\n' "$TASK" >&2; exit 2 ;;
esac
[[ $# == 0 || ( $# == 1 && $1 == --dry-run ) ]] || { printf 'Only --dry-run is supported\n' >&2; exit 2; }
export MODEL=${MODEL:-$ROOT/.cache/gemma/gemma-4-E4B-it}
export GPU=${GPU:-0}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export SEED=${SEED:-10000}
export EVAL_SEED=${EVAL_SEED:-123}
export EVAL_ATTEMPTS=${EVAL_ATTEMPTS:-1}
export EXPERIMENT=${EXPERIMENT:-${TASK}_gemma_pass1_seed${SEED}}
DEFAULT_STEPS=200
DEFAULT_TEST_FREQ=25
[[ $TASK != webshop ]] || DEFAULT_STEPS=100
[[ $TASK != sudoku ]] || DEFAULT_TEST_FREQ=10
export TEST_FREQ=${TEST_FREQ:-$DEFAULT_TEST_FREQ}
export SAVE_FREQ=${SAVE_FREQ:-100}
export STEPS=${STEPS:-$DEFAULT_STEPS}
export PYTHON=${PYTHON:-python}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$ROOT/.cache/gemma/vllm}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT/.cache/gemma/torchinductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$ROOT/.cache/gemma/triton}
case "$TASK" in
    sokoban|countdown|frozen_lake|webshop) export RUNS=${RUNS:-EVAL,PPO,GRPO,SNR,DAPO,TROPIC} ;;
    lean|sudoku) export RUNS=${RUNS:-EVAL,PPO,GRPO,SNR,TROPIC} ;;
esac
case "$TASK" in
    countdown|countdown_ablations) export PROTOCOL=${PROTOCOL:-original} DATA=${DATA:-tropic} ;;
esac
if [[ ${1:-} != --dry-run ]]; then
    if [[ $TASK == webshop ]]; then
        source "$ROOT/scripts/runs/gemma_webshop_runtime.sh"
        configure_gemma_webshop_runtime
    fi
    "$PYTHON" -m ragen.gemma.preflight --model "$MODEL"
fi
exec bash "$ROOT/scripts/runs/$LAUNCHER" "$@"
