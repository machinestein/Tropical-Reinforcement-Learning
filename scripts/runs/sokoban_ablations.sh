#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
GPU=${GPU:-0}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
STEPS=${STEPS:-200}
TEST_FREQ=${TEST_FREQ:-25}
SAVE_FREQ=${SAVE_FREQ:-100}
EVAL_ATTEMPTS=${EVAL_ATTEMPTS:-8}
SEED=${SEED:-10000}
EXPERIMENT=${EXPERIMENT:-sokoban_ablations_seed${SEED}}
SAVE_DIR=${SAVE_DIR:-$ROOT/saves}
PYTHON=${PYTHON:-python}
WANDB_PROJECT=${WANDB_PROJECT:-RAGEN2}
export WANDB_MODE=${WANDB_MODE:-disabled}

case "${1:-}" in
    -h|--help)
        printf '%s\n' \
            'Usage: bash scripts/runs/sokoban_ablations.sh [--dry-run]' \
            'Exactly three sequential TROPIC runs: no composition, successful fragments only, root-only sampling.' \
            'Each starts from the initial model with fresh memory; use the full TROPIC light run as the control.' \
            'Defaults match run_sokoban_light.sh: Qwen2.5-3B-Instruct, GPU 0, microbatch 1, 200 steps.' \
            'Validation: 512 boards from seed 123, 8 sampled attempts, temperature 0.5, every 25 steps.' \
            'Checkpoints every 100 steps; W&B project RAGEN2. Online tracking requires WANDB_MODE=online and wandb login.' \
            'Settings: MODEL, GPU, MICRO_BATCH_SIZE, STEPS, TEST_FREQ, SAVE_FREQ, EVAL_ATTEMPTS,' \
            'SEED, EXPERIMENT, SAVE_DIR, PYTHON, WANDB_PROJECT; optional WANDB_ENTITY/WANDB_MODE.' \
            'Existing folders are never overwritten. See README.md for semantics and diagnostics.'
        exit 0 ;;
    --dry-run) [[ $# == 1 ]] || { printf 'Only --dry-run is supported.\n' >&2; exit 2; } ;;
    '') [[ $# == 0 ]] || exit 2 ;;
    *) printf 'Unknown argument: %s (see --help)\n' "$1" >&2; exit 2 ;;
esac

PREFIX="RAGEN2_${MODEL##*/}_${EXPERIMENT}"
printf 'Sokoban ablations: three runs, W&B project %s\n' "$WANDB_PROJECT"
printf 'Run/folder prefix: %s\n' "$PREFIX"

exec bash "$ROOT/scripts/runs/run_sokoban_tropic_suite.sh" \
    --runs tropic_no_composition,tropic_successful_fragments_only,tropic_no_frontier \
    --model "$MODEL" --gpus "$GPU" --seeds "$SEED" \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --steps "$STEPS" --test-freq "$TEST_FREQ" --save-freq "$SAVE_FREQ" \
    --eval-mode config --eval-attempts "$EVAL_ATTEMPTS" \
    --output-dir "$SAVE_DIR" --name-prefix "$PREFIX" \
    --project "$WANDB_PROJECT" --wandb --python "$PYTHON" "$@"
