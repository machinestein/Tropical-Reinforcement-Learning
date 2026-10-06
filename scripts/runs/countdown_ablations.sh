#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

export SEED=${SEED:-10000}
export EXPERIMENT=${EXPERIMENT:-countdown_ablations_seed${SEED}}
export PROTOCOL=${PROTOCOL:-original}
export DATA=${DATA:-tropic}
export STEPS=${STEPS:-200}
export TEST_FREQ=${TEST_FREQ:-25}
export SAVE_FREQ=${SAVE_FREQ:-100}
export EVAL_PROBLEMS=${EVAL_PROBLEMS:-512}
export EVAL_ATTEMPTS=${EVAL_ATTEMPTS:-8}
export RUNS=${RUNS:-TROPIC,TROPIC_NO_COMPOSITION,TROPIC_SUCCESSFUL_FRAGMENTS_ONLY,TROPIC_NO_FRONTIER}

case "${1:-}" in
    -h|--help)
        printf '%s\n' \
            'Usage: bash scripts/runs/countdown_ablations.sh [--dry-run]' \
            'Default: four sequential jobs, each with fresh graph memory and the same initial model:' \
            '  TROPIC, TROPIC_NO_COMPOSITION, TROPIC_SUCCESSFUL_FRAGMENTS_ONLY, TROPIC_NO_FRONTIER.' \
            'RUNS selects a comma-separated subset in that order (e.g. RUNS=TROPIC_NO_FRONTIER).' \
            'Defaults: Qwen2.5-3B-Instruct, GPU 0, microbatch 1, 200 steps, training seed 10000.' \
            'PROTOCOL=original DATA=tropic matches the completed Countdown TROPIC run:' \
            '  one operation per turn, no warmup, disjoint generated n=3..6 data, 128-problem training pool.' \
            'Validation: 512 problems from seed 123, 8 sampled attempts, temperature 0.5, every 25 steps.' \
            'Checkpoints every 100 steps; W&B project RAGEN_COUNTDOWN; outputs under saves/.' \
            'Optional PROTOCOL=atomic shares one warmup across all selected jobs; it is a separate comparison.' \
            'Settings inherited from run_countdown_light.sh: MODEL, GPU, MICRO_BATCH_SIZE, STEPS, TEST_FREQ,' \
            'SAVE_FREQ, EVAL_PROBLEMS, EVAL_ATTEMPTS, EVAL_SEED, SEED, TRAIN_POOL_SIZE, EXPERIMENT, SAVE_DIR,' \
            'PYTHON, WANDB_PROJECT, PROTOCOL, WARMUP_MODEL, WARMUP_STEPS, WARMUP_MICRO_BATCH_SIZE, RUNS.' \
            'This ablation wrapper requires DATA=tropic. Existing run folders are never overwritten.' \
            'A failed job stops the suite. To retry it, choose RUNS and a new EXPERIMENT.' \
            'Example: GPU=0,1,2,3,4,5,6,7 PYTHON=venv/bin/python bash scripts/runs/countdown_ablations.sh'
        exit 0 ;;
    --dry-run) [[ $# == 1 ]] || { printf 'Only --dry-run is supported.\n' >&2; exit 2; } ;;
    '') [[ $# == 0 ]] || exit 2 ;;
    *) printf 'Unknown argument: %s (see --help)\n' "$1" >&2; exit 2 ;;
esac

[[ $DATA == tropic ]] || { printf 'Error: Countdown ablations require disjoint DATA=tropic splits.\n' >&2; exit 2; }
[[ $RUNS =~ ^[A-Z_]+(,[A-Z_]+)*$ ]] || { printf 'Error: RUNS must be a comma list of TROPIC variants.\n' >&2; exit 2; }
IFS=, read -r -a REQUESTED <<< "$RUNS"
for method in "${REQUESTED[@]}"; do
    case "$method" in
        TROPIC|TROPIC_NO_COMPOSITION|TROPIC_SUCCESSFUL_FRAGMENTS_ONLY|TROPIC_NO_FRONTIER) ;;
        *) printf 'Error: Unknown Countdown ablation: %s\n' "$method" >&2; exit 2 ;;
    esac
done

exec bash "$ROOT/scripts/runs/run_countdown_light.sh" "$@"
