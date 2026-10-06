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
EXPERIMENT=${EXPERIMENT:-sokoban_seed${SEED}}
SAVE_DIR=${SAVE_DIR:-$ROOT/saves}
PYTHON=${PYTHON:-python}
WANDB_PROJECT=${WANDB_PROJECT:-RAGEN2}
export WANDB_MODE=${WANDB_MODE:-disabled}
RUNS=${RUNS:-EVAL,PPO,GRPO,SNR,TROPIC}

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }

case "${1:-}" in
    -h|--help)
        printf '%s\n' \
            "Usage: bash scripts/runs/run_sokoban_light.sh [--dry-run]" \
            "Default runs: initial EVAL, PPO, GRPO, SNR (PPO + RAGEN-2 filtering), TROPIC." \
            "RUNS=DAPO selects only RAGEN-2 DAPO; comma lists such as RUNS=PPO,DAPO select multiple methods." \
            "DAPO: PPO/GAE, clip 0.2/0.28, no KL, token-mean, no SNR filtering." \
            "EVAL/PPO/GRPO/SNR/DAPO use _2_sokoban; only TROPIC uses _2_sokoban_tropic." \
            "Defaults: Qwen2.5-3B-Instruct, GPU 0, 200 iterations, seed 10000." \
            "Original microbatch: 1 sequence per GPU; global PPO minibatch: 32." \
            "Original sampled validation (temperature 0.5) on 512 boards with 8 attempts each," \
            "logging pass@1, pass@2, pass@4 and pass@8, every 25 steps; checkpoints every 100." \
            "Results: saves/RAGEN2_Qwen2.5-3B-Instruct_sokoban_seed10000_{method}/" \
            "W&B: project RAGEN2, with the same run names as the folders." \
            "Online tracking is opt-in: WANDB_MODE=online, wandb login, and optional WANDB_ENTITY." \
            "Settings: RUNS, MODEL, GPU, MICRO_BATCH_SIZE, STEPS, TEST_FREQ, SAVE_FREQ, EVAL_ATTEMPTS, SEED, EXPERIMENT, SAVE_DIR, PYTHON, WANDB_PROJECT." \
            "Example: GPU=1 EXPERIMENT=sokoban_v2 bash scripts/runs/run_sokoban_light.sh" \
            "Existing run folders are never overwritten; use a new EXPERIMENT for a new run."
        exit 0 ;;
    --dry-run) [[ $# == 1 ]] || { printf 'Only --dry-run is supported.\n' >&2; exit 2; } ;;
    '') [[ $# == 0 ]] || exit 2 ;;
    *) printf 'Unknown argument: %s (see --help)\n' "$1" >&2; exit 2 ;;
esac

ALL_METHODS=(EVAL PPO GRPO SNR DAPO TROPIC)
[[ $RUNS =~ ^[A-Z]+(,[A-Z]+)*$ ]] || die "RUNS must be a comma list of ${ALL_METHODS[*]} (got '$RUNS')"
IFS=, read -r -a REQUESTED <<< "$RUNS"
declare -A wanted=()
for method in "${REQUESTED[@]}"; do
    [[ " ${ALL_METHODS[*]} " == *" $method "* ]] || die "RUNS must be a comma list of ${ALL_METHODS[*]} (got '$method')"
    [[ ! -v wanted[$method] ]] || die "RUNS lists $method twice"
    wanted[$method]=1
done
METHODS=()
SUITE_RUNS=()
for method in "${ALL_METHODS[@]}"; do
    [[ -v wanted[$method] ]] || continue
    METHODS+=("$method")
    case "$method" in
        EVAL) SUITE_RUNS+=(base_original) ;;
        PPO) SUITE_RUNS+=(original_ppo) ;;
        GRPO) SUITE_RUNS+=(original_grpo) ;;
        SNR) SUITE_RUNS+=(original_ppo_snr) ;;
        DAPO) SUITE_RUNS+=(original_dapo) ;;
        TROPIC) SUITE_RUNS+=(tropic) ;;
    esac
done

PREFIX="RAGEN2_${MODEL##*/}_${EXPERIMENT}"
printf 'Light Sokoban comparison: %s runs, W&B project %s\n' "${#METHODS[@]}" "$WANDB_PROJECT"
printf 'Run/folder names: %s_{%s}\n' "$PREFIX" "$(IFS=,; printf '%s' "${METHODS[*]}")"
printf 'Baselines: original RAGEN-2 full-history protocol. TROPIC: separate state-only protocol.\n'

exec bash "$ROOT/scripts/runs/run_sokoban_tropic_suite.sh" \
    --runs "$(IFS=,; printf '%s' "${SUITE_RUNS[*]}")" \
    --model "$MODEL" --gpus "$GPU" --seeds "$SEED" \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --steps "$STEPS" --test-freq "$TEST_FREQ" --save-freq "$SAVE_FREQ" \
    --eval-mode config --eval-attempts "$EVAL_ATTEMPTS" \
    --output-dir "$SAVE_DIR" --name-prefix "$PREFIX" \
    --project "$WANDB_PROJECT" --wandb --python "$PYTHON" "$@"
