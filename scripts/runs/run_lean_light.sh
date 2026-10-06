#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
GPU=${GPU:-0}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
STEPS=${STEPS:-200}
TEST_FREQ=${TEST_FREQ:-25}
SAVE_FREQ=${SAVE_FREQ:-100}
EVAL_ATTEMPTS=${EVAL_ATTEMPTS:-8}
EVAL_PROBLEMS=${EVAL_PROBLEMS:-244}
EVAL_SEED=${EVAL_SEED:-123}
SEED=${SEED:-10000}
TRAIN_POOL_SIZE=${TRAIN_POOL_SIZE:-128}
EXPERIMENT=${EXPERIMENT:-lean_seed${SEED}}
SAVE_DIR=${SAVE_DIR:-$ROOT/saves}
PYTHON=${PYTHON:-python}
WANDB_PROJECT=${WANDB_PROJECT:-RAGEN_LEAN}
export WANDB_MODE=${WANDB_MODE:-disabled}
LEAN_SERVER_URL=${LEAN_SERVER_URL:-http://127.0.0.1:8000}
LEAN_DATASET=${LEAN_DATASET:-CoderBak/minif2f}
LEAN_DATASET_REVISION=${LEAN_DATASET_REVISION:-aa04566}
LEAN_TRAIN_PARTITION=${LEAN_TRAIN_PARTITION:-valid}
LEAN_VAL_PARTITION=${LEAN_VAL_PARTITION:-test}
LEAN_MAX_MODEL_LEN=${LEAN_MAX_MODEL_LEN:-8096}
RUNS=${RUNS:-EVAL,PPO,GRPO,SNR,TROPIC}
DRY_RUN=false

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
hydra_string() {
    local value=$1
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    printf '"%s"' "$value"
}
case "${1:-}" in
    -h|--help)
        printf '%s\n' \
            'Usage: bash scripts/runs/run_lean_light.sh [--dry-run]' \
            'Five sequential runs: initial EVAL, PPO, GRPO, SNR, then TROPIC.' \
            'RUNS=TROPIC (or a comma list such as EVAL,TROPIC) selects a subset, in the order above.' \
            'Baselines retain the original _7_lean policy/rewards; all runs use held-out theorem partitions.' \
            'Defaults: Qwen2.5-3B-Instruct, 200 steps, microbatch 1, validation every 25 steps.' \
            'Validation: 244 miniF2F-test theorems, 8 sampled attempts, pass@1/2/4/8.' \
            'Training: miniF2F-valid. A running Kimina server with Mathlib is required.' \
            'Context: 8096 tokens by default; LEAN_MAX_MODEL_LEN=16384 gives long proof states more room.' \
            'Larger contexts also raise the vLLM batch-token budget and enable chunked actor entropy.' \
            'Install only the client: python -m pip install "kimina-client==0.2.1"' \
            'Do not reinstall RAGEN extras into a working environment; they resolve legacy GPU dependency pins.' \
            'W&B: run wandb login once; optionally set WANDB_ENTITY; project defaults to RAGEN_LEAN.' \
            'Settings: RUNS, GPU, MODEL, MICRO_BATCH_SIZE, STEPS, TEST_FREQ, SAVE_FREQ, SEED, EXPERIMENT,' \
            'SAVE_DIR, PYTHON, WANDB_PROJECT, EVAL_ATTEMPTS, EVAL_PROBLEMS, EVAL_SEED, TRAIN_POOL_SIZE,' \
            'LEAN_SERVER_URL, LEAN_DATASET, LEAN_DATASET_REVISION, LEAN_TRAIN_PARTITION, LEAN_VAL_PARTITION,' \
            'LEAN_MAX_MODEL_LEN (must exceed the unchanged 512-token response limit).' \
            'Existing folders are never overwritten. Use a new EXPERIMENT for retries.' \
            'See README.md for the protocol and server setup.'
        exit 0 ;;
    --dry-run) [[ $# == 1 ]] || die 'Only --dry-run is supported'; DRY_RUN=true ;;
    '') [[ $# == 0 ]] || die 'Unexpected arguments' ;;
    *) die "Unknown argument: $1 (see --help)" ;;
esac
for variable in MICRO_BATCH_SIZE STEPS TEST_FREQ SAVE_FREQ EVAL_ATTEMPTS EVAL_PROBLEMS TRAIN_POOL_SIZE LEAN_MAX_MODEL_LEN; do
    [[ ${!variable} =~ ^[1-9][0-9]*$ ]] || die "$variable must be a positive integer"
done
(( LEAN_MAX_MODEL_LEN > 512 )) || die 'LEAN_MAX_MODEL_LEN must exceed the 512-token response limit'
for variable in SEED EVAL_SEED; do
    [[ ${!variable} =~ ^(0|[1-9][0-9]*)$ ]] || die "$variable must be a nonnegative integer"
done
[[ $GPU =~ ^[0-9]+(,[0-9]+)*$ ]] || die 'GPU must be comma-separated device indices'
IFS=, read -r -a GPU_IDS <<< "$GPU"
declare -A seen=()
for device in "${GPU_IDS[@]}"; do
    [[ ! -v seen[$device] ]] || die 'GPU indices must be unique'
    seen[$device]=1
done
(( 32 % (${#GPU_IDS[@]} * MICRO_BATCH_SIZE) == 0 )) || die 'Global minibatch 32 must divide into GPU microbatches'
[[ $EXPERIMENT =~ ^[A-Za-z0-9_.-]+$ ]] || die 'EXPERIMENT must contain only letters, numbers, dots, underscores or hyphens'
MODEL_LABEL=${MODEL##*/}
[[ $MODEL_LABEL =~ ^[A-Za-z0-9_.-]+$ ]] || die 'MODEL basename must be a valid run-name component'
[[ $LEAN_TRAIN_PARTITION != "$LEAN_VAL_PARTITION" ]] || die 'Train and validation partitions must differ'
ALL_METHODS=(EVAL PPO GRPO SNR TROPIC)
[[ $RUNS =~ ^[A-Z]+(,[A-Z]+)*$ ]] || die "RUNS must be a comma list of EVAL, PPO, GRPO, SNR, TROPIC (got '$RUNS')"
IFS=, read -r -a REQUESTED <<< "$RUNS"
declare -A wanted=()
for method in "${REQUESTED[@]}"; do
    [[ " ${ALL_METHODS[*]} " == *" $method "* ]] || die "RUNS must be a comma list of EVAL, PPO, GRPO, SNR, TROPIC (got '$method')"
    [[ ! -v wanted[$method] ]] || die "RUNS lists $method twice"
    wanted[$method]=1
done
METHODS=()
for method in "${ALL_METHODS[@]}"; do
    [[ -v wanted[$method] ]] && METHODS+=("$method")
done
(( ${#METHODS[@]} )) || die 'RUNS must select at least one run'
[[ $SAVE_DIR == /* ]] || SAVE_DIR="$ROOT/$SAVE_DIR"
PREFIX="RAGEN2_${MODEL_LABEL}_${EXPERIMENT}"
DATA_ARGS=(
    "custom_envs.Lean.env_config.dataset_name_or_path=$(hydra_string "$LEAN_DATASET")"
    "custom_envs.Lean.env_config.dataset_revision=$(hydra_string "$LEAN_DATASET_REVISION")"
    "custom_envs.Lean.env_config.dataset_partition=$(hydra_string "$LEAN_TRAIN_PARTITION")"
    "custom_envs.Lean.env_config.server_url=$(hydra_string "$LEAN_SERVER_URL")"
    "es_manager.val.env_config_overrides.Lean.dataset_partition=$(hydra_string "$LEAN_VAL_PARTITION")"
    "es_manager.val.env_groups=$EVAL_PROBLEMS" "es_manager.val.env_configs.n_groups=[$EVAL_PROBLEMS]"
    "es_manager.val.group_size=$EVAL_ATTEMPTS" "seed.val=$EVAL_SEED"
)
CONTEXT_ARGS=("actor_rollout_ref.rollout.max_model_len=$LEAN_MAX_MODEL_LEN")
if (( LEAN_MAX_MODEL_LEN > 8192 )); then
    CONTEXT_ARGS+=("actor_rollout_ref.rollout.max_num_batched_tokens=$LEAN_MAX_MODEL_LEN")
fi
if (( LEAN_MAX_MODEL_LEN > 8096 )); then
    CONTEXT_ARGS+=(actor_rollout_ref.actor.entropy_from_logits_with_chunking=true)
fi
printf 'Light Lean comparison: %s_{%s}\n' "$PREFIX" "$(IFS=,; printf '%s' "${METHODS[*]}")"
printf 'GPUs per run: %s | validation: %s theorems x %s attempts | artifacts: %s\n' "$GPU" "$EVAL_PROBLEMS" "$EVAL_ATTEMPTS" "$SAVE_DIR"
printf 'Context: %s tokens | response: 512 tokens | maximum prompt: %s tokens\n' "$LEAN_MAX_MODEL_LEN" "$((LEAN_MAX_MODEL_LEN - 512))"
printf 'Baseline policy: original Lean. TROPIC: goal-conditioned, root-verified proof fragments.\n'

MANIFEST=
if [[ $DRY_RUN == false ]]; then
    for method in "${METHODS[@]}"; do
        [[ ! -e "$SAVE_DIR/${PREFIX}_$method" ]] || die "Refusing to overwrite $SAVE_DIR/${PREFIX}_$method; use a new EXPERIMENT"
    done
    MANIFEST=$(mktemp)
    trap 'rm -f "$MANIFEST"' EXIT
    "$PYTHON" -m ragen.env.lean.preflight --output "$MANIFEST" --pool-size "$TRAIN_POOL_SIZE" "${DATA_ARGS[@]}" \
        || die 'Lean data/server preflight failed; no experiments started'
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -c '
import sys
import torch
assert torch.cuda.is_available(), "A CUDA GPU is required"
assert torch.cuda.device_count() == int(sys.argv[1]), "Not all requested GPUs are visible"
from ragen.workers.actor.dp_actor import DataParallelPPOActor
print("GPU and actor dependency preflight passed.")
' "${#GPU_IDS[@]}" || die 'GPU/dependency preflight failed; no experiments started'
fi

for method in "${METHODS[@]}"; do
    config=_7_lean_light
    overrides=(actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode=softmax)
    case "$method" in
        EVAL) overrides+=(trainer.val_only=true trainer.total_training_steps=1 trainer.save_freq=-1
                          critic.enable=false actor_rollout_ref.actor.use_ref=false) ;;
        PPO|SNR) overrides+=(algorithm.adv_estimator=gae critic.enable=true) ;;
        GRPO) overrides+=(algorithm.adv_estimator=grpo critic.enable=false algorithm.norm_adv_by_std_in_grpo=true
                          actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean) ;;
        TROPIC) config=_7_lean_tropic; overrides=("es_manager.train.seed_pool_size=$TRAIN_POOL_SIZE") ;;
    esac
    [[ $method != SNR ]] || overrides+=(actor_rollout_ref.rollout.rollout_filter_value=0.9)
    run_id="${PREFIX}_$method"
    run_dir="$SAVE_DIR/$run_id"
    command=(env "CUDA_VISIBLE_DEVICES=$GPU" "VERL_FILE_LOGGER_PATH=$run_dir/metrics.jsonl" "WANDB_DIR=$run_dir"
        "$PYTHON" -u "$ROOT/train.py" --config-name "$config" "${DATA_ARGS[@]}" "${CONTEXT_ARGS[@]}"
        "model_path=$(hydra_string "$MODEL")" "system.CUDA_VISIBLE_DEVICES=$(hydra_string "$GPU")"
        "trainer.n_gpus_per_node=${#GPU_IDS[@]}" trainer.nnodes=1
        "micro_batch_size_per_gpu=$MICRO_BATCH_SIZE" "seed.train=$SEED"
        "trainer.total_training_steps=$STEPS" "trainer.test_freq=$TEST_FREQ" "trainer.save_freq=$SAVE_FREQ"
        trainer.resume_mode=disable trainer.val_before_train=true trainer.validation_steps=1
        actor_rollout_ref.rollout.rollout_filter_strategy=top_p actor_rollout_ref.rollout.rollout_filter_value=1.0
        "trainer.project_name=$(hydra_string "$WANDB_PROJECT")" "trainer.experiment_name=$run_id"
        'trainer.logger=[console,file,wandb]'
        "trainer.default_local_dir=$(hydra_string "$run_dir/checkpoints")"
        "trainer.local_log_dir=$(hydra_string "$run_dir/generations")"
        "trainer.validation_data_dir=$(hydra_string "$run_dir/validation")"
        "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH=$(hydra_string "$run_dir/metrics.jsonl")"
        "++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR=$(hydra_string "$run_dir")"
        "hydra.run.dir=$(hydra_string "$run_dir/hydra")" hydra.job.chdir=false "${overrides[@]}")
    for variable in WANDB_ENTITY WANDB_MODE WANDB_BASE_URL; do
        if [[ -n ${!variable:-} ]]; then
            command+=("++ray_kwargs.ray_init.runtime_env.env_vars.$variable=$(hydra_string "${!variable}")")
        fi
    done
    printf -v command_text '%q ' "${command[@]}"
    printf '\nRUN %s\nCOMMAND %s\n' "$run_id" "$command_text"
    [[ $DRY_RUN == false ]] || continue
    mkdir -p "$SAVE_DIR"
    mkdir "$run_dir"
    cp "$MANIFEST" "$run_dir/dataset_manifest.json"
    printf '%s\n' "$command_text" > "$run_dir/command.sh"
    git rev-parse HEAD > "$run_dir/git_head.txt"
    git status --short > "$run_dir/git_status.txt"
    date -u +%FT%TZ > "$run_dir/started_at.txt"
    if "${command[@]}" 2>&1 | tee "$run_dir/train.log"; then
        printf '0\n' > "$run_dir/exit_code.txt"
        date -u +%FT%TZ > "$run_dir/COMPLETED"
    else
        status=$?
        printf '%s\n' "$status" > "$run_dir/exit_code.txt"
        printf 'Run failed (%s): %s\n' "$status" "$run_id" >&2
        exit "$status"
    fi
done
printf '\nLean suite finished (dry-run=%s).\n' "$DRY_RUN"
