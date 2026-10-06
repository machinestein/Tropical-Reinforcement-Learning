#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$ROOT/scripts/runs/ragen2_dapo_overrides.sh"
cd "$ROOT"

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
GPU=${GPU:-0}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
STEPS=${STEPS:-200}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-100}
EVAL_ATTEMPTS=${EVAL_ATTEMPTS:-1}
EVAL_PROBLEMS=${EVAL_PROBLEMS:-512}
EVAL_SEED=${EVAL_SEED:-123}
SEED=${SEED:-10000}
TRAIN_POOL_SIZE=${TRAIN_POOL_SIZE:-128}
DATA=${DATA:-tropic}
PROTOCOL=${PROTOCOL:-original}
WARMUP_MODEL=${WARMUP_MODEL:-}
WARMUP_STEPS=${WARMUP_STEPS:-50}
WARMUP_MICRO_BATCH_SIZE=${WARMUP_MICRO_BATCH_SIZE:-1}
EXPERIMENT=${EXPERIMENT:-countdown_seed${SEED}}
SAVE_DIR=${SAVE_DIR:-$ROOT/saves}
PYTHON=${PYTHON:-python}
WANDB_PROJECT=${WANDB_PROJECT:-RAGEN_COUNTDOWN}
export WANDB_MODE=${WANDB_MODE:-disabled}
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
            'Usage: bash scripts/runs/run_countdown_light.sh [--dry-run]' \
            'Five sequential jobs: EVAL, PPO, GRPO, SNR, TROPIC; all selected GPUs per job.' \
            'RUNS=TROPIC (or a comma list such as EVAL,TROPIC) selects a subset, in the order above.' \
            'RUNS=DAPO selects RAGEN-2 DAPO: PPO/GAE, clip 0.2/0.28, no KL, token-mean, no SNR filtering.' \
            'Optional TROPIC variants: TROPIC_NO_COMPOSITION, TROPIC_SUCCESSFUL_FRAGMENTS_ONLY, TROPIC_NO_FRONTIER.' \
            'See countdown_ablations.sh for the full-TROPIC control plus these three ablations.' \
            'Baselines load the original _4_countdown directly (single-turn expression, original rewards).' \
            'Only TROPIC loads _4_countdown_tropic: one exact binary operation per response over the live numbers.' \
            'PROTOCOL=atomic: all five use _4_countdown_atomic, identical binary rewards and a shared atomic warmup.' \
            'Atomic mode appends _atomic to the experiment prefix, and first trains one 50-step full-weight warmup.' \
            'WARMUP_MODEL=/path/to/completed/WARMUP/model reuses a verified warmup; RL starts from its weights.' \
            'DATA=tropic (default): generated n=3..6 problems with + - * /, disjoint train/validation files.' \
            'DATA=original: RAGEN data/countdown/train.parquet (3-4 inputs, +/- solvable) for every run.' \
            'Defaults: Qwen2.5-3B-Instruct, microbatch 1, minibatch 32, 200 steps, validation every 10 steps,' \
            '512 validation problems x 1 sampled attempt. Set EVAL_ATTEMPTS=8 TEST_FREQ=25 for nested pass@1..8.' \
            'W&B (opt-in with WANDB_MODE=online): wandb login; project defaults to RAGEN_COUNTDOWN; optional WANDB_ENTITY.' \
            'Settings: PROTOCOL, WARMUP_MODEL, WARMUP_STEPS, WARMUP_MICRO_BATCH_SIZE, RUNS, DATA, GPU, MODEL, MICRO_BATCH_SIZE, STEPS, TEST_FREQ, SAVE_FREQ, SEED, EXPERIMENT, SAVE_DIR,' \
            'PYTHON, WANDB_PROJECT, EVAL_ATTEMPTS, EVAL_PROBLEMS, EVAL_SEED, TRAIN_POOL_SIZE.' \
            'Only selected folders are checked. Archive a failed folder or use a new EXPERIMENT to retry.' \
            'See README.md for the protocol.'
        exit 0 ;;
    --dry-run) [[ $# == 1 ]] || die 'Only --dry-run is supported'; DRY_RUN=true ;;
    '') [[ $# == 0 ]] || die 'Unexpected arguments' ;;
    *) die "Unknown argument: $1 (see --help)" ;;
esac
for variable in MICRO_BATCH_SIZE STEPS TEST_FREQ SAVE_FREQ EVAL_ATTEMPTS EVAL_PROBLEMS TRAIN_POOL_SIZE; do
    [[ ${!variable} =~ ^[1-9][0-9]*$ ]] || die "$variable must be a positive integer"
done
for variable in SEED EVAL_SEED; do
    [[ ${!variable} =~ ^(0|[1-9][0-9]*)$ ]] || die "$variable must be a nonnegative integer"
done
[[ $DATA == tropic || $DATA == original ]] || die 'DATA must be tropic or original'
[[ $PROTOCOL == original || $PROTOCOL == atomic ]] || die 'PROTOCOL must be original or atomic'
if [[ $PROTOCOL == atomic ]]; then
    [[ $DATA == tropic ]] || die 'PROTOCOL=atomic requires disjoint DATA=tropic splits'
    for variable in WARMUP_STEPS WARMUP_MICRO_BATCH_SIZE; do
        [[ ${!variable} =~ ^[1-9][0-9]*$ ]] || die "$variable must be a positive integer"
    done
else
    [[ -z $WARMUP_MODEL ]] || die 'WARMUP_MODEL is only supported with PROTOCOL=atomic'
fi
[[ $DATA == original || $EVAL_PROBLEMS -le 512 ]] || die 'The generated validation file has 512 distinct problems'
[[ $GPU =~ ^[0-9]+(,[0-9]+)*$ ]] || die 'GPU must be comma-separated device indices'
IFS=, read -r -a GPU_IDS <<< "$GPU"
declare -A seen=()
for device in "${GPU_IDS[@]}"; do
    [[ ! -v seen[$device] ]] || die 'GPU indices must be unique'
    seen[$device]=1
done
(( 32 % (${#GPU_IDS[@]} * MICRO_BATCH_SIZE) == 0 )) || die 'Global minibatch 32 must divide into GPU microbatches'
[[ $EXPERIMENT =~ ^[A-Za-z0-9_.-]+$ ]] || die 'Invalid EXPERIMENT name'
MODEL_LABEL=${MODEL##*/}
[[ $MODEL_LABEL =~ ^[A-Za-z0-9_.-]+$ ]] || die 'MODEL basename must be a valid run-name component'
ALL_METHODS=(EVAL PPO GRPO SNR DAPO TROPIC TROPIC_NO_COMPOSITION TROPIC_SUCCESSFUL_FRAGMENTS_ONLY TROPIC_NO_FRONTIER)
[[ $RUNS =~ ^[A-Z_]+(,[A-Z_]+)*$ ]] || die "RUNS must be a comma list of ${ALL_METHODS[*]} (got '$RUNS')"
IFS=, read -r -a REQUESTED <<< "$RUNS"
declare -A wanted=()
for method in "${REQUESTED[@]}"; do
    [[ " ${ALL_METHODS[*]} " == *" $method "* ]] || die "RUNS must be a comma list of ${ALL_METHODS[*]} (got '$method')"
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
TRAIN_MODEL=$MODEL
if [[ $PROTOCOL == atomic ]]; then
    (( 128 % (${#GPU_IDS[@]} * WARMUP_MICRO_BATCH_SIZE) == 0 )) || die 'Warmup global batch 128 must divide into GPU microbatches'
    PREFIX+=_atomic
    WARMUP_DIR="$SAVE_DIR/${PREFIX}_WARMUP"
    TRAIN_MODEL=${WARMUP_MODEL:-$WARMUP_DIR/model}
    [[ $TRAIN_MODEL == /* ]] || TRAIN_MODEL="$ROOT/$TRAIN_MODEL"
fi
EVAL_ARGS=("es_manager.val.env_groups=$EVAL_PROBLEMS" "es_manager.val.env_configs.n_groups=[$EVAL_PROBLEMS]"
           "es_manager.val.group_size=$EVAL_ATTEMPTS" "seed.val=$EVAL_SEED")
BASELINE_DATA_ARGS=()
TROPIC_DATA_ARGS=()
if [[ $DATA == tropic ]]; then
    BASELINE_DATA_ARGS=('custom_envs.Countdown.env_config={train_path: data/countdown/tropic_train.parquet, solvable_filter: false, max_instances: 2048}'
                        '++es_manager.val.env_config_overrides.Countdown.train_path=data/countdown/tropic_val.parquet'
                        '++es_manager.val.env_config_overrides.Countdown.max_instances=512')
else
    TROPIC_DATA_ARGS=(custom_envs.Countdown.env_config.train_path=data/countdown/train.parquet
                      custom_envs.Countdown.env_config.solvable_filter=true
                      custom_envs.Countdown.env_config.max_instances=20000
                      es_manager.val.env_config_overrides.Countdown.train_path=data/countdown/train.parquet
                      es_manager.val.env_config_overrides.Countdown.max_instances=20000)
fi
printf 'Light Countdown: %s_{%s}\n' "$PREFIX" "$(IFS=,; printf '%s' "${METHODS[*]}")"
printf 'GPUs per job: %s | data: %s | validation: %s problems x %s attempts | W&B: %s\n' \
    "$GPU" "$DATA" "$EVAL_PROBLEMS" "$EVAL_ATTEMPTS" "$WANDB_PROJECT"
if [[ $PROTOCOL == atomic ]]; then
    printf 'Matched atomic protocol: shared warmup, one operation per turn, binary rewards, recurring training pool.\n'
    printf 'Initial checkpoint for every method: %s\n' "$TRAIN_MODEL"
else
    printf 'Baselines: original _4_countdown single-turn expressions. TROPIC: one operation per step, exact states.\n'
fi

MANIFEST=
if [[ $DRY_RUN == false ]]; then
    for method in "${METHODS[@]}"; do
        [[ ! -e "$SAVE_DIR/${PREFIX}_$method" ]] || die "Refusing to overwrite $SAVE_DIR/${PREFIX}_$method; use a new EXPERIMENT"
    done
    MANIFEST=$(mktemp)
    trap 'rm -f "$MANIFEST"' EXIT
    "$PYTHON" -m ragen.env.countdown.preflight --output "$MANIFEST" --pool-size "$TRAIN_POOL_SIZE" \
        "seed.train=$SEED" "${EVAL_ARGS[@]}" "${TROPIC_DATA_ARGS[@]}" \
        || die 'Countdown preflight failed; no experiments started. See README.md'
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -c '
import sys
import torch
assert torch.cuda.is_available(), "A CUDA GPU is required"
assert torch.cuda.device_count() == int(sys.argv[1]), "Not all selected GPUs are visible"
from ragen.workers.actor.dp_actor import DataParallelPPOActor
print("GPU and actor dependency preflight passed.")
' "${#GPU_IDS[@]}" || die 'GPU/dependency preflight failed; no experiments started'
fi

if [[ $PROTOCOL == atomic ]]; then
    warmup_command=(env "CUDA_VISIBLE_DEVICES=$GPU" "WANDB_PROJECT=$WANDB_PROJECT" "WANDB_DIR=$WARMUP_DIR"
        "$PYTHON" -m torch.distributed.run --standalone "--nproc_per_node=${#GPU_IDS[@]}"
        --module ragen.env.countdown.warmup --model "$MODEL" --output "$WARMUP_DIR"
        --steps "$WARMUP_STEPS" --micro-batch-size "$WARMUP_MICRO_BATCH_SIZE")
    if [[ -n $WARMUP_MODEL || -f "$WARMUP_DIR/COMPLETED" ]]; then
        printf 'Reusing shared warmup: %s\n' "$TRAIN_MODEL"
    else
        printf 'WARMUP_COMMAND '; printf '%q ' "${warmup_command[@]}"; printf '\n'
        if [[ $DRY_RUN == false ]]; then
            [[ ! -e $WARMUP_DIR ]] || die 'Incomplete warmup folder exists; use a new EXPERIMENT'
            mkdir -p "$SAVE_DIR"
            "${warmup_command[@]}" 2>&1 | tee "$SAVE_DIR/${PREFIX}_WARMUP.log"
        fi
    fi
    if [[ $DRY_RUN == false ]]; then
        "$PYTHON" -m ragen.env.countdown.warmup --model "$MODEL" --check-model "$TRAIN_MODEL" \
            || die 'Shared warmup verification failed; no RL experiments started'
    fi
fi

for method in "${METHODS[@]}"; do
    config=_4_countdown
    overrides=(actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode=softmax "${BASELINE_DATA_ARGS[@]}")
    case "$method" in
        EVAL) overrides+=(trainer.val_only=true trainer.total_training_steps=1 trainer.save_freq=-1
                          critic.enable=false actor_rollout_ref.actor.use_ref=false) ;;
        PPO|SNR) overrides+=(algorithm.adv_estimator=gae critic.enable=true) ;;
        GRPO) overrides+=(algorithm.adv_estimator=grpo critic.enable=false algorithm.norm_adv_by_std_in_grpo=true
                          actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean) ;;
        TROPIC|TROPIC_*) config=_4_countdown_tropic
                overrides=("es_manager.train.seed_pool_size=$TRAIN_POOL_SIZE" "${TROPIC_DATA_ARGS[@]}") ;;
    esac
    if [[ $PROTOCOL == atomic ]]; then
        config=_4_countdown_atomic
        overrides=("es_manager.train.seed_pool_size=$TRAIN_POOL_SIZE"
                   actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode=softmax)
        case "$method" in
            EVAL) overrides+=(trainer.val_only=true trainer.total_training_steps=1 trainer.save_freq=-1
                              critic.enable=false) ;;
            PPO|SNR) overrides+=(algorithm.adv_estimator=gae critic.enable=true) ;;
            GRPO) overrides+=(algorithm.adv_estimator=grpo critic.enable=false algorithm.norm_adv_by_std_in_grpo=true
                              actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean) ;;
            TROPIC|TROPIC_*) overrides+=(trainer.method=tropic critic.enable=false es_manager.train.group_size=8) ;;
        esac
    fi
    case "$method" in
        DAPO) overrides+=("${RAGEN2_DAPO_OVERRIDES[@]}") ;;
        TROPIC_NO_COMPOSITION) overrides+=(tropic.composition_enabled=false) ;;
        TROPIC_SUCCESSFUL_FRAGMENTS_ONLY) overrides+=(+tropic.successful_fragments_only=true) ;;
        TROPIC_NO_FRONTIER) overrides+=(tropic.frontier_enabled=false) ;;
    esac
    [[ $method != SNR ]] || overrides+=(actor_rollout_ref.rollout.rollout_filter_value=0.9)
    run_id="${PREFIX}_$method"
    run_dir="$SAVE_DIR/$run_id"
    command=(env "CUDA_VISIBLE_DEVICES=$GPU" "VERL_FILE_LOGGER_PATH=$run_dir/metrics.jsonl" "WANDB_DIR=$run_dir"
        "$PYTHON" -u "$ROOT/train.py" --config-name "$config" "${EVAL_ARGS[@]}"
        "model_path=$(hydra_string "$TRAIN_MODEL")" "system.CUDA_VISIBLE_DEVICES=$(hydra_string "$GPU")"
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
    if [[ $PROTOCOL == atomic ]]; then
        cp "$(dirname "$TRAIN_MODEL")/warmup_manifest.json" "$run_dir/warmup_manifest.json"
    fi
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
printf '\nCountdown suite finished (dry-run=%s).\n' "$DRY_RUN"
