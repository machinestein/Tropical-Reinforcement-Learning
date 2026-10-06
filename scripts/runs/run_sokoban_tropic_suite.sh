#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$ROOT/scripts/runs/ragen2_dapo_overrides.sh"
SUITE=core
RUNS_OPTION=""
SEEDS_OPTION=10000
GPUS=0
MICRO_BATCH_SIZE=1
PPO_MINI_BATCH_SIZE=32
STEPS=200
EVAL_SEED=123
EVAL_PROBLEMS=512
EVAL_ATTEMPTS=1
EVAL_MODE=config
TEST_FREQ=10
SAVE_FREQ=100
MODEL=Qwen/Qwen2.5-3B-Instruct
PYTHON_BIN=${PYTHON:-python}
OUTPUT_DIR=""
PROJECT=sokoban_tropic_comparison
NAME_PREFIX=""
DRY_RUN=false
SKIP_COMPLETED=false
CONTINUE_ON_ERROR=false
LOGGERS='[console,file]'

CORE=(base_state ppo grpo ppo_snr grpo_snr verified_replay tropic)
ORIGINAL=(base_original original_ppo original_grpo original_ppo_snr)
ABLATIONS=(tropic_no_frontier tropic_no_composition tropic_successful_fragments_only tropic_no_coverage tropic_basis1)

usage() {
    printf '%s\n' \
        "Usage: bash scripts/runs/run_sokoban_tropic_suite.sh [options]" \
        "  --suite NAME          core (default), original, ablations, all" \
        "  --runs LIST           Explicit comma-separated run names; overrides --suite" \
        "  --seeds LIST          Training environment seed pools (default: 10000)" \
        "  --gpus LIST           GPUs for each sequential job (default: 0)" \
        "  --micro-batch-size N  Sequences per GPU for training/scoring (default: 1; try 4 on 8x A100 80GB)" \
        "  --steps N             Training iterations (default: 200)" \
        "  --eval-seed N         First held-out board seed (default: 123)" \
        "  --eval-problems N     Distinct held-out boards (default: 512)" \
        "  --eval-attempts N     Sampled attempts per board; logs pass@k for powers of two up to N (default: 1)" \
        "  --eval-mode MODE      config (default: inherit validation settings), or greedy" \
        "  --test-freq N         Validation interval, including final step (default: 10)" \
        "  --save-freq N         Checkpoint interval; -1 disables saving (default: 100)" \
        "  --model PATH          Initial HF model/path (default: Qwen/Qwen2.5-3B-Instruct)" \
        "  --python PATH         Python executable (default: PYTHON env var or python)" \
        "  --output-dir PATH     Artifact root (default: new timestamped results directory)" \
        "  --project NAME        W&B/file logging project (default: sokoban_tropic_comparison)" \
        "  --name-prefix NAME    Shared run/folder prefix; requires exactly one training seed" \
        "  --wandb               Also send metrics to W&B (console/file always enabled)" \
        "  --dry-run             Print shell-escaped commands; no Python or GPU jobs, no files" \
        "  --skip-completed      Skip a completed run only if its command is identical" \
        "  --continue-on-error   Try remaining runs after a failed job; still exit nonzero" \
        "  -h, --help            Show this help" \
        "" \
        "Core: ${CORE[*]}" \
        "Original protocol: ${ORIGINAL[*]}" \
        "Optional original protocol: original_dapo (RAGEN-2 DAPO; select with --runs)" \
        "Ablations: ${ABLATIONS[*]}" \
        "Extra diagnostic: tropic_pilot (frozen policy; select with --runs)"
}

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
require_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 requires a value"; }
uint() { [[ "$2" =~ ^(0|[1-9][0-9]{0,8})$ ]] || die "$1 must be a nonnegative integer below 1000000000"; }
positive() { uint "$1" "$2"; (( $2 > 0 )) || die "$1 must be positive"; }
hydra_string() {
    local value=${1//\\/\\\\}
    value=${value//\"/\\\"}
    printf '"%s"' "$value"
}
run_label() {
    case "$1" in
        base_state|base_original) printf 'EVAL' ;;
        original_ppo) printf 'PPO' ;;
        original_grpo) printf 'GRPO' ;;
        original_dapo) printf 'DAPO' ;;
        ppo_snr|original_ppo_snr) printf 'SNR' ;;
        *) printf '%s' "${1^^}" ;;
    esac
}

while (( $# )); do
    case "$1" in
        --suite|--runs|--seeds|--gpus|--micro-batch-size|--steps|--eval-seed|--eval-problems|--eval-attempts|--eval-mode|--test-freq|--save-freq|--model|--python|--output-dir|--project|--name-prefix)
            require_value "$@"
            case "$1" in
                --suite) SUITE=$2 ;; --runs) RUNS_OPTION=$2 ;; --seeds) SEEDS_OPTION=$2 ;;
                --gpus) GPUS=$2 ;; --steps) STEPS=$2 ;; --eval-seed) EVAL_SEED=$2 ;;
                --micro-batch-size) MICRO_BATCH_SIZE=$2 ;;
                --eval-problems) EVAL_PROBLEMS=$2 ;; --test-freq) TEST_FREQ=$2 ;;
                --eval-attempts) EVAL_ATTEMPTS=$2 ;;
                --eval-mode) EVAL_MODE=$2 ;;
                --save-freq) SAVE_FREQ=$2 ;; --model) MODEL=$2 ;; --python) PYTHON_BIN=$2 ;;
                --output-dir) OUTPUT_DIR=$2 ;;
                --project) PROJECT=$2 ;; --name-prefix) NAME_PREFIX=$2 ;;
            esac
            shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --skip-completed) SKIP_COMPLETED=true; shift ;;
        --continue-on-error) CONTINUE_ON_ERROR=true; shift ;;
        --wandb) LOGGERS='[console,file,wandb]'; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown argument: $1 (see --help)" ;;
    esac
done

positive --steps "$STEPS"
positive --micro-batch-size "$MICRO_BATCH_SIZE"
positive --eval-problems "$EVAL_PROBLEMS"
positive --eval-attempts "$EVAL_ATTEMPTS"
positive --test-freq "$TEST_FREQ"
uint --eval-seed "$EVAL_SEED"
[[ "$EVAL_MODE" == config || "$EVAL_MODE" == greedy ]] || die "--eval-mode must be config or greedy"
[[ "$EVAL_MODE" == config || "$EVAL_ATTEMPTS" == 1 ]] || die "--eval-attempts above 1 requires --eval-mode config (sampled validation)"
[[ "$SAVE_FREQ" == -1 ]] || positive --save-freq "$SAVE_FREQ"
[[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--gpus expects comma-separated numeric device IDs"
[[ "$SEEDS_OPTION" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--seeds expects comma-separated integers"
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
IFS=',' read -r -a SEEDS <<< "$SEEDS_OPTION"
if [[ -n "$NAME_PREFIX" ]]; then
    [[ "$NAME_PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "--name-prefix must be a safe filename (letters, numbers, _, ., -)"
    (( ${#SEEDS[@]} == 1 )) || die "--name-prefix requires one training seed to avoid duplicate run names"
fi
declare -A seen=()
for gpu in "${GPU_IDS[@]}"; do
    uint --gpus "$gpu"
    [[ -z ${seen[$gpu]+present} ]] || die "Duplicate GPU: $gpu"
    seen[$gpu]=1
done
seen=()
for seed in "${SEEDS[@]}"; do
    uint --seeds "$seed"
    [[ -z ${seen[$seed]+present} ]] || die "Duplicate seed: $seed"
    seen[$seed]=1
done

case "$SUITE" in
    core) RUNS=("${CORE[@]}") ;;
    original) RUNS=("${ORIGINAL[@]}") ;;
    ablations) RUNS=("${ABLATIONS[@]}") ;;
    all) RUNS=("${CORE[@]}" "${ORIGINAL[@]}" "${ABLATIONS[@]}") ;;
    *) die "Unknown suite: $SUITE" ;;
esac
if [[ -n "$RUNS_OPTION" ]]; then
    [[ "$RUNS_OPTION" =~ ^[a-z0-9_]+(,[a-z0-9_]+)*$ ]] || die "Malformed --runs list"
    IFS=',' read -r -a RUNS <<< "$RUNS_OPTION"
    SUITE=custom
fi
seen=()
declare -A seen_labels=()
JOB_COUNT=0
TRAIN_SPAN=0
for run in "${RUNS[@]}"; do
    case "$run" in
        base_state|base_original) JOB_COUNT=$((JOB_COUNT + 1)) ;;
        ppo|grpo|ppo_snr|grpo_snr|verified_replay|tropic|tropic_no_frontier|tropic_no_composition|tropic_successful_fragments_only|tropic_no_coverage|tropic_basis1|tropic_pilot)
            JOB_COUNT=$((JOB_COUNT + ${#SEEDS[@]}))
            (( TRAIN_SPAN >= 128 )) || TRAIN_SPAN=128 ;;
        original_ppo|original_grpo|original_ppo_snr|original_dapo)
            JOB_COUNT=$((JOB_COUNT + ${#SEEDS[@]}))
            (( STEPS * 8 <= TRAIN_SPAN )) || TRAIN_SPAN=$((STEPS * 8)) ;;
        *) die "Unknown run: $run" ;;
    esac
    [[ -z ${seen[$run]+present} ]] || die "Duplicate run: $run"
    seen[$run]=1
    if [[ -n "$NAME_PREFIX" ]]; then
        label=$(run_label "$run")
        [[ -z ${seen_labels[$label]+present} ]] || die "Duplicate run label $label with --name-prefix; omit the prefix when mixing original and state-only baselines"
        seen_labels[$label]=1
    fi
    case "$run" in
        base_*|ppo*|grpo*|original_ppo*|original_grpo|original_dapo)
            (( PPO_MINI_BATCH_SIZE % (${#GPU_IDS[@]} * MICRO_BATCH_SIZE) == 0 )) || \
                die "Global PPO minibatch $PPO_MINI_BATCH_SIZE must be divisible by GPU count (${#GPU_IDS[@]}) * --micro-batch-size ($MICRO_BATCH_SIZE)"
            ;;
    esac
done
for seed in "${SEEDS[@]}"; do
    if (( TRAIN_SPAN > 0 && seed <= EVAL_SEED + EVAL_PROBLEMS - 1 && seed + TRAIN_SPAN - 1 >= EVAL_SEED )); then
        die "Training seed range starting at $seed overlaps held-out evaluation seeds"
    fi
done

[[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="$ROOT/results/sokoban_tropic/$(date -u +%Y%m%dT%H%M%SZ)_$$"
[[ "$OUTPUT_DIR" == /* ]] || OUTPUT_DIR="$PWD/$OUTPUT_DIR"
if [[ "$PYTHON_BIN" == */* && "$PYTHON_BIN" != /* ]]; then
    PYTHON_BIN="$PWD/$PYTHON_BIN"
fi
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/ragen-matplotlib}
cd "$ROOT"

printf 'Suite: %s | jobs: %s | GPUs per job: %s | training seeds: %s\n' "$SUITE" "$JOB_COUNT" "$GPUS" "$SEEDS_OPTION"
printf 'Microbatch per GPU: %s | global PPO minibatch: %s\n' "$MICRO_BATCH_SIZE" "$PPO_MINI_BATCH_SIZE"
printf 'Evaluation: %s fixed boards from seed %s, mode=%s, %s attempt(s) per board (pass@1..%s)\n' \
    "$EVAL_PROBLEMS" "$EVAL_SEED" "$EVAL_MODE" "$EVAL_ATTEMPTS" "$EVAL_ATTEMPTS"
printf 'Artifacts: %s\n' "$OUTPUT_DIR"
if [[ "$DRY_RUN" == false ]]; then
    command -v "$PYTHON_BIN" >/dev/null || die "Python executable not found: $PYTHON_BIN"
    CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -c '
import sys
import torch
assert torch.cuda.is_available(), "A CUDA GPU is required; use --dry-run to inspect the suite."
assert torch.cuda.device_count() == int(sys.argv[1]), "Not all requested GPUs are visible."
from ragen.workers.actor.dp_actor import DataParallelPPOActor
print("GPU and actor dependency preflight passed.")
' "${#GPU_IDS[@]}" || die "GPU/dependency preflight failed; no experiments started"
fi

run_experiment() {
    local run=$1 seed=$2 config=_2_sokoban_state
    local run_id="${run}_seed${seed}" run_dir status
    local -a overrides=()
    case "$run" in
        base_original|original_ppo|original_grpo|original_ppo_snr|original_dapo)
            config=_2_sokoban
            overrides+=(actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode=softmax) ;;
        tropic*|verified_replay) config=_2_sokoban_tropic ;;
    esac
    case "$run" in
        base_*)
            run_id="${run}_eval${EVAL_SEED}"
            overrides+=(trainer.val_only=true trainer.total_training_steps=1 trainer.save_freq=-1
                        critic.enable=false actor_rollout_ref.actor.use_ref=false) ;;
        grpo*|original_grpo)
            overrides+=(algorithm.adv_estimator=grpo critic.enable=false
                        algorithm.norm_adv_by_std_in_grpo=true
                        actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean) ;;
        original_ppo*)
            overrides+=(algorithm.adv_estimator=gae critic.enable=true) ;;
        original_dapo) overrides+=("${RAGEN2_DAPO_OVERRIDES[@]}") ;;
        ppo*)
            overrides+=(algorithm.adv_estimator=gae critic.enable=true)
            if (( MICRO_BATCH_SIZE > 1 )); then
                overrides+=(actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
                            critic.loss_agg_mode=seq-mean-token-mean)
            fi
            ;;
    esac
    case "$run" in
        *_snr) overrides+=(actor_rollout_ref.rollout.rollout_filter_value=0.9) ;;
    esac
    case "$run" in
        verified_replay) overrides+=(tropic.frontier_enabled=false tropic.composition_enabled=false) ;;
        tropic_no_frontier) overrides+=(tropic.frontier_enabled=false) ;;
        tropic_no_composition) overrides+=(tropic.composition_enabled=false) ;;
        tropic_successful_fragments_only) overrides+=(+tropic.successful_fragments_only=true) ;;
        tropic_no_coverage) overrides+=(tropic.coverage_enabled=false) ;;
        tropic_basis1) overrides+=(tropic.basis_size=1) ;;
        tropic_pilot) overrides+=(tropic.collection_only=true) ;;
    esac
    if [[ "$EVAL_MODE" == greedy ]]; then
        overrides+=(actor_rollout_ref.rollout.val_kwargs.do_sample=false
                    actor_rollout_ref.rollout.val_kwargs.temperature=0.0
                    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
                    actor_rollout_ref.rollout.val_kwargs.top_k=-1)
    fi
    if [[ -n "$NAME_PREFIX" ]]; then
        run_id="${NAME_PREFIX}_$(run_label "$run")"
    fi
    run_dir="$OUTPUT_DIR/$run_id"
    local -a command=(env "CUDA_VISIBLE_DEVICES=$GPUS" "VERL_FILE_LOGGER_PATH=$run_dir/metrics.jsonl" "WANDB_DIR=$run_dir"
        "$PYTHON_BIN" -u "$ROOT/train.py" --config-name "$config"
        "model_path=$(hydra_string "$MODEL")"
        "system.CUDA_VISIBLE_DEVICES=$(hydra_string "$GPUS")"
        "trainer.n_gpus_per_node=${#GPU_IDS[@]}" trainer.nnodes=1
        "micro_batch_size_per_gpu=$MICRO_BATCH_SIZE" "ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
        "seed.train=$seed" "seed.val=$EVAL_SEED"
        "trainer.total_training_steps=$STEPS" "trainer.test_freq=$TEST_FREQ" "trainer.save_freq=$SAVE_FREQ"
        trainer.resume_mode=disable trainer.val_before_train=true trainer.validation_steps=1
        "es_manager.val.env_groups=$EVAL_PROBLEMS" "es_manager.val.env_configs.n_groups=[$EVAL_PROBLEMS]"
        "es_manager.val.group_size=$EVAL_ATTEMPTS"
        actor_rollout_ref.rollout.rollout_filter_strategy=top_p actor_rollout_ref.rollout.rollout_filter_value=1.0
        "trainer.project_name=$(hydra_string "$PROJECT")" "trainer.experiment_name=$run_id" "trainer.logger=$LOGGERS"
        "trainer.default_local_dir=$(hydra_string "$run_dir/checkpoints")"
        "trainer.local_log_dir=$(hydra_string "$run_dir/generations")"
        "trainer.validation_data_dir=$(hydra_string "$run_dir/validation")"
        "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH=$(hydra_string "$run_dir/metrics.jsonl")"
        "++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR=$(hydra_string "$run_dir")"
        "hydra.run.dir=$(hydra_string "$run_dir/hydra")" hydra.job.chdir=false
        "${overrides[@]}")
    local variable
    for variable in WANDB_ENTITY WANDB_MODE; do
        if [[ -n ${!variable:-} ]]; then
            command+=("++ray_kwargs.ray_init.runtime_env.env_vars.${variable}=$(hydra_string "${!variable}")")
        fi
    done
    local command_text
    printf -v command_text '%q ' "${command[@]}"
    printf '\nRUN %s\nCOMMAND %s\n' "$run_id" "$command_text"
    [[ "$DRY_RUN" == false ]] || return 0
    if [[ -e "$run_dir" ]]; then
        if [[ "$SKIP_COMPLETED" == true && -f "$run_dir/COMPLETED" && -f "$run_dir/command.sh" ]] \
                && [[ "$(<"$run_dir/command.sh")" == "$command_text" ]]; then
            printf 'Skipping completed run: %s\n' "$run_id"
            return 0
        fi
        printf 'Refusing to overwrite %s. Use a new --output-dir; incomplete runs are not resumed.\n' "$run_dir" >&2
        return 2
    fi
    mkdir -p "$OUTPUT_DIR" || return 2
    mkdir "$run_dir" || return 2
    printf '%s\n' "$command_text" > "$run_dir/command.sh"
    date -u +%FT%TZ > "$run_dir/started_at.txt"
    git rev-parse HEAD > "$run_dir/git_head.txt"
    git status --short > "$run_dir/git_status.txt"
    if "${command[@]}" 2>&1 | tee "$run_dir/train.log"; then
        printf '0\n' > "$run_dir/exit_code.txt"
        date -u +%FT%TZ > "$run_dir/COMPLETED"
        return 0
    else
        status=$?
        printf '%s\n' "$status" > "$run_dir/exit_code.txt"
        printf 'Run failed (%s): %s\n' "$status" "$run_id" >&2
        return "$status"
    fi
}

FAILED=0
for run in "${RUNS[@]}"; do
    RUN_SEEDS=("${SEEDS[@]}")
    [[ "$run" != base_* ]] || RUN_SEEDS=("${SEEDS[0]}")
    for seed in "${RUN_SEEDS[@]}"; do
        if run_experiment "$run" "$seed"; then
            :
        else
            FAILED=$((FAILED + 1))
            [[ "$CONTINUE_ON_ERROR" == true ]] || exit 1
        fi
    done
done
if [[ "$DRY_RUN" == true ]]; then
    printf '\nDry run complete: %s jobs planned; none launched.\n' "$JOB_COUNT"
else
    printf '\nFinished suite: %s planned jobs, %s failed.\n' "$JOB_COUNT" "$FAILED"
fi
(( FAILED == 0 ))
