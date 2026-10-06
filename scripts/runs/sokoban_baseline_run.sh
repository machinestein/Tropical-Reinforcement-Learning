#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
METHOD=${1:?Use run_sokoban_maxrl.sh or run_sokoban_tstar.sh}
shift
[[ $METHOD == MAXRL || $METHOD == TSTAR ]] || exit 2
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
GPU=${GPU:-0}
PYTHON=${PYTHON:-python}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
STEPS=${STEPS:-200}
TEST_FREQ=${TEST_FREQ:-25}
SAVE_FREQ=${SAVE_FREQ:-100}
EVAL_PROBLEMS=${EVAL_PROBLEMS:-512}
EVAL_ATTEMPTS=${EVAL_ATTEMPTS:-8}
SEED=${SEED:-10000}
EVAL_SEED=${EVAL_SEED:-123}
EXPERIMENT=${EXPERIMENT:-sokoban_seed${SEED}}
SAVE_DIR=${SAVE_DIR:-$ROOT/saves}
WANDB_PROJECT=${WANDB_PROJECT:-RAGEN2}
export WANDB_MODE=${WANDB_MODE:-disabled}
TSTAR_KL_SAMPLES=${TSTAR_KL_SAMPLES:-16}
TSTAR_GRAFT_BATCH_SIZE=${TSTAR_GRAFT_BATCH_SIZE:-128}
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
            "Usage: bash scripts/runs/run_sokoban_${METHOD,,}.sh [--dry-run]" \
            'One independent run. Default: Qwen2.5-3B-Instruct, GPU 0, 200 iterations.' \
            'Original light Sokoban: full history, 5 turns, up to 2 actions per turn, 8 boards x 16 attempts.' \
            'Validation: 512 boards x 8 attempts, temperature 0.5, every 25 iterations; save every 100.' \
            'Training uses binary terminal success. MAXRL: inverse-mean advantage, one unclipped update.' \
            'TSTAR: cognitive tree + GRPO credit + generated thought pairs + EMA-reference DPO.' \
            'Settings: MODEL, GPU, PYTHON, MICRO_BATCH_SIZE, STEPS, TEST_FREQ, SAVE_FREQ, SEED, EVAL_SEED,' \
            'EVAL_PROBLEMS, EVAL_ATTEMPTS, EXPERIMENT, SAVE_DIR, WANDB_PROJECT, WANDB_MODE.' \
            'T-STAR: TSTAR_KL_SAMPLES=16; TSTAR_GRAFT_BATCH_SIZE=128 uniformly samples stored pairs.' \
            'Set TSTAR_GRAFT_BATCH_SIZE=0 to use all accumulated pairs at every update.' \
            'Folders: RAGEN2_<model>_<experiment>_MAXRL or _TSTAR. Existing folders are never overwritten.' \
            'Training resume is not supported yet; saved actor checkpoints can be evaluated/exported.' \
            'Details: README.md'
        exit 0 ;;
    --dry-run) [[ $# == 1 ]] || die 'Unexpected arguments'; DRY_RUN=true ;;
    '') [[ $# == 0 ]] || die 'Unexpected arguments' ;;
    *) die "Unknown argument: $1 (see --help)" ;;
esac
for variable in MICRO_BATCH_SIZE STEPS TEST_FREQ SAVE_FREQ EVAL_PROBLEMS EVAL_ATTEMPTS TSTAR_KL_SAMPLES; do
    [[ ${!variable} =~ ^[1-9][0-9]*$ ]] || die "$variable must be a positive integer"
done
for variable in SEED EVAL_SEED TSTAR_GRAFT_BATCH_SIZE; do
    [[ ${!variable} =~ ^(0|[1-9][0-9]*)$ ]] || die "$variable must be a nonnegative integer"
done
[[ $GPU =~ ^[0-9]+(,[0-9]+)*$ ]] || die 'GPU must contain comma-separated device indices'
IFS=, read -r -a GPU_IDS <<< "$GPU"
declare -A seen=()
for device in "${GPU_IDS[@]}"; do
    [[ ! -v seen[$device] ]] || die 'GPU indices must be unique'
    seen[$device]=1
done
[[ $EXPERIMENT =~ ^[A-Za-z0-9_.-]+$ ]] || die 'Invalid EXPERIMENT name'
MODEL_LABEL=${MODEL##*/}
[[ $MODEL_LABEL =~ ^[A-Za-z0-9_.-]+$ ]] || die 'Invalid MODEL basename'
[[ $SAVE_DIR == /* ]] || SAVE_DIR="$ROOT/$SAVE_DIR"
run_id="RAGEN2_${MODEL_LABEL}_${EXPERIMENT}_${METHOD}"
run_dir="$SAVE_DIR/$run_id"
config="_2_sokoban_${METHOD,,}"
overrides=("model_path=$(hydra_string "$MODEL")" "system.CUDA_VISIBLE_DEVICES=$(hydra_string "$GPU")"
    "trainer.n_gpus_per_node=${#GPU_IDS[@]}" trainer.nnodes=1
    "micro_batch_size_per_gpu=$MICRO_BATCH_SIZE" "seed.train=$SEED" "seed.val=$EVAL_SEED"
    "trainer.total_training_steps=$STEPS" "trainer.test_freq=$TEST_FREQ" "trainer.save_freq=$SAVE_FREQ"
    trainer.resume_mode=disable trainer.val_before_train=true trainer.validation_steps=1
    "es_manager.val.env_groups=$EVAL_PROBLEMS" "es_manager.val.env_configs.n_groups=[$EVAL_PROBLEMS]"
    "es_manager.val.group_size=$EVAL_ATTEMPTS"
    "trainer.project_name=$(hydra_string "$WANDB_PROJECT")" "trainer.experiment_name=$run_id"
    'trainer.logger=[console,file,wandb]'
    "trainer.default_local_dir=$(hydra_string "$run_dir/checkpoints")"
    "trainer.local_log_dir=$(hydra_string "$run_dir/generations")"
    "trainer.validation_data_dir=$(hydra_string "$run_dir/validation")"
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH=$(hydra_string "$run_dir/metrics.jsonl")"
    "++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR=$(hydra_string "$run_dir")"
    "hydra.run.dir=$(hydra_string "$run_dir/hydra")" hydra.job.chdir=false)
if [[ $METHOD == TSTAR ]]; then
    overrides+=("tstar.kl_samples=$TSTAR_KL_SAMPLES" "tstar.graft_batch_size=$TSTAR_GRAFT_BATCH_SIZE")
fi
for variable in WANDB_ENTITY WANDB_MODE WANDB_BASE_URL; do
    if [[ -n ${!variable:-} ]]; then
        overrides+=("++ray_kwargs.ray_init.runtime_env.env_vars.$variable=$(hydra_string "${!variable}")")
    fi
done
command=(env "CUDA_VISIBLE_DEVICES=$GPU" "VERL_FILE_LOGGER_PATH=$run_dir/metrics.jsonl" "WANDB_DIR=$run_dir"
    "$PYTHON" -u "$ROOT/train.py" --config-name "$config" "${overrides[@]}")
printf -v command_text '%q ' "${command[@]}"
printf 'RUN %s\nCOMMAND %s\n' "$run_id" "$command_text"
[[ $DRY_RUN == false ]] || exit 0
[[ ! -e $run_dir ]] || die "Refusing to overwrite $run_dir; choose a new EXPERIMENT"
CONFIG_TMP=$(mktemp)
trap 'rm -f "$CONFIG_TMP"' EXIT
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m ragen.sokoban_baselines.preflight \
    --config-name "$config" --output "$CONFIG_TMP" --require-cuda "${overrides[@]}" \
    || die 'Preflight failed; no experiment started'
mkdir -p "$SAVE_DIR"
mkdir "$run_dir"
cp "$CONFIG_TMP" "$run_dir/config.yaml"
printf '%s\n' "$command_text" > "$run_dir/command.sh"
date -u +%FT%TZ > "$run_dir/STARTED"
if "${command[@]}" 2>&1 | tee "$run_dir/train.log"; then
    date -u +%FT%TZ > "$run_dir/COMPLETED"
else
    status=$?
    date -u +%FT%TZ > "$run_dir/FAILED"
    exit "$status"
fi
