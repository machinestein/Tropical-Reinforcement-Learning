#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    printf '%s\n' 'Usage: bash setup.sh' \
        'Creates venv for Qwen/Phi using the pinned Python 3.12 CUDA stack.' \
        'Options: BASE_PYTHON=python3, VENV_DIR=venv, MAX_JOBS=8.' \
        'FLASH_ATTN_WHEEL may specify a compatible FlashAttention 2.8.1 wheel.' \
        'Without a wheel, building FlashAttention requires a CUDA toolkit and compiler.'
    exit 0
fi
[[ $# == 0 ]] || { printf 'Unexpected argument; see --help.\n' >&2; exit 2; }
cd "$ROOT"
BASE_PYTHON=${BASE_PYTHON:-python3}
VENV_DIR=${VENV_DIR:-$ROOT/venv}
"$BASE_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"'
[[ $(realpath -m "$VENV_DIR") != $(realpath -m "$ROOT/gemma_venv") ]] || {
    printf 'Use scripts/setup_gemma_venv.sh for the Gemma environment.\n' >&2; exit 2;
}
"$BASE_PYTHON" -m venv "$VENV_DIR"
PYTHON="$VENV_DIR/bin/python"
"$PYTHON" -m pip install --upgrade pip wheel
"$PYTHON" -m pip install -r "$ROOT/requirements/qwen.lock.txt"
export MAX_JOBS=${MAX_JOBS:-8}
"$PYTHON" -m pip install --no-build-isolation "${FLASH_ATTN_WHEEL:-flash-attn==2.8.1}"
"$PYTHON" -m pip install --no-deps --no-build-isolation -e "$ROOT" -e "$ROOT/verl" -e "$ROOT/external/webshop-minimal"
printf '\nReady. Activate with: source %s/bin/activate\n' "$VENV_DIR"
