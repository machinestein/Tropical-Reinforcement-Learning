#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
GEMMA_VENV=${GEMMA_VENV:-$ROOT/gemma_venv}
BASE_PYTHON=${BASE_PYTHON:-python3}
[[ $(realpath -m "$GEMMA_VENV") != $(realpath -m "$ROOT/venv") ]] || {
    printf 'Refusing to install Gemma packages into the existing venv.\n' >&2; exit 2;
}
"$BASE_PYTHON" -m venv "$GEMMA_VENV"
"$GEMMA_VENV/bin/python" -m pip install -r "$ROOT/requirements/gemma.lock.txt"
"$GEMMA_VENV/bin/python" - "$ROOT" <<'PY'
from pathlib import Path
import sys
import sysconfig
root = Path(sys.argv[1]).resolve()
pth = Path(sysconfig.get_paths()['purelib']) / 'ragen_gemma.pth'
# Avoid resolving the root package's legacy vLLM dependency pins. Runtime
# compatibility is enabled only in this dedicated environment, including Ray.
pth.write_text('\n'.join([
    str(root), str(root / 'verl'), str(root / 'external/webshop-minimal'),
    'import ragen.gemma.compat; ragen.gemma.compat.activate()', '',
]))
PY
printf '\nGemma environment ready. Activate it with:\n  source %s/bin/activate\n' "$GEMMA_VENV"
printf 'Then prepare weights once:\n  python -m ragen.gemma.prepare\n'
