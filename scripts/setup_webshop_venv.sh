#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-python}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    printf '%s\n' \
        'Usage: PYTHON=venv/bin/python bash scripts/setup_webshop_venv.sh' \
        'Requires a working RAGEN venv and Java 21+ on PATH (and JAVA_HOME if needed by PyJNIus).' \
        'Installs WebShop extras with every already-installed package version constrained.' \
        'Downloads en_core_web_sm, then installs the local simulator without dependencies.' \
        'Does not install system packages, use Conda, rebuild indexes or download the full dataset.'
    exit 0
fi
[[ $# == 0 ]] || { printf 'Unexpected arguments; see --help.\n' >&2; exit 2; }
"$PYTHON" -c 'import sys; assert sys.prefix != sys.base_prefix, "Activate your venv or set PYTHON=venv/bin/python"'
"$PYTHON" - <<'PY'
import re
import shutil
import subprocess
if not shutil.which("java"):
    raise SystemExit("Java is missing. Install Java 21+ through your system administrator, then retry.")
result = subprocess.run(["java", "-version"], capture_output=True, text=True, check=True)
match = re.search(r'version "(\d+)', result.stderr + result.stdout)
if match is None or int(match.group(1)) < 21:
    raise SystemExit("Use Java 21+ for this WebShop setup; check PATH and JAVA_HOME.")
PY

CONSTRAINTS=$(mktemp)
trap 'rm -f "$CONSTRAINTS"' EXIT
"$PYTHON" - "$CONSTRAINTS" <<'PY'
from importlib import metadata
from pathlib import Path
import sys
versions = {dist.metadata['Name']: dist.version for dist in metadata.distributions() if dist.metadata['Name']}
Path(sys.argv[1]).write_text(''.join(f'{name}=={version}\n' for name, version in sorted(versions.items())))
PY

"$PYTHON" -m pip install --constraint "$CONSTRAINTS" -r external/webshop-minimal/requirements.txt
PIP_CONSTRAINT="$CONSTRAINTS" "$PYTHON" -m spacy download en_core_web_sm
"$PYTHON" -m pip install --no-deps --no-build-isolation -e external/webshop-minimal
"$PYTHON" -c 'from ragen.env.webshop.tropic_env import WebShopTropicEnv; print("WebShop imports OK; run the light launcher preflight next.")'
