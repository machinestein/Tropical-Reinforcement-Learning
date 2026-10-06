"""Exercise setup orchestration without installing into the working venv."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/setup_webshop_venv.sh"


def test_webshop_setup_help_does_not_invoke_python():
    result = subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True, text=True,
                            env={**os.environ, "PYTHON": "/nonexistent"}, timeout=10)
    assert result.returncode == 0
    assert "Java 21+" in result.stdout and "without dependencies" in result.stdout


@pytest.mark.parametrize("guard_status", [0, 9])
def test_webshop_setup_guards_and_constraints(tmp_path, guard_status):
    fake_python = tmp_path / "python"
    calls = tmp_path / "calls.jsonl"
    fake_python.write_text(f"#!{sys.executable}\n" + '''
import json
import os
from pathlib import Path
import sys
args = sys.argv[1:]
constraint = (args[args.index("--constraint") + 1] if "--constraint" in args
              else os.environ.get("PIP_CONSTRAINT"))
record = {"args": args, "constraints": Path(constraint).read_text() if constraint else None}
with Path(os.environ["SETUP_CALLS"]).open("a") as stream:
    stream.write(json.dumps(record) + "\\n")
if args[0] == "-c" and "sys.prefix" in args[1]:
    sys.exit(int(os.environ["GUARD_STATUS"]))
if args[0] == "-":
    code = sys.stdin.read()
    if len(args) == 2:
        sys.argv = args
        exec(compile(code, "<setup constraints>", "exec"))
''')
    fake_python.chmod(0o755)
    result = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, timeout=30,
                            env={**os.environ, "PYTHON": str(fake_python), "SETUP_CALLS": str(calls),
                                 "GUARD_STATUS": str(guard_status)})
    assert result.returncode == guard_status, result.stderr
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    if guard_status:
        assert len(records) == 1
        return
    assert len(records) == 7
    install, model, editable = records[3:6]
    assert install["args"][:3] == ["-m", "pip", "install"]
    assert "torch==" in install["constraints"] and "vllm==" in install["constraints"]
    assert "--no-deps" in editable["args"] and "--no-build-isolation" in editable["args"]
    assert model["args"] == ["-m", "spacy", "download", "en_core_web_sm"]
    assert model["constraints"] == install["constraints"]
    assert not Path(records[2]["args"][1]).exists()
