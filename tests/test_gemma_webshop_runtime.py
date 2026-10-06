"""Exercise JVM preloading before Python starts, without training or GPU allocation."""

import os
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/runs/gemma_webshop_runtime.sh"
WRAPPER = ROOT / "scripts/runs/run_gemma_light.sh"


def settings(**overrides):
    env = dict(os.environ)
    for name in ("JAVA_HOME", "JDK_HOME", "JRE_HOME", "JVM_PATH", "LD_PRELOAD", "MODEL"):
        env.pop(name, None)
    env.update(overrides)
    return env


@pytest.fixture
def jdk():
    compiler = shutil.which("javac")
    if not compiler:
        pytest.skip("A JDK is required for the native signal-chaining check")
    home = Path(compiler).resolve().parent.parent
    library = home / "lib/libjsig.so"
    if not library.is_file():
        pytest.skip("The selected JDK has no libjsig.so")
    return home


def test_preserves_existing_preloads_and_does_not_duplicate_library(jdk):
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; configure_gemma_webshop_runtime; '
         'configure_gemma_webshop_runtime; printf "PRELOAD=%s\\n" "$LD_PRELOAD"', "probe", str(HELPER)],
        env=settings(JAVA_HOME=str(jdk), LD_PRELOAD="libm.so.6"),
        text=True, capture_output=True, check=True,
    )
    preload = next(line.removeprefix("PRELOAD=") for line in result.stdout.splitlines() if line.startswith("PRELOAD="))
    assert preload == f"{(jdk / 'lib/libjsig.so').resolve()}:libm.so.6"


def test_missing_selected_jdk_fails_before_python(tmp_path):
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; configure_gemma_webshop_runtime', "probe", str(HELPER)],
        env=settings(JAVA_HOME=str(tmp_path)), text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert "libjsig.so is missing" in result.stderr


def test_explicit_jvm_path_selects_its_own_signal_library(jdk):
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; configure_gemma_webshop_runtime', "probe", str(HELPER)],
        env=settings(JAVA_HOME="/unused", JVM_PATH=str(jdk / "lib/server/libjvm.so")),
        text=True, capture_output=True, check=True,
    )
    assert str((jdk / "lib/server/libjsig.so").resolve()) in result.stdout


@pytest.mark.parametrize("task,preload_expected", [("webshop", True), ("sokoban", False)])
def test_wrapper_applies_fix_only_to_webshop_before_first_python(tmp_path, jdk, task, preload_expected):
    executable = tmp_path / "fake-python"
    executable.write_text('#!/usr/bin/env bash\nprintf "PRELOAD=%s\\n" "${LD_PRELOAD:-}"\nexit 42\n')
    executable.chmod(0o755)
    result = subprocess.run(
        ["bash", str(WRAPPER), task], cwd=ROOT,
        env=settings(JAVA_HOME=str(jdk), PYTHON=str(executable), RUNS="EVAL", GPU="0"),
        text=True, capture_output=True,
    )
    assert result.returncode == 42  
    preload = next(line.removeprefix("PRELOAD=") for line in result.stdout.splitlines() if line.startswith("PRELOAD="))
    assert bool(preload) == preload_expected
    if preload_expected:
        assert preload == str((jdk / "lib/libjsig.so").resolve())


def test_webshop_dry_run_needs_neither_jdk_nor_python(tmp_path):
    result = subprocess.run(
        ["bash", str(WRAPPER), "webshop", "--dry-run"], cwd=ROOT,
        env=settings(JAVA_HOME="/missing", PYTHON="/missing", RUNS="EVAL", GPU="0",
                     EVAL_PROBLEMS="512", EVAL_ATTEMPTS="1", SAVE_DIR=str(tmp_path / "unused")),
        text=True, capture_output=True, check=True,
    )
    assert "es_manager.val.env_groups=512" in result.stdout
    assert "es_manager.val.group_size=1" in result.stdout
    assert "trainer.val_only=true" in result.stdout
    assert "JVM signal chaining" not in result.stdout
    assert not (tmp_path / "unused").exists()


def test_late_tvm_load_preserves_jvm_exception_handling(tmp_path, jdk):
    python = ROOT / "gemma_venv/bin/python"
    libraries = list((ROOT / "gemma_venv/lib").glob("python*/site-packages/tvm_ffi/lib/libtvm_ffi.so"))
    if not python.is_file() or len(libraries) != 1:
        pytest.skip("Requires the pinned Gemma venv and TVM-FFI shared library")
    library = libraries[0]
    site = library.parents[2]
    source = tmp_path / "SignalProbe.java"
    source.write_text('''public class SignalProbe {
  static int length(String s) { return s.length(); }
  public static int warm() { int n=0; for (int i=0; i<2000000; ++i) n+=length("abc"); return n; }
  public static int check() { int n=0; for (int i=0; i<1000; ++i) { try { length(null); } catch (NullPointerException e) { ++n; } } return n; }
}''')
    subprocess.run([str(jdk / "bin/javac"), str(source)], check=True, timeout=30, capture_output=True)
    probe = tmp_path / "probe.py"
    probe.write_text('''import ctypes, resource, sys
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
sys.path.insert(0, sys.argv[1])
import jnius_config
jnius_config.set_classpath(sys.argv[2])
jnius_config.add_options("-Xbatch", "-XX:-TieredCompilation")
from jnius import autoclass
Probe = autoclass("SignalProbe")
assert Probe.warm() == 6000000
ctypes.CDLL(sys.argv[3])
assert Probe.check() == 1000
print("JVM signal chaining passed")
''')
    result = subprocess.run(
        ["bash", "-ec", 'source "$1"; configure_gemma_webshop_runtime; shift; exec "$@"',
         "probe", str(HELPER), str(python), "-S", str(probe), str(site), str(tmp_path), str(library)],
        env=settings(JAVA_HOME=str(jdk), CUDA_VISIBLE_DEVICES=""), cwd=tmp_path,
        text=True, capture_output=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "JVM signal chaining passed" in result.stdout
