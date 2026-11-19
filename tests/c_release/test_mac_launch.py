"""Offline tests for macOS/Linux Bash launcher, without starting Streamlit."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "Start Jerry-C.sh"
pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="requires POSIX Bash and executable stub",
)


def test_bash_syntax():
    check = subprocess.run(["bash", "-n", str(LAUNCHER)], check=False, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


def synthetic_root(tmp_path: Path) -> Path:
    root = tmp_path / "isolated source copy with spaces"
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "app").mkdir()
    shutil.copy2(LAUNCHER, root / LAUNCHER.name)
    python = root / ".venv" / "bin" / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$STUB_TRACE"\n'
        'if [[ "$1" == "-B" ]]; then exit "${STUB_SETUP_EXIT:-0}"; fi\n'
        'exit 0\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    return root


def invoke(root: Path, tmp_path: Path, env_updates=None):
    trace = tmp_path / "script_calls.txt"
    env = {**os.environ, "STUB_TRACE": str(trace)}
    env.pop("DESKCHECK_PORT", None)
    env.update(env_updates or {})
    outcome = subprocess.run(
        ["bash", str(root / LAUNCHER.name)], check=False, cwd=tmp_path,
        capture_output=True, text=True, env=env, timeout=8,
    )
    log = trace.read_text(encoding="utf-8").splitlines() if trace.exists() else []
    return outcome, log


def test_mac_launcher_checked_before_streamlit_and_keeps_same_port(tmp_path):
    root = synthetic_root(tmp_path)
    run, calls = invoke(root, tmp_path, {"DESKCHECK_PORT": "8517"})
    assert run.returncode == 0, run.stderr
    assert calls == [
        "-B scripts/check_c_setup.py --profile ui --port 8517",
        ("-m streamlit run app/streamlit_app.py --server.address=127.0.0.1 "
         "--server.port=8517 --server.maxUploadSize=32 --browser.gatherUsageStats=false"),
    ]


def test_mac_launcher_does_not_start_server_if_setup_fails(tmp_path):
    root = synthetic_root(tmp_path)
    run, calls = invoke(root, tmp_path, {"STUB_SETUP_EXIT": "23"})
    assert run.returncode == 23
    assert len(calls) == 1 and calls[0].startswith("-B scripts/check_c_setup.py")


@pytest.mark.parametrize("port", ["0", "65536", "a-number", "-3", "12.3", "999999"])
def test_mac_launcher_rejects_bad_port_before_probing(tmp_path, port):
    root = synthetic_root(tmp_path)
    run, calls = invoke(root, tmp_path, {"DESKCHECK_PORT": port})
    assert run.returncode == 2
    assert not calls


def test_mac_launcher_reports_missing_venv_without_install(tmp_path):
    root = synthetic_root(tmp_path)
    (root / ".venv" / "bin" / "python").unlink()
    run, calls = invoke(root, tmp_path)
    assert run.returncode == 1
    assert "Missing .venv/bin/python" in run.stderr
    assert not calls
