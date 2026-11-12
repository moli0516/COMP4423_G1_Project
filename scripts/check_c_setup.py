"""Read-only setup diagnostics. Does not load models, install, or start a server.

Uses only stdlib until probing installed dependencies, so broken environments can
still receive a useful error. A validated checkpoint hash is not inference evidence.
"""

import argparse
import importlib
import importlib.metadata
import json
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "pyproject.toml",
    "requirements.txt",
    "configs/fake_app.json",
    "scripts/predict_pair.py",
    "scripts/predict_real_pair.py",
    "scripts/evaluate_real_pairs.py",
    "scripts/run_c_demo.py",
    "src/deskcheck/application/comparison_service.py",
)


def check_setup(root: Path, *, profile="ui", port=None, real_config=None) -> dict:
    checks = []
    install = subprocess.list2cmdline(
        [sys.executable, "-m", "pip", "install", "-e", str(root) + "[dev,ui]"]
    )

    def add(name, passed, detail, action=""):
        checks.append({"name": name, "passed": passed, "detail": str(detail), "action": action})

    add("python", sys.version_info >= (3, 11), f"{sys.executable} ({sys.version.split()[0]})",
        "Use Python >=3.11 in this project's .venv.")
    files = REQUIRED_FILES + (("app/streamlit_app.py",) if profile == "ui" else ())
    missing = [name for name in files if not (root / name).is_file()]
    add("source_files", not missing, ", ".join(missing) if missing else "Required entries present",
        "Restore the complete C source checkout/archive; preserve existing work.")
    dependencies = [("numpy", "numpy"), ("PIL", "Pillow")]
    if profile == "ui":
        dependencies.append(("streamlit", "streamlit"))
    requirements = {}
    try:
        # packaging is already supplied by the existing dev/UI dependency set.
        requirement_type = importlib.import_module("packaging.requirements").Requirement
        toml = importlib.import_module("tomllib")
        project = toml.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        declared = project["dependencies"]
        if profile == "ui":
            declared = declared + project["optional-dependencies"]["ui"]
        requirements = {r.name.casefold(): r for r in map(requirement_type, declared)}
        add("version_policy", True, "Selected core/UI constraints loaded from pyproject.toml")
    except (ImportError, OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        add("version_policy", False, f"{type(error).__name__}: {error}", install)
    for module_name, distribution in dependencies:
        try:
            importlib.import_module(module_name)
            version = importlib.metadata.version(distribution)
            requirement = requirements.get(distribution.casefold())
            supported = requirement is not None and version in requirement.specifier
            add(distribution, supported, f"{version}; requires {requirement or 'unverified policy'}",
                install)
        except (ImportError, OSError, ValueError, RuntimeError, AttributeError) as error:
            add(distribution, False, f"{type(error).__name__}: {error}", install)
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        add("pip_check", completed.returncode == 0,
            (completed.stdout + completed.stderr).strip(), install)
    except (OSError, subprocess.TimeoutExpired) as error:
        add("pip_check", False, error, install)
    try:
        config_type = importlib.import_module("deskcheck.application.app_config").AppConfig
        config = config_type.load(root / "configs/fake_app.json")
        add("fake_config", True, f"mode=fake config_sha256={config.config_sha256}")
    except (ImportError, OSError, ValueError, TypeError, AttributeError) as error:
        add("fake_config", False, f"{type(error).__name__}: {error}",
            "Check configs/fake_app.json; synthetic mode requires explicit valid settings.")
    if port is not None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                probe.bind(("127.0.0.1", port))
            add("ui_port", True, f"127.0.0.1:{port} available at check time")
        except OSError as error:
            add("ui_port", False, f"127.0.0.1:{port}: {error}",
                "An existing UI may already be running: inspect it first. Otherwise choose "
                "another port for check_c_setup.py --port and streamlit --server.port. "
                "This checker does not stop other services.")
    real_status = "NOT_REQUESTED"
    if real_config is not None:
        try:
            settings_type = importlib.import_module("deskcheck.application.real_settings").RealSettings
            settings = settings_type.load(real_config)
            add("real_artifacts", True,
                f"config_sha256={settings.config_sha256} "
                f"checkpoint_sha256={settings.checkpoint_sha256}; hash-validated only")
            real_status = "HASH_VALIDATED"
        except (ImportError, OSError, ValueError, TypeError, AttributeError) as error:
            add("real_artifacts", False, f"{type(error).__name__}: {error}",
                "Supply A's reviewed local checkpoint and matching real config. No fake fallback.")
            real_status = "INVALID"
    return {
        "status": "setup_ready" if all(c["passed"] for c in checks) else "blocked",
        "profile": profile,
        "root": str(root.resolve()),
        "checks": checks,
        "real_artifacts": real_status,
        "real_backend": "UNVERIFIED",
        "real_inference": "NOT_RUN",
        "warning": "Setup checks only. Synthetic readiness is not real-model acceptance.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cli", "ui"), default="ui")
    parser.add_argument("--port", type=int, help="Probe local UI port without starting a service")
    parser.add_argument("--real-config", type=Path, help="Validate artifact hashes without model load")
    parser.add_argument("--json", action="store_true", help="Emit one structured JSON report")
    args = parser.parse_args()
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("--port must be 1..65535")
    sys.path.insert(0, str(ROOT / "src"))
    report = check_setup(ROOT, profile=args.profile, port=args.port, real_config=args.real_config)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for check in report["checks"]:
            print(f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}: {check['detail']}")
            if not check["passed"] and check["action"]:
                print(f"  Action: {check['action']}")
        print(f"{report['status']}; {report['warning']}")
        print(f"Real artifacts: {report['real_artifacts']}; backend UNVERIFIED; inference NOT_RUN")
    return 0 if report["status"] == "setup_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
