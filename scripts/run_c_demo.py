"""Create synthetic inputs and run the real C CLI into a new demo directory."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deskcheck.application.demo_pair import make_demo_pair
from deskcheck.application.demo_verifier import verify_demo


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic C demo; no trained model")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    target = args.output.absolute()
    if target.exists() or target.is_symlink():
        print(f"Refusing to overwrite demo: {target}", file=sys.stderr)
        return 2
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".c-demo-", dir=target.parent) as temp:
            staging = Path(temp) / "demo"
            staging.mkdir()
            before, after = make_demo_pair()
            (staging / "before.png").write_bytes(before)
            (staging / "after.png").write_bytes(after)
            command = [
                sys.executable,
                "-B",
                str(ROOT / "scripts" / "predict_pair.py"),
                "--config",
                str(ROOT / "configs" / "fake_app.json"),
                "--before",
                str(staging / "before.png"),
                "--after",
                str(staging / "after.png"),
                "--pair-id",
                "c-demo",
                "--output",
                str(staging / "bundle"),
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=60
            )
            if result.returncode:
                raise RuntimeError(result.stderr or result.stdout or "C CLI failed")
            records, changes = verify_demo(
                staging, before, after, ROOT / "configs" / "fake_app.json"
            )
            if target.exists() or target.is_symlink():
                raise FileExistsError("Demo target appeared during publication")
            staging.rename(target)
        print(
            f"Synthetic demo ready: {target}; {records} records, {changes} changes. "
            "No trained inference."
        )
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"Demo failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
