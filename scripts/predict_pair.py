"""Run the explicit fake-only C path; publish a fresh four-file bundle."""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deskcheck.application.app_config import AppConfig
from deskcheck.application.comparison_service import ComparisonService
from deskcheck.application.image_pair_loader import ImagePairLoader
from deskcheck.application.result_renderer import ResultRenderer
from deskcheck.application.result_serializer import ResultSerializer
from deskcheck.detection.fake_color_detector import FakeColorDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic-only C integration; no trained model")
    for name in ("config", "before", "after", "pair-id", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    target = Path(args.output).absolute()
    if target.exists() or target.is_symlink():
        print(f"Refusing to overwrite output: {target}", file=sys.stderr)
        return 2
    staging = None
    try:
        config = AppConfig.load(args.config)
        service = ComparisonService(detector=FakeColorDetector(), config=config)
        before, after = ImagePairLoader().load(args.before, args.after, args.pair_id)
        result = service.compare_frames(before, after, pair_id=args.pair_id)
        images = ResultRenderer().render(before, after, result)
        serializer = ResultSerializer()
        payloads = {
            "result.json": serializer.to_json(result),
            "result.csv": serializer.to_csv(result),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        for name, text in payloads.items():
            (staging / name).write_bytes(text.encode("utf-8"))
        for side, image in zip(("before", "after"), images):
            image.save(staging / f"annotated_{side}.png", format="PNG")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Output appeared during publication")
        staging.rename(target)
        print(
            f"mode=fake records={len(result['records'])} "
            f"changes={sum(r['indicates_change'] for r in result['records'])} output={target}"
        )
        return 0
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        print(f"Comparison failed: {error}", file=sys.stderr)
        return 1
    finally:
        # Delete only our own unpublished staging directory, never the requested target.
        if (
            staging is not None
            and staging.exists()
            and staging.parent.resolve() == target.parent.resolve()
            and staging.name.startswith(f".{target.name}-")
        ):
            shutil.rmtree(staging)


if __name__ == "__main__":
    raise SystemExit(main())
