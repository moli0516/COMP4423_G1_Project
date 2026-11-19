"""Run an existing local Ultralytics .pt checkpoint on a real image pair.

Never downloads weights, never substitutes fake detections, and refuses to overwrite.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deskcheck.application.comparison_service import ComparisonService
from deskcheck.application.image_pair_loader import ImagePairLoader
from deskcheck.application.real_settings import RealSettings
from deskcheck.application.result_renderer import ResultRenderer
from deskcheck.application.result_serializer import ResultSerializer
from deskcheck.detection.ultralytics_detector import UltralyticsDetector


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real YOLO inference using supplied local weights; no auto-download"
    )
    for name in ("config", "before", "after", "pair-id", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    target = Path(args.output).absolute()
    if target.exists() or target.is_symlink():
        print(f"Refusing to overwrite output: {target}", file=sys.stderr)
        return 2
    staging = None
    try:
        settings = RealSettings.load(args.config)
        detector = UltralyticsDetector(settings)
        service = ComparisonService(detector=detector, config=settings)
        frames = ImagePairLoader().load(args.before, args.after, args.pair_id)
        result = service.compare_frames(*frames, pair_id=args.pair_id)
        images = ResultRenderer().render(*frames, result)
        serializer = ResultSerializer()
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        (staging / "result.json").write_bytes(serializer.to_json(result).encode("utf-8"))
        (staging / "result.csv").write_bytes(serializer.to_csv(result).encode("utf-8"))
        for side, picture in zip(("before", "after"), images):
            picture.save(staging / f"annotated_{side}.png", format="PNG")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Another process created output while inference ran")
        staging.rename(target)
        print(
            f"mode=real checkpoint={settings.checkpoint_sha256} "
            f"records={len(result['records'])} "
            f"changes={sum(r['indicates_change'] for r in result['records'])} "
            f"output={target}"
        )
        return 0
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        print(f"Real inference failed without publishing output: {error}", file=sys.stderr)
        return 1
    finally:
        # Remove only this call's temporary staging directory; never the requested target.
        if (
            staging is not None
            and staging.exists()
            and staging.parent.resolve() == target.parent.resolve()
            and staging.name.startswith(f".{target.name}-")
        ):
            shutil.rmtree(staging)


if __name__ == "__main__":
    raise SystemExit(main())
