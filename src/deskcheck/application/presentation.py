"""Presentation-only helpers; never reclassify B's results or mutate exports."""

from math import isfinite

from PIL import Image

from deskcheck.application.app_config import CLASS_NAMES
from deskcheck.application.result_serializer import ResultSerializer


def comparison_rows(result: dict, changes_only: bool = False) -> list[dict]:
    rows = []
    data = ResultSerializer().snapshot(result)
    scores = {
        side: {d["class_id"]: d["confidence"] for d in data.get(f"{side}_detections", [])}
        for side in ("before", "after")
    }
    for record in data["records"]:
        if changes_only and not record["indicates_change"]:
            continue
        class_id = record["class_id"]
        if class_id not in range(len(CLASS_NAMES)):
            raise ValueError("Unsupported canonical class")
        rows.append(
            {
                "Object ID": f"object-{class_id}",
                "Object": CLASS_NAMES[class_id],
                "Status": record["status"],
                "Before detection": record["before_detection_id"] or "absent",
                "After detection": record["after_detection_id"] or "absent",
                "Before confidence": scores["before"].get(class_id),
                "After confidence": scores["after"].get(class_id),
                "Normalized distance": record["distance_norm"],
                "Movement threshold": record["movement_threshold"],
            }
        )
    return rows


def comparison_wipe(before: Image.Image, after: Image.Image, percent: float) -> Image.Image:
    """Show before on the left and after on the right, in original coordinates."""
    if type(percent) not in (int, float) or not isfinite(percent) or not 0 <= percent <= 100:
        raise ValueError("Wipe percentage must be finite within [0, 100]")
    if before.size != after.size or before.mode != after.mode:
        raise ValueError("Wipe images must have identical dimensions and modes")
    split = round(before.width * percent / 100)
    result = after.copy()
    if split:
        result.paste(before.crop((0, 0, split, before.height)), (0, 0))
    return result
