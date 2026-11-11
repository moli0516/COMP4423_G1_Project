"""Render copies in canonical coordinates, including missing-side/empty cases."""

from typing import ClassVar

from PIL import Image, ImageDraw

from deskcheck.application.app_config import CLASS_NAMES
from deskcheck.application.result_serializer import ResultSerializer


class ResultRenderer:
    COLORS: ClassVar[dict[str, str]] = {
        "added": "#008A52",
        "removed": "#C43832",
        "moved": "#9A6300",
        "unchanged": "#2868A0",
    }

    def render(self, before, after, result: dict) -> tuple[Image.Image, Image.Image]:
        data = ResultSerializer().snapshot(result)
        if any([frame.width, frame.height] != data["image_size"] for frame in (before, after)):
            raise ValueError("Renderer frame dimensions disagree with result")
        images = tuple(Image.fromarray(frame.array.copy()) for frame in (before, after))
        for side, image in zip(("before", "after"), images):
            draw = ImageDraw.Draw(image)
            for record in data["records"]:
                box = record[f"{side}_box"]
                if box is None:
                    continue
                class_id = record["class_id"]
                if class_id not in range(len(CLASS_NAMES)):
                    raise ValueError("Cannot render unsupported class")
                color = self.COLORS[record["status"]]
                draw.rectangle(box, outline=color, width=2)
                label = f"{CLASS_NAMES[class_id]} {record['status']}"
                draw.text((box[0], max(0, box[1] - 12)), label, fill=color)
        return images
