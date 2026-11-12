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
            selected = {d["class_id"]: d for d in data.get(f"{side}_detections", [])}
            occupied = [r[f"{side}_box"] for r in data["records"] if r[f"{side}_box"]]
            for record in data["records"]:
                box = record[f"{side}_box"]
                if box is None:
                    continue
                class_id = record["class_id"]
                if class_id not in range(len(CLASS_NAMES)):
                    raise ValueError("Cannot render unsupported class")
                color = self.COLORS[record["status"]]
                draw.rectangle(box, outline=color, width=2)
                # Same basic-scope correspondence label as the UI's object table.
                label = f"object-{class_id}"
                lines = [label]
                if class_id in selected:
                    lines.append(f"conf {selected[class_id]['confidence']:.3f}")
                width = max(draw.textbbox((0, 0), text)[2] for text in lines) + 4
                height = len(lines) * 13 + 2
                x, y = self._label_position(box, width, height, image.size, occupied)
                occupied.append([x, y, x + width, y + height])
                draw.rectangle((x, y, min(image.width, x + width), y + height), fill=color)
                for index, text in enumerate(lines):
                    draw.text((x + 2, y + index * 13), text, fill="white")
        return images

    @staticmethod
    def _label_position(box, width, height, size, occupied):
        """Prefer nearby free space so tiny demo images retain visible objects."""
        candidates = [
            (box[0], box[1] - height), (box[0], box[3]),
            (box[2], box[1]), (box[0] - width, box[1]),
        ]
        candidates = [
            (max(0, min(x, size[0] - width)), max(0, min(y, size[1] - height)))
            for x, y in candidates
        ]

        def overlap(position):
            x, y = position
            return sum(
                max(0, min(x + width, other[2]) - max(x, other[0]))
                * max(0, min(y + height, other[3]) - max(y, other[1]))
                for other in occupied
            )

        return min(candidates, key=overlap)
