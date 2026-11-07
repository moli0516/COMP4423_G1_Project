"""Owner B: association contract after filtering to one detection per class."""

from deskcheck.domain.detection import Detection


class ObjectMatcher:
    def match(
        self, before: list[Detection], after: list[Detection]
    ) -> list[tuple[Detection | None, Detection | None]]:
        raise NotImplementedError("Match by class, including unmatched sides; no distance gate")
