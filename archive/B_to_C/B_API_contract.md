## B API 契約

### Imports

```
from deskcheck.changes.change_analyzer import ChangeAnalyzer
from deskcheck.detection.detection_filter import DetectionFilter
from deskcheck.matching.object_matcher import ObjectMatcher
```

### Initialization

```
detection_filter = DetectionFilter(
    confidence_threshold=confidence_threshold,
)

matcher = ObjectMatcher()

analyzer = ChangeAnalyzer(
    movement_threshold=movement_threshold,
)
```

### Order of calling

```
filtered_before = detection_filter.apply(before_detections)
filtered_after = detection_filter.apply(after_detections)

matches = matcher.match(filtered_before, filtered_after)

records = analyzer.analyze(
    matches,
    image_size=(width, height),
)
```

### Rules

* First filter，then match。
* The order of`image_size` is `(width, height)`。
* The frame of Detections use canonical image coords。
* The **Matcher** pairs items strictly by class, without adding any extra distance or IoU association gates.
* The **Analyzer** does not perform confidence filtering.
* **Backend NMS** is handled by the detector adapter.
* **Component C** does not re-implement Component B's matching or status logic.
