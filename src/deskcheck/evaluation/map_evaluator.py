"""Backend mAP wrapper with validated settings recording; separate from P/R/F1.

Owner decision (2026-11-26): mAP@0.5:0.95 and mAP@0.5 come from the backend
validator's own settings. They must not be forced to the application operating
point, and validation settings that cannot be observed are recorded as
unobserved rather than assumed.

This module never imports Ultralytics or torch. The validation callable is
injected, so tests use a stub result object; the E2-specific 640/960 script is
left untouched because it binds a fixed inventory, two arms, receipts and
latency measurements that E1 does not need.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

VALIDATION_SETTING_KEYS = (
    "data",
    "split",
    "imgsz",
    "batch",
    "device",
    "workers",
    "conf",
    "iou",
    "max_det",
    "rect",
    "plots",
    "save_json",
    "project",
    "name",
)

UNOBSERVED = {"state": "unobserved"}


def _observed(value: Any) -> dict[str, Any]:
    return {"state": "observed", "value": value}


def _as_metric(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise RuntimeError(f"Backend validator returned a non-numeric {name}: {value!r}")
    if not 0 <= float(value) <= 1:
        raise RuntimeError(f"Backend validator returned {name} outside [0, 1]: {value!r}")
    return float(value)


def _per_class_ap(results: object, class_names: tuple[str, ...]) -> dict:
    class_result = getattr(results, "class_result", None)
    support = getattr(results, "nt_per_class", None)
    if not callable(class_result) or support is None:
        return {
            "state": "unavailable",
            "reason": "Validator result did not expose per-class results",
        }
    try:
        supports = list(support)
    except TypeError:
        return {
            "state": "unavailable",
            "reason": "Validator per-class support is not iterable",
        }
    if len(supports) < len(class_names):
        return {
            "state": "unavailable",
            "reason": "Validator per-class support does not cover every canonical class",
        }
    per_class = {}
    try:
        for index, name in enumerate(class_names):
            row = class_result(index)
            # Ultralytics class_result order: (P, R, AP50, AP50-95).
            per_class[name] = {
                "support": int(supports[index]),
                "AP50": float(row[2]),
                "AP50_95": float(row[3]),
            }
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        return {
            "state": "unavailable",
            "reason": f"Validator per-class results are unreadable: {exc}",
        }
    return per_class


def evaluate_backend_map(
    run_validation: Any,
    *,
    validation_settings: dict[str, Any],
    class_names: tuple[str, ...] | list[str],
    backend_version: str | None = None,
) -> dict:
    """Run the injected validator and record mAP plus the settings actually used.

    Only explicitly supplied settings are forwarded. Every other setting is
    reported as unobserved: it is the backend's own runtime default and is not
    assumed or described as observed here.
    """
    if not callable(run_validation):
        raise TypeError("run_validation must be callable")
    if not isinstance(validation_settings, dict):
        raise TypeError("validation_settings must be a dictionary")
    unknown = set(validation_settings) - set(VALIDATION_SETTING_KEYS)
    if unknown:
        raise ValueError(f"Unknown validation settings {sorted(unknown)}")
    names = tuple(class_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("class_names must be nonempty strings")

    observed: dict[str, Any] = {}
    for key, value in validation_settings.items():
        if value is None:
            raise ValueError(f"validation setting {key!r} must be an explicit value or omitted")
        if isinstance(value, float) and not isfinite(value):
            raise ValueError(f"validation setting {key!r} must be finite")
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"validation setting {key!r} must be a JSON scalar")
        observed[key] = value

    try:
        results = run_validation(**observed)
    except Exception as exc:
        raise RuntimeError("Backend validation failed before metrics could be recorded") from exc

    box = getattr(results, "box", None)
    if box is None:
        raise RuntimeError("Backend validation result does not expose box metrics")
    map50_95 = _as_metric("mAP50-95", getattr(box, "map", None))
    map50 = _as_metric("mAP50", getattr(box, "map50", None))

    return {
        "protocol_version": "backend-map-wrapper-v1",
        "metric_scope": (
            "backend validator mAP; uses validator settings, not the application operating point"
        ),
        "metrics": {"mAP50_95": map50_95, "mAP50": map50},
        "per_class_ap": _per_class_ap(results, names),
        "validation_settings": {
            "observed": observed,
            "unobserved": sorted(set(VALIDATION_SETTING_KEYS) - set(observed)),
            "note": (
                "Only settings listed as observed were passed explicitly. Unobserved "
                "settings were left to the backend and are not assumed or recorded."
            ),
        },
        "backend": {
            "version": _observed(backend_version)
            if isinstance(backend_version, str) and backend_version.strip()
            else UNOBSERVED,
        },
        "matching_note": (
            "mAP is computed by the backend validator. Its matching differs from the "
            "custom precision/recall/F1 evaluator, so the counts are not interchangeable."
        ),
    }
