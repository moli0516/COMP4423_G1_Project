"""Unit tests for ImageFrame invariants (B's rule 5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deskcheck.domain.image_frame import ImageFrame


def _make(h: int = 4, w: int = 6) -> ImageFrame:
    return ImageFrame(array=np.zeros((h, w, 3), dtype=np.uint8), width=w, height=h)


def test_valid_frame_constructs() -> None:
    f = _make(4, 6)
    assert f.width == 6
    assert f.height == 4
    assert f.source_path is None


def test_rejects_non_ndarray() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        ImageFrame(array=[[[0, 0, 0]]], width=1, height=1)  # type: ignore[arg-type]


def test_rejects_non_uint8_dtype() -> None:
    with pytest.raises(ValueError, match="dtype must be uint8"):
        ImageFrame(array=np.zeros((4, 6, 3), dtype=np.float32), width=6, height=4)


def test_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match="shape must be"):
        ImageFrame(array=np.zeros((4, 6), dtype=np.uint8), width=6, height=4)


def test_rejects_wrong_channels() -> None:
    with pytest.raises(ValueError, match="shape must be"):
        ImageFrame(array=np.zeros((4, 6, 4), dtype=np.uint8), width=6, height=4)


def test_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="disagrees"):
        ImageFrame(array=np.zeros((4, 6, 3), dtype=np.uint8), width=99, height=4)


def test_rejects_nonpositive_dims() -> None:
    # shape matches (H, W) but one dimension is zero
    with pytest.raises(ValueError, match="disagrees|positive"):
        ImageFrame(array=np.zeros((0, 1, 3), dtype=np.uint8), width=1, height=0)


def test_rejects_zero_width_with_matching_shape() -> None:
    with pytest.raises(ValueError, match="disagrees|positive"):
        ImageFrame(array=np.zeros((1, 0, 3), dtype=np.uint8), width=0, height=1)


def test_copy_returns_independent_array() -> None:
    f = _make(4, 6)
    g = f.copy()
    assert g.array is not f.array
    g.array[0, 0, 0] = 255
    assert f.array[0, 0, 0] == 0


def test_source_path_optional_and_preserved() -> None:
    f = ImageFrame(
        array=np.zeros((4, 6, 3), dtype=np.uint8), width=6, height=4,
        source_path=Path("data/raw/pilot/p1_before.jpg"),
    )
    assert f.source_path is not None
    assert f.copy().source_path == f.source_path


def test_frozen_prevents_attribute_reassignment() -> None:
    import dataclasses
    f = _make(4, 6)
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.width = 99  # type: ignore[misc]
