from math import nan

import pytest

from deskcheck.domain.bounding_box import BoundingBox
from deskcheck.domain.change_status import ChangeStatus
from deskcheck.domain.detection import Detection


def test_box_center_and_bounds():
    box = BoundingBox(0, 10, 20, 30)
    box.validate(100, 100)
    assert box.center() == (10, 20)


@pytest.mark.parametrize("box", [BoundingBox(0, 0, 0, 1), BoundingBox(-1, 0, 1, 1)])
def test_invalid_box(box):
    with pytest.raises(ValueError):
        box.validate(100, 100)


def test_nonfinite_box():
    with pytest.raises(ValueError):
        BoundingBox(nan, 0, 10, 10).validate(100, 100)


def test_status_semantics():
    assert ChangeStatus.MOVED.indicates_change
    assert not ChangeStatus.UNCHANGED.indicates_change


def test_invalid_confidence():
    with pytest.raises(ValueError):
        Detection("d1", 0, 1.1, BoundingBox(0, 0, 1, 1))
