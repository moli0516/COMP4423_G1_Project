"""Proposed team status vocabulary."""

from enum import Enum


class ChangeStatus(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    MOVED = "moved"
    UNCHANGED = "unchanged"

    @property
    def indicates_change(self) -> bool:
        return self is not ChangeStatus.UNCHANGED
