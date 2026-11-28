"""Shared evidence labels written by the evaluation runner and enforced by readers.

The runner records these values in ``run_metadata.json``; evaluators and
comparators read them to decide what the saved artifacts contain. A single
constant keeps the writer and the reader from drifting apart, which would make
a valid run look unsupported.
"""

IMAGE_DETECTION_EVIDENCE_LABEL = "post_B_filter per-image detections"
