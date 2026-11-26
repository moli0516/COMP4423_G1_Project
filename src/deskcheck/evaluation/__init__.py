"""Owner B with A: event/no-change scoring, run evidence and comparison reports.

Detector mAP is only wrapped for recording from an injected backend validator;
this package does not compute detector AP. Its precision/recall/F1 path is an
application-pipeline diagnostic over post-B-filter detections.
"""
