from __future__ import annotations

from collections.abc import Iterable

from .config import CANONICAL_CLASSES
from .inference import Detection


def summarize(detections: Iterable[Detection]) -> dict:
    counts = {name: 0 for name in CANONICAL_CLASSES}
    for det in detections:
        if det.class_name in counts:
            counts[det.class_name] += 1

    no_helmet = counts["no_helmet"]
    people = counts["person"]
    violations = {"no_helmet": no_helmet}

    if no_helmet > 0:
        status = "VIOLATION"
        messages = [f"{no_helmet} explicit no-helmet violation(s) detected."]
    elif people > 0:
        status = "NO_EXPLICIT_VIOLATION"
        messages = [
            "No explicit no-helmet label was detected. This does not prove full PPE compliance."
        ]
    else:
        status = "NO_PERSON"
        messages = ["No person was detected in this frame."]

    return {
        "counts": counts,
        "violations": violations,
        "status": status,
        "messages": messages,
    }
