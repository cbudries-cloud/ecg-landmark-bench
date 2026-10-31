"""Strict validation of model responses.

A response is either VALID (a list of finite timestamps in seconds within the
recording) or INVALID with a recorded reason. Invalid responses are never
repaired, and never with reference labels; they are scored as zero predictions.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

KEY = "qrs_peaks_s"
_FENCE = re.compile(r"^```(?:json)?\s*\n(.*)\n```$", re.DOTALL)


@dataclass
class ParsedResponse:
    valid: bool
    peaks_s: list[float] = field(default_factory=list)
    reason: str | None = None          # why invalid
    notes: list[str] = field(default_factory=list)  # non-fatal observations (e.g. code fence)


def parse_response(text: str | None, duration_s: float) -> ParsedResponse:
    """Parse `{"qrs_peaks_s": [...]}`.

    Accepted: the JSON object alone, optionally wrapped in one ```json fence.
    Rejected (invalid): anything else, missing/extra keys, non-list, non-numeric
    or boolean items, NaN/inf, or any timestamp outside [0, duration_s], which
    almost always means the wrong unit (e.g. milliseconds or samples).
    Duplicates and empty lists are valid; duplicates are kept as separate
    predictions (an extra copy can only become a false positive).
    """
    if text is None or not text.strip():
        return ParsedResponse(False, reason="empty_response")
    body = text.strip()
    notes = []
    m = _FENCE.match(body)
    if m:
        body, notes = m.group(1).strip(), ["code_fence_stripped"]
    try:
        obj = json.loads(body, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    except (json.JSONDecodeError, ValueError):
        return ParsedResponse(False, reason="malformed_json", notes=notes)
    if not isinstance(obj, dict) or set(obj) != {KEY}:
        return ParsedResponse(False, reason="wrong_schema", notes=notes)
    values = obj[KEY]
    if not isinstance(values, list):
        return ParsedResponse(False, reason="not_a_list", notes=notes)
    peaks = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return ParsedResponse(False, reason="non_numeric_value", notes=notes)
        if not math.isfinite(v):
            return ParsedResponse(False, reason="non_finite_value", notes=notes)
        if v < 0 or v > duration_s:
            return ParsedResponse(False, reason="outside_recording_likely_wrong_unit", notes=notes)
        peaks.append(float(v))
    if len(peaks) != len(set(peaks)):
        notes.append("duplicate_timestamps")
    return ParsedResponse(True, sorted(peaks), notes=notes)
