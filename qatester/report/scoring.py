"""Scoring reported bugs against the answer key (PLAN: Evaluation protocol; D25).

A reported bug counts as a real defect when its oracle family is the defect's expected
one and it lies within 128 mu of the defect's recorded position, or inside one of the
rectangles the defect occupies or cuts off. Used by the evaluation (M9) and by the
demo's "planted vs found" view. Reads the defect list, so it runs only after testing,
never inside an agent or an oracle.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

MATCH_RADIUS_MU = 128.0
# A family that also names a defect correctly (D44): a player who falls into a trap pit
# and stays there until time runs out is stranded, which the softlock family reports.
ALSO_ACCEPTED = {"geometry_trap": {"softlock"}}


def matches(oracle: str, x: float, y: float, defect: Mapping[str, Any]) -> bool:
    families = {defect["expected_oracle"]} | ALSO_ACCEPTED.get(defect["kind"], set())
    if oracle not in families:
        return False
    near = math.hypot(x - defect["position"][0], y - defect["position"][1]) <= MATCH_RADIUS_MU
    inside = any(r[0] <= x <= r[2] and r[1] <= y <= r[3] for r in defect["region"])
    return near or inside


def score(reported: Sequence[Mapping[str, Any]],
          defects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:  # fmt: skip
    """`reported`: report findings (oracle + position). Returns, per defect, the
    reports that match it; the reports that match none (false alarms); precision,
    recall and F1."""
    found: dict[str, list[str]] = {d["id"]: [] for d in defects}
    false_alarms = []
    for r in reported:
        hit = [d["id"] for d in defects
               if matches(r["oracle"], r["position"][0], r["position"][1], d)]  # fmt: skip
        for i in hit:
            found[i].append(r.get("id", ""))
        if not hit:
            false_alarms.append(r.get("id", ""))
    true_reports = len(reported) - len(false_alarms)
    precision = true_reports / len(reported) if reported else None
    recall = sum(bool(v) for v in found.values()) / len(defects) if defects else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else (0.0 if reported and defects else None))  # fmt: skip
    return {"found": found, "missed": [i for i, v in found.items() if not v],
            "false_alarms": false_alarms, "precision": precision, "recall": recall, "f1": f1}
