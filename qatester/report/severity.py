"""Step 3: severity = w_o * oracle_weight + w_b * blocking_factor + w_f * frequency_factor
(PLAN: From findings to a usable bug report). Uses the intended level only."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from typing import Any

from qatester.mapgen.groundtruth import GroundTruth
from qatester.report.cluster import Cluster

ORACLE_WEIGHT = {"crash": 1.0, "softlock": 0.9, "out_of_bounds": 0.8, "invariant": 0.7,
                 "unreachable": 0.6, "stuck": 0.5}  # fmt: skip
DEFAULT_WEIGHTS = {"oracle": 0.5, "blocking": 0.3, "frequency": 0.2}


def room_of(truth: GroundTruth, x: float, y: float) -> str:
    """The room containing the point, else the nearest one (for points outside)."""
    inside = truth.room_at(x, y)
    if inside is not None:
        return inside

    def dist(r: dict[str, Any]) -> float:
        x0, y0, x1, y1 = r["rect"]
        return math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))

    return str(min(truth.data["rooms"], key=dist)["id"])


def critical_rooms(truth: GroundTruth) -> set[str]:
    """Rooms on every route from spawn to the goal room (endpoints included), in the
    intended portal graph. Rooms sharing a sector (union_with) are one node."""
    d = truth.data
    sector = {r["id"]: r["sector"] for r in d["rooms"]}
    goal = d["goal"]["room"]
    if goal is None:
        return set()
    start, target = sector[d["spawn"]["room"]], sector[goal]
    edges = [(sector[e["from"]], sector[e["to"]]) for e in d["portal_graph"]]

    def reachable(without: int | None) -> bool:
        if without in (start, target):
            return False
        seen, todo = {start}, deque([start])
        while todo:
            s = todo.popleft()
            if s == target:
                return True
            for a, b in edges:
                if a == s and b != without and b not in seen:
                    seen.add(b)
                    todo.append(b)
        return False

    if not reachable(None):
        return set()
    nodes = set(sector.values())
    cut = {n for n in nodes if not reachable(n)}
    return {r for r, s in sector.items() if s in cut}


def severity(c: Cluster, room: str, critical: set[str],
             weights: Mapping[str, float] | None = None) -> float:  # fmt: skip
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    blocking = 1.0 if room in critical else 0.3
    frequency = min(1.0, c.size / 10)
    return round(w["oracle"] * ORACLE_WEIGHT[c.oracle] + w["blocking"] * blocking
                 + w["frequency"] * frequency, 4)  # fmt: skip
