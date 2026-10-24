"""Ground truth JSON: the oracles' input and the evaluation's answer key (PLAN: Derived
outputs; Family 1-6).

Everything here describes the *intended* level, compiled from the spec without defects,
plus the list of injected defects with world positions. The buggy geometry itself never
appears: oracles must observe it through the engine and compare against intent.

qatester/agents/** MUST NOT import this module (tests/test_no_groundtruth_leak.py).
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from qatester.mapgen.geometry import PLAT_DOWN_WAIT_UP_STAY, CompiledMap, DefectRecord, Rect
from qatester.mapgen.spec import MapSpec, shared_wall

SCHEMA_VERSION = 1
Variant = Literal["buggy", "fixed"]


def _rect(r: Rect) -> list[int]:
    return [r.x0, r.y0, r.x1, r.y1]


def _portal_area(cm: CompiledMap, portal_id: str) -> list[int]:
    for prefix in ("door:", "lift:", "pad:"):
        if f"{prefix}{portal_id}" in cm.sectors:
            return _rect(cm.region_of(f"{prefix}{portal_id}"))
    p = cm.spec.portal(portal_id)  # open portal: the opening owned by room a
    wall = shared_wall(cm.spec.room(p.a), cm.spec.room(p.b))
    assert wall is not None
    centre = (wall.lo + wall.hi) // 2
    if p.at is not None:
        centre = p.at[1] if wall.axis == "x" else p.at[0]
    lo, hi = centre - p.width // 2, centre - p.width // 2 + p.width
    if wall.axis == "x":
        return [wall.at - 8, lo, wall.at + 8, hi]
    return [lo, wall.at - 8, hi, wall.at + 8]


def _walkable(intended: CompiledMap) -> dict[str, Any]:
    spec = intended.spec
    size = spec.cell_size
    x0 = min(r.x0 for r in spec.rooms)
    y0 = min(r.y0 for r in spec.rooms)
    x1 = max(r.x1 for r in spec.rooms)
    y1 = max(r.y1 for r in spec.rooms)
    width = -(-(x1 - x0) // size)
    height = -(-(y1 - y0) // size)
    rows = []
    for j in range(height):
        cy = y0 + (j + 0.5) * size
        rows.append("".join(
            "1" if intended.owner_at(x0 + (i + 0.5) * size, cy) is not None else "0"
            for i in range(width)
        ))  # fmt: skip
    return {"origin": [x0, y0], "cell_size": size, "width": width, "height": height,
            "rows": rows}  # fmt: skip


def _sectors(intended: CompiledMap) -> list[dict[str, Any]]:
    keys = list(intended.sectors)
    neighbours: dict[str, set[str]] = {k: set() for k in keys}
    lift_bounded: set[str] = set()
    for line in intended.lines:
        if line.back is not None:
            neighbours[line.front].add(line.back)
            neighbours[line.back].add(line.front)
        if line.special is not None and line.special.special == PLAT_DOWN_WAIT_UP_STAY:
            lift_bounded.update(k for k in (line.front, line.back) if k is not None)
    return [
        {
            "index": i, "key": s.key, "kind": s.kind, "floor": s.floor, "ceiling": s.ceiling,
            "rooms": list(s.rooms), "portal": s.portal, "tag": s.tag,
            "neighbours": sorted(keys.index(n) for n in neighbours[s.key]),
            "lift_trigger": s.key in lift_bounded,
        }
        for i, s in enumerate(intended.sectors.values())
    ]  # fmt: skip


def _defect(rec: DefectRecord) -> dict[str, Any]:
    return {
        "id": rec.id, "kind": rec.kind, "target": rec.target,
        "expected_oracle": rec.expected_oracle, "severity": rec.severity,
        "params": rec.params, "notes": rec.notes.strip(),
        "position": list(rec.position), "region": [_rect(r) for r in rec.region],
        "affected_rooms": list(rec.affected_rooms), "change": rec.change,
    }  # fmt: skip


def ground_truth(spec: MapSpec, intended: CompiledMap, built: CompiledMap,
                 variant: Variant) -> dict[str, Any]:  # fmt: skip
    """`intended` is compiled without defects; `built` is what went into the WAD."""
    xs, ys, owner = intended.owner_grid()
    keys = list(intended.sectors)
    switches = {t.opens: t.id for t in spec.things if t.type == "switch"}
    edges = []
    for p in spec.portals:
        ends = [(p.a, p.b)] if p.kind == "teleport" else [(p.a, p.b), (p.b, p.a)]
        edges += [{"from": a, "to": b, "via": p.id, "key": p.requires,
                   "switch": switches.get(p.id)} for a, b in ends]  # fmt: skip
    spawn = next(t for t in spec.things if t.type == "player_start")
    goal_room = spec.thing(spec.goal.target).room if spec.goal.target else None
    return {
        "schema_version": SCHEMA_VERSION,
        "map": spec.map_lump,
        "name": spec.name,
        "slot": spec.slot,
        "variant": variant,
        "cell_size": spec.cell_size,
        "spawn": {"room": spawn.room, "pos": list(spawn.pos)},
        "rooms": [
            {"id": r.id, "rect": [r.x0, r.y0, r.x1, r.y1], "floor": r.floor,
             "ceiling": r.ceiling, "sector": keys.index(intended.room_sector(r.id))}
            for r in spec.rooms
        ],
        "walkable": _walkable(intended),
        "walls": [[*line.v1, *line.v2] for line in intended.lines if line.back is None],
        "sectors": _sectors(intended),
        "sector_grid": {
            "xs": xs, "ys": ys,
            "owner": [[keys.index(k) if k is not None else -1 for k in col] for col in owner],
        },
        "portals": [
            {"id": p.id, "a": p.a, "b": p.b, "kind": p.kind, "requires": p.requires,
             "switch": switches.get(p.id), "area": _portal_area(intended, p.id),
             "dest": list(intended.teleport_dests[p.id]) if p.kind == "teleport" else None}
            for p in spec.portals
        ],
        # Wall switches and exits as built in the intended level: thing id -> segment.
        "switch_lines": {
            m.source: [m.lo, m.at, m.hi, m.at] if m.axis == "y" else [m.at, m.lo, m.at, m.hi]
            for m in intended.marks
        },
        "portal_graph": edges,
        "things": [
            {"id": t.id, "type": t.type, "room": t.room, "pos": list(t.pos),
             "interactive": t.interactive}
            for t in spec.things
        ],
        "monster_count": sum(t.type.startswith("monster_") for t in spec.things),
        "goal": {**spec.goal.model_dump(), "room": goal_room},
        "defects": [_defect(r) for r in built.defects],
    }


def write_ground_truth(path: Path, gt: dict[str, Any]) -> None:
    path.write_text(json.dumps(gt, indent=1) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class GroundTruth:
    """Read-side view used by the oracles (M4) and the evaluation (M9)."""

    data: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> GroundTruth:
        return cls(json.loads(path.read_text(encoding="utf-8")))

    @property
    def defects(self) -> list[dict[str, Any]]:
        if "defects" not in self.data:
            raise PermissionError("this is an intent-only view: oracles may not read defects")
        return list(self.data["defects"])

    def intent_only(self) -> GroundTruth:
        """The view oracles get: everything except the defect list (the answer key)."""
        return GroundTruth({k: v for k, v in self.data.items() if k != "defects"})

    def sector_at(self, x: float, y: float) -> int:
        """Index of the intended sector containing (x, y); -1 outside the level."""
        grid = self.data["sector_grid"]
        xs, ys = grid["xs"], grid["ys"]
        i = bisect.bisect_right(xs, x) - 1
        j = bisect.bisect_right(ys, y) - 1
        if not (0 <= i < len(xs) - 1 and 0 <= j < len(ys) - 1):
            return -1
        return int(grid["owner"][i][j])

    def room_at(self, x: float, y: float) -> str | None:
        for r in self.data["rooms"]:
            x0, y0, x1, y1 = r["rect"]
            if x0 <= x <= x1 and y0 <= y <= y1:
                return str(r["id"])
        return None

    def walkable(self, ix: int, iy: int) -> bool:
        w = self.data["walkable"]
        return bool(0 <= ix < w["width"] and 0 <= iy < w["height"] and w["rows"][iy][ix] == "1")

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        w = self.data["walkable"]
        ox, oy = w["origin"]
        return int((x - ox) // w["cell_size"]), int((y - oy) // w["cell_size"])
