"""The Fixer's edit menu (docs/FIXER_PLAN.md section 3, milestone F1).

Given one QA report and the current level, list the entities worth looking at and, for
each, the candidate edits, all derived from the LEVEL ALONE (never the answer key):

- linedefs within `radius` of the report;
- trigger lines anywhere whose target tag points at a sector near the report, or at a tag
  no sector has (a dangling tag is visible in the level itself). A dead switch is reported
  at the door it fails to open, while the broken line is the switch, often rooms away;
- the sectors those nearby lines touch;
- things within `radius`, plus the teleport destinations of nearby teleport lines.

Candidate edits per entity:

| entity            | candidates                                                        |
|-------------------|-------------------------------------------------------------------|
| two-sided linedef | toggle `blocking`; copy the special of a nearby line of the same sector |
| tagged trigger    | set its target tag to each sector tag in the level (nearest first) |
| locked door line  | set its lock to each key the level contains                       |
| sector            | set its floor to each neighbouring sector's floor                 |
| typed thing       | another type of the same class (health, armour, ammo, keys)        |
| teleport dest     | the centre of its sector, facing 0/90/180/270                     |

Entities come nearest first, at most `max_entities`, each with at most `max_edits`, so a
two-step lettered choice (entity, then edit) fits the LLM's option picker (FIXER_PLAN).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from qatester.fixer.edits import (
    THING_CLASSES,
    CopySpecial,
    Edit,
    EditError,
    MoveThing,
    SetArg,
    SetFlag,
    SetFloor,
    SetThingType,
    apply,
    thing_class,
)
from qatester.mapgen import udmf
from qatester.mapgen.geometry import (
    DOOR_LOCKED_RAISE,
    DOOR_OPEN,
    DOOR_RAISE,
    PLAT_DOWN_WAIT_UP_STAY,
    TELEPORT,
)

RADIUS_MU = 256.0
MAX_ENTITIES = 25
MAX_EDITS = 25
TAG_SPECIALS = frozenset({DOOR_OPEN, DOOR_RAISE, DOOR_LOCKED_RAISE, PLAT_DOWN_WAIT_UP_STAY})
SPECIAL_NAMES = {DOOR_OPEN: "Door_Open", DOOR_RAISE: "Door_Raise",
                 DOOR_LOCKED_RAISE: "Door_LockedRaise", PLAT_DOWN_WAIT_UP_STAY: "Plat_DownWaitUpStay",
                 TELEPORT: "Teleport", 243: "Exit_Normal"}  # fmt: skip
KEY_LOCKS = {13: 1, 5: 2, 6: 3}  # red, blue, yellow key card -> Door_LockedRaise lock
TELEPORT_DEST = udmf.THING_EDNUMS["teleport_dest"]


@dataclass(frozen=True)
class Report:
    """What QA hands the Fixer: one report from report.json, nothing else."""

    oracle: str
    kind: str
    position: tuple[float, float, float]
    evidence: str = ""


@dataclass(frozen=True)
class Entity:
    kind: Literal["linedef", "sector", "thing"]
    index: int
    distance: float
    summary: str
    edits: tuple[Edit, ...]


@dataclass(frozen=True)
class Menu:
    report: Report
    entities: tuple[Entity, ...]

    def all_edits(self) -> list[Edit]:
        return [e for ent in self.entities for e in ent.edits]


class _Level:
    """Geometry derived from the TextMap: line midpoints and sectors, sector bounding
    boxes and neighbours, tags. Read-only."""

    def __init__(self, tm: udmf.TextMap) -> None:
        self.tm = tm
        self.ends = [(tm.vertices[ld.v1], tm.vertices[ld.v2]) for ld in tm.linedefs]
        self.sides = [(tm.sidedefs[ld.sidefront].sector,
                       tm.sidedefs[ld.sideback].sector if ld.sideback is not None else None)
                      for ld in tm.linedefs]  # fmt: skip
        self.lines_of: dict[int, list[int]] = {}
        self.neighbours: dict[int, set[int]] = {}
        for i, (f, b) in enumerate(self.sides):
            for s in (f, b):
                if s is not None:
                    self.lines_of.setdefault(s, []).append(i)
            if b is not None and b != f:
                self.neighbours.setdefault(f, set()).add(b)
                self.neighbours.setdefault(b, set()).add(f)
        self.bbox: dict[int, tuple[float, float, float, float]] = {}
        for s, lines in self.lines_of.items():
            xs = [v.x for i in lines for v in self.ends[i]]
            ys = [v.y for i in lines for v in self.ends[i]]
            self.bbox[s] = (min(xs), min(ys), max(xs), max(ys))
        self.tagged: dict[int, list[int]] = {}
        for s, sec in enumerate(tm.sectors):
            if sec.id:
                self.tagged.setdefault(sec.id, []).append(s)

    def line_distance(self, i: int, x: float, y: float) -> float:
        (a, b) = self.ends[i]
        dx, dy = b.x - a.x, b.y - a.y
        t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((x - a.x) * dx + (y - a.y) * dy)
                                                   / (dx * dx + dy * dy)))  # fmt: skip
        return math.hypot(x - (a.x + t * dx), y - (a.y + t * dy))

    def sector_distance(self, s: int, x: float, y: float) -> float:
        x0, y0, x1, y1 = self.bbox[s]
        return math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))

    def contains(self, s: int, x: float, y: float) -> bool:
        """Point in sector by the crossing rule over all its lines (holes included)."""
        inside = False
        for i in self.lines_of.get(s, []):
            a, b = self.ends[i]
            if (a.y > y) != (b.y > y) and x < a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y):
                inside = not inside
        return inside

    def sector_at(self, x: float, y: float) -> int | None:
        hits = [s for s in self.lines_of if self.contains(s, x, y)]
        if not hits:
            return None
        return min(hits, key=lambda s: (self.bbox[s][2] - self.bbox[s][0])
                   * (self.bbox[s][3] - self.bbox[s][1]))  # fmt: skip


def _valid(tm: udmf.TextMap, e: Edit) -> bool:
    try:
        apply(tm, [e])
    except EditError:
        return False
    return True


def _line_summary(lv: _Level, i: int) -> str:
    ld = lv.tm.linedefs[i]
    f, b = lv.sides[i]
    secs = lv.tm.sectors
    where = (f"between sector {f} (floor {secs[f].heightfloor}) and sector {b} "
             f"(floor {secs[b].heightfloor})" if b is not None
             else f"one-sided wall of sector {f}")  # fmt: skip
    flags = ", ".join(sorted(ld.flags & {"blocking", "playeruse", "playercross"})) or "no flags"
    special = (f"; special {SPECIAL_NAMES.get(ld.special, ld.special)}{tuple(ld.args)}"
               if ld.special else "")  # fmt: skip
    return f"linedef {i}: {where}; {flags}{special}"


def build_menu(tm: udmf.TextMap, report: Report, radius: float = RADIUS_MU,
               max_entities: int = MAX_ENTITIES, max_edits: int = MAX_EDITS) -> Menu:  # fmt: skip
    lv = _Level(tm)
    x, y = report.position[0], report.position[1]
    near_lines = {i: d for i in range(len(tm.linedefs))
                  if (d := lv.line_distance(i, x, y)) <= radius}  # fmt: skip
    near_sectors = {s for i in near_lines for s in lv.sides[i] if s is not None}
    tag_order = sorted(lv.tagged, key=lambda t: min(lv.sector_distance(s, x, y)
                                                     for s in lv.tagged[t]))  # fmt: skip

    lines: dict[int, float] = dict(near_lines)
    for i, ld in enumerate(tm.linedefs):  # triggers aimed at nearby sectors, or at nothing
        if ld.special in TAG_SPECIALS and ld.args and ld.args[0]:
            targets = lv.tagged.get(ld.args[0], [])
            if not targets:
                lines[i] = 0.0  # a dangling tag: the line is visibly broken
            elif any(s in near_sectors for s in targets):
                lines[i] = min(lines.get(i, math.inf),
                               min(lv.sector_distance(s, x, y) for s in targets))  # fmt: skip
    keys = sorted({KEY_LOCKS[t.type] for t in tm.things if t.type in KEY_LOCKS})

    entities: list[Entity] = []
    for i, dist in lines.items():
        ld, (f, b) = tm.linedefs[i], lv.sides[i]
        edits: list[Edit] = []
        if ld.special in TAG_SPECIALS and ld.args:
            edits += [SetArg(i, 0, t) for t in tag_order if t != ld.args[0]]
        if ld.special == DOOR_LOCKED_RAISE:
            cur = ld.args[3] if len(ld.args) > 3 else 0
            edits += [SetArg(i, 3, k) for k in keys if k != cur]
        if b is not None:
            # Only a line DRAWN as a solid wall (a middle texture) that you can walk
            # through may be made blocking: that is what a walk-through wall looks like.
            # Doorways, openings and teleport-pad edges are drawn open, and a trigger
            # line stops working when blocked; making those blocking walls off part of
            # the level instead of repairing it (seen in the first loop batch: four
            # teleport-pad edges blocked to make a teleport loop "go away").
            sides = (tm.sidedefs[ld.sidefront], tm.sidedefs[ld.sideback])  # type: ignore[index]
            drawn_wall = any(sd.texturemiddle for sd in sides)
            if "blocking" in ld.flags or (drawn_wall and ld.special == 0):
                edits.append(SetFlag(i, "blocking", "blocking" not in ld.flags))
            if ld.special == 0:
                edits += [CopySpecial(j, i) for j in sorted(near_lines, key=near_lines.get)  # type: ignore[arg-type]
                          if j != i and tm.linedefs[j].special
                          and {f, b} & {s for s in lv.sides[j] if s is not None}]  # fmt: skip
        edits = [e for e in edits if _valid(tm, e)][:max_edits]
        if edits:
            entities.append(Entity("linedef", i, dist, _line_summary(lv, i), tuple(edits)))

    # A lift platform (the sector a Plat_ line's tag names) bridges a height difference on
    # purpose: neither it nor the rooms beside it get floor edits. Flattening a lift's
    # lower room "repairs" a one-way lift by making the lift pointless (seen in the first
    # loop batch); the repair is the missing trigger. Pits never border a lift.
    lifts = {s for ld in tm.linedefs if ld.special == PLAT_DOWN_WAIT_UP_STAY and ld.args
             for s in lv.tagged.get(ld.args[0], [])}  # fmt: skip
    no_floor_edit = lifts | {n for s in lifts for n in lv.neighbours.get(s, ())}
    for s in near_sectors - no_floor_edit:
        sec = tm.sectors[s]
        floors = sorted({tm.sectors[n].heightfloor for n in lv.neighbours.get(s, ())}
                        - {sec.heightfloor})  # fmt: skip
        floor_edits: list[Edit] = [e for h in floors if _valid(tm, e := SetFloor(s, h))]
        if floor_edits:
            summary = (f"sector {s}: floor {sec.heightfloor}, ceiling {sec.heightceiling}; "
                       f"neighbour floors {sorted(tm.sectors[n].heightfloor for n in lv.neighbours.get(s, ()))}")  # fmt: skip
            entities.append(Entity("sector", s, lv.sector_distance(s, x, y), summary,
                                   tuple(floor_edits[:max_edits])))  # fmt: skip

    near_tids = {tm.linedefs[i].args[0] for i in near_lines
                 if tm.linedefs[i].special == TELEPORT and tm.linedefs[i].args}  # fmt: skip
    for t, th in enumerate(tm.things):
        dist = math.hypot(th.x - x, th.y - y)
        if dist > radius and not (th.type == TELEPORT_DEST and th.id in near_tids):
            continue
        edits = []
        cls = thing_class(th.type)
        if cls is not None:
            edits += [SetThingType(t, k) for k in sorted(THING_CLASSES[cls]) if k != th.type]
        if th.type == TELEPORT_DEST:
            home = lv.sector_at(th.x, th.y)
            if home is not None:
                x0, y0, x1, y1 = lv.bbox[home]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                edits += [MoveThing(t, cx, cy, a) for a in (0, 90, 180, 270)
                          if (cx, cy, a) != (th.x, th.y, th.angle)]  # fmt: skip
        edits = [e for e in edits if _valid(tm, e)][:max_edits]
        if edits:
            what = f"thing {t}: type {th.type} at ({th.x:.0f}, {th.y:.0f}) facing {th.angle}"
            entities.append(Entity("thing", t, min(dist, radius), what, tuple(edits)))

    entities.sort(key=lambda e: (e.distance, e.kind, e.index))
    return Menu(report, tuple(entities[:max_entities]))
