"""MapSpec Pydantic models and loader validation (PLAN: Environment, part A).

Field names are normative. Two extensions, recorded in docs/decisions.md D13:
`Thing.type == "switch"` with `Thing.opens` (a remote door switch, needed by MAP04),
and `key_yellow`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PortalKind = Literal["open", "door", "locked_door", "lift", "teleport"]
ThingType = Literal[
    "player_start", "key_red", "key_blue", "key_yellow", "health", "armour", "ammo",
    "monster_imp", "exit_switch", "switch", "marker",
]  # fmt: skip
GoalKind = Literal["reach_exit", "collect_all", "survive"]
DefectKind = Literal[
    "missing_blocking", "wrong_tag", "wrong_special", "geometry_trap", "one_way_lift",
    "teleport_loop", "bad_pickup",
]  # fmt: skip
Severity = Literal["low", "medium", "high"]
KEY_TYPES = ("key_red", "key_blue", "key_yellow")
WALL_THING_TYPES = ("exit_switch", "switch")  # emitted as linedefs, not things


class SpecError(ValueError):
    pass


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Room(_Model):
    id: str
    x0: int
    y0: int
    x1: int
    y1: int
    floor: int
    ceiling: int
    tag: int | None = None
    damage: int = 0
    light: int = 160
    union_with: str | None = None

    @model_validator(mode="after")
    def _positive_extent(self) -> Room:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"room {self.id}: need x1 > x0 and y1 > y0")
        return self

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


class Portal(_Model):
    id: str
    a: str
    b: str
    kind: PortalKind
    width: int = 96
    requires: str | None = None
    # Adjacent kinds: centre of the opening on the shared wall (default: its midpoint).
    # Teleport: centre of the pad in room a (required).
    at: tuple[int, int] | None = None


class Thing(_Model):
    id: str
    type: ThingType
    room: str
    pos: tuple[int, int]
    interactive: bool = False
    opens: str | None = None  # switch only: the door portal it opens


class Goal(_Model):
    kind: GoalKind
    target: str | None = None
    requires: list[str] = Field(default_factory=list)
    time_limit_tics: int


class Defect(_Model):
    id: str
    kind: DefectKind
    target: str
    params: dict[str, object] = Field(default_factory=dict)
    expected_oracle: str
    severity: Severity
    notes: str = ""


@dataclass(frozen=True)
class SharedWall:
    """Collinear overlap of two room edges. `axis` is the constant coordinate's axis:
    "x" means a vertical wall at x = `at`, spanning y in [lo, hi]."""

    axis: Literal["x", "y"]
    at: int
    lo: int
    hi: int

    @property
    def length(self) -> int:
        return self.hi - self.lo


def shared_wall(a: Room, b: Room) -> SharedWall | None:
    for x in {a.x1, a.x0}:
        if x in (b.x0, b.x1) and (x == a.x1) != (x == b.x1):
            lo, hi = max(a.y0, b.y0), min(a.y1, b.y1)
            if hi > lo:
                return SharedWall("x", x, lo, hi)
    for y in {a.y1, a.y0}:
        if y in (b.y0, b.y1) and (y == a.y1) != (y == b.y1):
            lo, hi = max(a.x0, b.x0), min(a.x1, b.x1)
            if hi > lo:
                return SharedWall("y", y, lo, hi)
    return None


def _overlap_area(a: Room, b: Room) -> int:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if w > 0 and h > 0 else 0


class MapSpec(_Model):
    name: str
    slot: int = Field(ge=1, le=32)
    cell_size: int = 64
    rooms: list[Room] = Field(min_length=1)
    portals: list[Portal] = Field(default_factory=list)
    things: list[Thing]
    goal: Goal
    defects: list[Defect] = Field(default_factory=list)
    seed: int

    @property
    def map_lump(self) -> str:
        return f"MAP{self.slot:02d}"

    def room(self, room_id: str) -> Room:
        return next(r for r in self.rooms if r.id == room_id)

    def portal(self, portal_id: str) -> Portal:
        return next(p for p in self.portals if p.id == portal_id)

    def thing(self, thing_id: str) -> Thing:
        return next(t for t in self.things if t.id == thing_id)

    def without_defects(self) -> MapSpec:
        """The `fixed` build variant (PLAN: Build matrix)."""
        return self.model_copy(update={"defects": []})

    @model_validator(mode="after")
    def _validate(self) -> MapSpec:
        self._check_ids()
        self._check_player_start()
        self._check_portals()
        self._check_keys()
        self._check_overlaps()
        self._check_things()
        self._check_defects()
        self.check_reachable()
        return self

    # Rule: ids are unique across rooms, portals and things (needed by defect targets).
    def _check_ids(self) -> None:
        ids = [r.id for r in self.rooms] + [p.id for p in self.portals]
        ids += [t.id for t in self.things]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise SpecError(f"duplicate ids: {dupes}")

    # Rule 1: exactly one player_start.
    def _check_player_start(self) -> None:
        n = sum(t.type == "player_start" for t in self.things)
        if n != 1:
            raise SpecError(f"need exactly one player_start, found {n}")

    # Rule 2: portal rooms exist and share a wall at least `width` long (except teleports).
    def _check_portals(self) -> None:
        room_ids = {r.id for r in self.rooms}
        for p in self.portals:
            for end in (p.a, p.b):
                if end not in room_ids:
                    raise SpecError(f"portal {p.id}: unknown room {end!r}")
            if p.a == p.b:
                raise SpecError(f"portal {p.id}: connects room {p.a!r} to itself")
            if (p.kind == "locked_door") != (p.requires is not None):
                raise SpecError(f"portal {p.id}: `requires` is for locked_door portals only")
            if p.kind == "teleport":
                if p.at is None or not self.room(p.a).contains(*p.at):
                    raise SpecError(f"portal {p.id}: teleport needs `at` inside room {p.a!r}")
                continue
            wall = shared_wall(self.room(p.a), self.room(p.b))
            if wall is None or wall.length < p.width:
                have = 0 if wall is None else wall.length
                raise SpecError(
                    f"portal {p.id}: rooms {p.a!r} and {p.b!r} share {have} mu of wall, "
                    f"need >= {p.width}"
                )
            if p.at is not None:
                along, across = (p.at[1], p.at[0]) if wall.axis == "x" else p.at
                if across != wall.at or not (
                    wall.lo + p.width / 2 <= along <= wall.hi - p.width / 2
                ):
                    raise SpecError(f"portal {p.id}: `at` {p.at} does not fit on the wall")
            if p.kind == "lift" and self.room(p.a).floor == self.room(p.b).floor:
                raise SpecError(f"portal {p.id}: a lift needs rooms at different floor heights")

    # Rule 3: every `requires` names a key type with a matching thing.
    def _check_keys(self) -> None:
        present = {t.type for t in self.things}
        wanted = [(p.id, p.requires) for p in self.portals if p.requires]
        wanted += [("goal", k) for k in self.goal.requires]
        for owner, key in wanted:
            if key not in KEY_TYPES or key not in present:
                raise SpecError(f"{owner}: requires {key!r}, which no key_* thing provides")

    # Rule 5: rooms overlap only as declared union_with pairs.
    def _check_overlaps(self) -> None:
        for i, a in enumerate(self.rooms):
            for b in self.rooms[i + 1 :]:
                paired = a.union_with == b.id or b.union_with == a.id
                if _overlap_area(a, b) and not paired:
                    raise SpecError(f"rooms {a.id!r} and {b.id!r} overlap without union_with")
            if a.union_with is not None and a.union_with not in {r.id for r in self.rooms}:
                raise SpecError(f"room {a.id}: union_with unknown room {a.union_with!r}")

    def _check_things(self) -> None:
        room_ids = {r.id for r in self.rooms}
        door_ids = {p.id for p in self.portals if p.kind == "door"}
        for t in self.things:
            if t.room not in room_ids:
                raise SpecError(f"thing {t.id}: unknown room {t.room!r}")
            if not self.room(t.room).contains(*t.pos):
                raise SpecError(f"thing {t.id}: pos {t.pos} is outside room {t.room!r}")
            if (t.type == "switch") != (t.opens is not None):
                raise SpecError(f"thing {t.id}: `opens` is for switch things only")
            if t.opens is not None and t.opens not in door_ids:
                raise SpecError(f"thing {t.id}: opens {t.opens!r}, which is not a door portal")
        target = self.goal.target
        if self.goal.kind == "reach_exit":
            if target is None or target not in {t.id for t in self.things}:
                raise SpecError("goal reach_exit needs a target thing")
            if self.thing(target).type != "exit_switch":
                raise SpecError(f"goal target {target!r} is not an exit_switch")

    # Rule 6: every defect's target exists. Its kind is checked by the DefectKind type.
    def _check_defects(self) -> None:
        targets = {r.id for r in self.rooms} | {p.id for p in self.portals}
        targets |= {t.id for t in self.things}
        for d in self.defects:
            if d.target not in targets:
                raise SpecError(f"defect {d.id}: unknown target {d.target!r}")

    # Rule 4: the intended portal graph connects every room and the goal is reachable.
    def check_reachable(self) -> set[str]:
        """BFS over the intended portal graph, ignoring geometry and defects.

        Doors opened by a switch are passable once the switch's room is reached; locked
        doors once a matching key's room is reached; teleports are one-way a -> b.
        Returns the reachable room ids; raises SpecError if any room or the goal is not.
        """
        spawn = next(t.room for t in self.things if t.type == "player_start")
        switched = {t.opens: t.room for t in self.things if t.type == "switch"}
        unions = [(r.id, r.union_with) for r in self.rooms if r.union_with is not None]
        reached = {spawn}
        changed = True
        while changed:
            changed = False
            for a, b in unions:  # a union_with pair is one walkable space
                if (a in reached) != (b in reached):
                    reached |= {a, b}
                    changed = True
            keys = {t.type for t in self.things if t.room in reached and t.type in KEY_TYPES}
            for p in self.portals:
                if p.kind == "locked_door" and p.requires not in keys:
                    continue
                if p.id in switched and switched[p.id] not in reached:
                    continue
                ends: Iterable[tuple[str, str]] = [(p.a, p.b)]
                if p.kind != "teleport":
                    ends = [(p.a, p.b), (p.b, p.a)]
                for src, dst in ends:
                    if src in reached and dst not in reached:
                        reached.add(dst)
                        changed = True
        unreached = sorted({r.id for r in self.rooms} - reached)
        if unreached:
            raise SpecError(f"rooms unreachable from spawn: {unreached}")
        keys = {t.type for t in self.things if t.type in KEY_TYPES}
        missing = sorted(set(self.goal.requires) - keys)
        if missing:
            raise SpecError(f"goal requires keys that cannot be collected: {missing}")
        return reached


def load_spec(path: Path) -> MapSpec:
    with path.open(encoding="utf-8") as f:
        return MapSpec.model_validate(yaml.safe_load(f))
