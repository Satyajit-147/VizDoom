"""Defect kind registry (PLAN: Defect injection, mechanically).

Each kind mutates a draft CompiledMap after rooms, portals and switches are laid out
and before linedefs are built, then returns a DefectRecord for the ground truth. Most of
the plan's mutations live below the MapSpec's level of abstraction (linedef flags,
special arguments, TIDs), so they act on the compiled map, not the spec
(docs/decisions.md D20).

| kind             | mutation                                                          |
|------------------|-------------------------------------------------------------------|
| missing_blocking | wall segment becomes a passable fake wall into a pocket (D20)     |
| wrong_tag        | the switch's Door_Open targets a tag no sector carries            |
| wrong_special    | Door_LockedRaise lock argument set to 0                           |
| geometry_trap    | room floor lowered by `drop` (> 24) below its neighbours          |
| one_way_lift     | lift trigger kept on the upper side only                          |
| teleport_loop    | two teleports land the player facing each other's pads            |
| bad_pickup       | medikit replaced by a Soulsphere (health over 100)                |
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

from qatester.mapgen import udmf
from qatester.mapgen.geometry import (
    DOOR_LOCKED_RAISE,
    PAD,
    PLAYER_RADIUS,
    STEP_MAX,
    CompiledMap,
    DefectRecord,
    GeometryError,
    Rect,
    Region,
    SectorInfo,
    Special,
)
from qatester.mapgen.spec import Defect, Portal, Room, shared_wall

POCKET_DEPTH = 128  # how far past a missing wall the player can walk
MIN_GAP = 2 * PLAYER_RADIUS + 8  # narrowest fake-wall gap a player fits through
LOOP_STANDOFF = PAD // 2 + 24  # teleport_loop arrival distance from the pad centre
SOULSPHERE = 2013


def _record(d: Defect, position: tuple[float, float, float], region: list[Rect],
            affected: list[str], change: str) -> DefectRecord:  # fmt: skip
    return DefectRecord(
        d.id, d.kind, d.target, d.expected_oracle, d.severity, dict(d.params), d.notes,
        position, tuple(region), tuple(affected), change,
    )  # fmt: skip


def _room(cm: CompiledMap, d: Defect) -> Room:
    if d.target not in {r.id for r in cm.spec.rooms}:
        raise GeometryError(f"defect {d.id}: {d.kind} targets a room, got {d.target!r}")
    return cm.spec.room(d.target)


def _portal(cm: CompiledMap, d: Defect, *kinds: str) -> Portal:
    ids = {p.id for p in cm.spec.portals}
    if d.target not in ids or cm.spec.portal(d.target).kind not in kinds:
        raise GeometryError(f"defect {d.id}: {d.kind} targets a {'/'.join(kinds)} portal")
    return cm.spec.portal(d.target)


def _opening_centre(cm: CompiledMap, p: Portal) -> tuple[float, float]:
    wall = shared_wall(cm.spec.room(p.a), cm.spec.room(p.b))
    assert wall is not None
    along = (wall.lo + wall.hi) / 2
    if p.at is not None:
        along = p.at[1] if wall.axis == "x" else p.at[0]
    return (wall.at, along) if wall.axis == "x" else (along, wall.at)


def _rect(room: Room) -> Rect:
    return Rect(room.x0, room.y0, room.x1, room.y1)


def missing_blocking(cm: CompiledMap, d: Defect) -> DefectRecord:
    """params: wall (north/south/east/west), segment [lo, hi] in absolute map units along
    that wall, optional depth. The segment must lie on an outer (one-sided) wall."""
    room = _room(cm, d)
    wall = d.params.get("wall")
    seg = d.params.get("segment")
    depth = int(str(d.params.get("depth", POCKET_DEPTH)))
    if wall not in ("north", "south", "east", "west") or not (
        isinstance(seg, list) and len(seg) == 2
    ):
        raise GeometryError(f"defect {d.id}: needs params wall and segment [lo, hi]")
    lo, hi = int(seg[0]), int(seg[1])
    span = (room.x0, room.x1) if wall in ("north", "south") else (room.y0, room.y1)
    if not (span[0] < lo < hi < span[1]) or hi - lo < MIN_GAP:
        raise GeometryError(f"defect {d.id}: segment {seg} must fit inside the {wall} wall "
                            f"and be at least {MIN_GAP} mu long")  # fmt: skip
    pocket = {
        "north": Rect(lo, room.y1, hi, room.y1 + depth),
        "south": Rect(lo, room.y0 - depth, hi, room.y0),
        "east": Rect(room.x1, lo, room.x1 + depth, hi),
        "west": Rect(room.x0 - depth, lo, room.x0, hi),
    }[wall]
    for r in cm.regions:
        w = min(r.rect.x1, pocket.x1) - max(r.rect.x0, pocket.x0)
        h = min(r.rect.y1, pocket.y1) - max(r.rect.y0, pocket.y0)
        if w > 0 and h > 0:
            raise GeometryError(f"defect {d.id}: the space behind the {wall} wall of "
                                f"{room.id} is not empty")  # fmt: skip
    sector = cm.room_sector(room.id)
    # The wall must be a real outer wall here: room on one side, nothing on the other.
    inward = {"north": (0, -1), "south": (0, 1), "east": (-1, 0), "west": (1, 0)}[wall]
    for t in (lo + 1.0, (lo + hi) / 2, hi - 1.0):
        edge: tuple[float, float]
        if wall in ("north", "south"):
            edge = (t, room.y1 if wall == "north" else room.y0)
        else:
            edge = (room.x1 if wall == "east" else room.x0, t)
        inside = cm.owner_at(edge[0] + inward[0] * 0.5, edge[1] + inward[1] * 0.5)
        if inside != sector:
            raise GeometryError(f"defect {d.id}: segment is not on a wall of {room.id} "
                                "(a portal or wall strip is in the way)")  # fmt: skip
    key = f"pocket:{d.id}"
    info = cm.sectors[sector]
    cm.sectors[key] = SectorInfo(key, "pocket", info.floor, info.ceiling, light=info.light)
    cm.regions.append(Region(pocket, key, 2))
    mid = (lo + hi) / 2
    if wall in ("north", "south"):
        pos = (mid, float(room.y1 if wall == "north" else room.y0), float(room.floor))
    else:
        pos = (float(room.x1 if wall == "east" else room.x0), mid, float(room.floor))
    return _record(d, pos, [pocket], [room.id],
                   f"{wall} wall of {room.id} [{lo},{hi}] is passable; "
                   f"{depth} mu pocket behind it")  # fmt: skip


def wrong_tag(cm: CompiledMap, d: Defect) -> DefectRecord:
    things = {t.id: t for t in cm.spec.things}
    switch = things.get(d.target)
    if switch is None or switch.type != "switch" or switch.opens is None:
        raise GeometryError(f"defect {d.id}: wrong_tag targets a switch thing")
    i = next(i for i, m in enumerate(cm.marks) if m.source == switch.id)
    unused = max([s.tag for s in cm.sectors.values()] + [0]) + 1
    old = cm.marks[i].special
    cm.marks[i] = replace(cm.marks[i], special=Special(old.special, (unused, *old.args[1:]),
                                                       old.flags))  # fmt: skip
    door = cm.spec.portal(switch.opens)
    cut_off = sorted(
        {r.id for r in cm.spec.rooms} - cm.spec.reachable_rooms(without=[door.id])
    )
    x, y = _opening_centre(cm, door)
    floor = cm.spec.room(door.a).floor
    return _record(d, (x, y, float(floor)), [_rect(cm.spec.room(r)) for r in cut_off], cut_off,
                   f"switch {switch.id} targets tag {unused} instead of door {door.id} "
                   f"(tag {old.args[0]})")  # fmt: skip


def wrong_special(cm: CompiledMap, d: Defect) -> DefectRecord:
    p = _portal(cm, d, "locked_door")
    info = cm.sectors[f"door:{p.id}"]
    assert info.face is not None and info.face.special == DOOR_LOCKED_RAISE
    old_lock = info.face.args[3]
    info.face = Special(info.face.special, (*info.face.args[:3], 0), info.face.flags)
    x, y = _opening_centre(cm, p)
    return _record(d, (x, y, float(info.floor)), [cm.region_of(info.key)], [p.a, p.b],
                   f"{p.id} emitted with lock 0 instead of {old_lock} ({p.requires})")


def geometry_trap(cm: CompiledMap, d: Defect) -> DefectRecord:
    room = _room(cm, d)
    drop = int(str(d.params.get("drop", 0)))
    if drop <= STEP_MAX:
        raise GeometryError(f"defect {d.id}: drop must exceed the {STEP_MAX} mu step height")
    if any(p.kind in ("lift", "teleport") and room.id in (p.a, p.b) for p in cm.spec.portals):
        raise GeometryError(f"defect {d.id}: {room.id} has a lift or teleport, so it is no trap")
    info = cm.sectors[cm.room_sector(room.id)]
    info.floor -= drop
    cx, cy = _rect(room).centre
    return _record(d, (cx, cy, float(info.floor)), [_rect(room)], [room.id],
                   f"{room.id} floor lowered {drop} mu to {info.floor}")


def one_way_lift(cm: CompiledMap, d: Defect) -> DefectRecord:
    p = _portal(cm, d, "lift")
    ra, rb = cm.spec.room(p.a), cm.spec.room(p.b)
    lower = ra if ra.floor < rb.floor else rb
    info = cm.sectors[f"lift:{p.id}"]
    info.face_skip = info.face_skip | {cm.room_sector(lower.id)}
    lift = cm.region_of(info.key)
    cx, cy = lift.centre
    return _record(d, (cx, cy, float(lower.floor)), [lift, _rect(lower)], [lower.id],
                   f"{p.id} has no trigger on the {lower.id} side; only the upper side "
                   "can call the lift")  # fmt: skip


def teleport_loop(cm: CompiledMap, d: Defect) -> DefectRecord:
    """params: partner, the teleport going the other way. Each teleport lands the player
    just in front of the other's pad, facing it, so walking on loops forever."""
    a = _portal(cm, d, "teleport")
    partner = str(d.params.get("partner", ""))
    if partner not in {p.id for p in cm.spec.portals} or cm.spec.portal(partner).kind != (
        "teleport"
    ):
        raise GeometryError(f"defect {d.id}: needs params partner, the opposite teleport")
    b = cm.spec.portal(partner)
    if (a.a, a.b) != (b.b, b.a):
        raise GeometryError(f"defect {d.id}: {a.id} and {b.id} must connect the same rooms "
                            "in opposite directions")  # fmt: skip
    for src, dst in ((a, b), (b, a)):  # src now lands in front of dst's pad
        assert dst.at is not None
        room = cm.spec.room(dst.a)
        cx, cy = _rect(room).centre
        vx, vy = cx - dst.at[0], cy - dst.at[1]
        norm = math.hypot(vx, vy) or 1.0
        x = dst.at[0] + LOOP_STANDOFF * vx / norm
        y = dst.at[1] + LOOP_STANDOFF * vy / norm
        angle = round(math.degrees(math.atan2(-vy, -vx))) % 360
        tid = cm.teleport_tids[src.id]
        i = next(i for i, t in enumerate(cm.things)
                 if t.type == udmf.THING_EDNUMS["teleport_dest"] and t.id == tid)  # fmt: skip
        cm.things[i] = replace(cm.things[i], x=x, y=y, angle=angle)
        cm.teleport_dests[src.id] = (x, y)
        cm.exempt.add(f"teleport_dest:{src.id}")
    assert a.at is not None
    pads = [cm.region_of(f"pad:{a.id}"), cm.region_of(f"pad:{b.id}")]
    floor = cm.spec.room(a.a).floor
    return _record(d, (float(a.at[0]), float(a.at[1]), float(floor)), pads, [a.a, a.b],
                   f"{a.id} and {b.id} land the player facing each other's pads")


def bad_pickup(cm: CompiledMap, d: Defect) -> DefectRecord:
    thing = next((t for t in cm.spec.things if t.id == d.target), None)
    if thing is None or thing.type != "health":
        raise GeometryError(f"defect {d.id}: bad_pickup targets a health thing")
    i = cm.thing_index[thing.id]
    cm.things[i] = replace(cm.things[i], type=SOULSPHERE)
    floor = cm.spec.room(thing.room).floor
    return _record(d, (float(thing.pos[0]), float(thing.pos[1]), float(floor)), [],
                   [thing.room], f"{thing.id} is a Soulsphere (+100 health, cap 200)")


REGISTRY: dict[str, Callable[[CompiledMap, Defect], DefectRecord]] = {
    "missing_blocking": missing_blocking,
    "wrong_tag": wrong_tag,
    "wrong_special": wrong_special,
    "geometry_trap": geometry_trap,
    "one_way_lift": one_way_lift,
    "teleport_loop": teleport_loop,
    "bad_pickup": bad_pickup,
}


def apply_defect(cm: CompiledMap, d: Defect) -> DefectRecord:
    return REGISTRY[d.kind](cm, d)

