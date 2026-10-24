"""MapSpec rectangles -> sectors, linedefs, sidedefs, things (PLAN: Geometry generation).

Construction (docs/decisions.md D14):

- Every room is a rectangle owned by its sector. Rooms joined by `union_with` share one.
- A wall shared by two rooms becomes a WALL-thick strip of void straddling the shared
  edge, so each room sees an ordinary one-sided wall. Doom has no zero-thickness solid
  wall between two sectors.
- A portal punches its opening (width x WALL) back out of that strip:
  open -> owned by room a; door/locked_door -> its own door sector, closed
  (ceiling == floor) until opened; lift -> a platform sector extending LIFT_DEPTH into
  the lower room. A teleport is a PAD-sized pad sector inside room a whose edges carry
  the Teleport special, sending the player to a TeleportDest at room b's centre.
- Exit switches and door switches are WALL_MARK-long specials on the one-sided wall of
  their room nearest to the thing's `pos`.

The plane is cut into a grid along every rectangle edge. Each grid cell gets the
highest-priority region containing it, and a linedef is emitted wherever two
neighbouring cells have different owners. Collinear pieces with identical attributes
are then merged. Front sides always face the sector on the right of v1 -> v2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from qatester.mapgen import udmf
from qatester.mapgen.spec import WALL_THING_TYPES, MapSpec, Portal, shared_wall

WALL = 16  # thickness of the void strip between adjacent rooms, and of door sectors
LIFT_DEPTH = 64  # how far a lift platform reaches into the lower room past the strip
PAD = 64  # teleport pad side
WALL_MARK = 64  # length of a switch linedef
STEP_MAX = 24  # PLAN units: maximum step-up height
PLAYER_HEIGHT = 56
PLAYER_RADIUS = 16
DOOR_OPEN_GAP = 4  # Doom doors open to the lowest neighbouring ceiling minus 4

DOOR_SPEED, DOOR_DELAY = 16, 150
LIFT_SPEED, LIFT_DELAY = 32, 105
LOCK_NUMBERS = {"key_red": 1, "key_blue": 2, "key_yellow": 3}
FIRST_AUTO_TAG = 100

# Specials (PLAN Appendix A, "Linedef specials used by the compiler").
DOOR_OPEN = 11
DOOR_RAISE = 12
DOOR_LOCKED_RAISE = 13
PLAT_DOWN_WAIT_UP_STAY = 62
TELEPORT = 70
EXIT_NORMAL = 243

TEX_WALL = "BRICK7"
TEX_DOOR = "BIGDOOR2"
TEX_LIFT = "PLAT1"
TEX_SWITCH = "SW1COMP"

SectorKind = Literal["room", "door", "lift", "pad", "pocket"]  # pocket: defect-only space
USE_FLAGS = frozenset({"playeruse", "repeatspecial"})


class GeometryError(ValueError):
    pass


@dataclass(frozen=True)
class Rect:
    x0: int
    y0: int
    x1: int
    y1: int

    def contains(self, x: float, y: float) -> bool:
        # Half-open, so a point on the seam between two touching regions has one owner.
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2


@dataclass(frozen=True)
class Special:
    special: int
    args: tuple[int, ...]
    flags: frozenset[str]


@dataclass
class SectorInfo:
    key: str
    kind: SectorKind
    floor: int
    ceiling: int  # door sectors: the open height; the WAD emits them closed
    light: int = 160
    tag: int = 0
    damage: int = 0
    rooms: tuple[str, ...] = ()  # room ids for room sectors
    portal: str | None = None  # owning portal for door/lift/pad sectors
    face: Special | None = None  # on lines where this sector is the back, a room the front
    face_skip: frozenset[str] = frozenset()  # front sectors that get no face special


@dataclass(frozen=True)
class Region:
    rect: Rect
    sector: str | None  # None = void
    priority: int


@dataclass(frozen=True)
class WallMark:
    """A special on the one-sided wall of `sector` along an axis-aligned segment."""

    axis: Literal["x", "y"]  # "x": vertical wall at x = at
    at: int
    lo: int
    hi: int
    sector: str
    special: Special
    texture: str
    source: str  # thing or defect id

    def covers(self, axis: str, at: int, lo: int, hi: int) -> bool:
        return axis == self.axis and at == self.at and self.lo <= lo and hi <= self.hi


@dataclass(frozen=True)
class Line:
    v1: tuple[int, int]
    v2: tuple[int, int]
    front: str
    back: str | None
    special: Special | None
    texture: str | None  # one-sided middle texture override (switches)
    source: str | None


@dataclass(frozen=True)
class DefectRecord:
    """What an injected defect changed and where, for the ground truth (PLAN rule 3:
    "a defect you cannot declare is a defect you cannot score")."""

    id: str
    kind: str
    target: str
    expected_oracle: str
    severity: str
    params: dict[str, object]
    notes: str
    position: tuple[float, float, float]
    region: tuple[Rect, ...]  # area the defect occupies or cuts off
    affected_rooms: tuple[str, ...]
    change: str  # one-line description of the mutation


@dataclass
class CompiledMap:
    spec: MapSpec
    sectors: dict[str, SectorInfo]
    regions: list[Region]
    lines: list[Line]
    things: list[udmf.Thing]
    marks: list[WallMark] = field(default_factory=list)
    teleport_dests: dict[str, tuple[float, float]] = field(default_factory=dict)  # portal id
    teleport_tids: dict[str, int] = field(default_factory=dict)  # portal id -> TID
    thing_index: dict[str, int] = field(default_factory=dict)  # spec thing id -> things[i]
    defects: list[DefectRecord] = field(default_factory=list)
    exempt: set[str] = field(default_factory=set)  # validate.check_things names to skip

    def owner_at(self, x: float, y: float) -> str | None:
        return _owner(self.regions, x, y)

    def room_sector(self, room_id: str) -> str:
        return next(s.key for s in self.sectors.values() if room_id in s.rooms)

    def region_of(self, sector: str) -> Rect:
        return next(r.rect for r in self.regions if r.sector == sector)

    def owner_grid(self) -> tuple[list[int], list[int], list[list[str | None]]]:
        """Grid lines and the owning sector of each cell, owner[i][j] for the cell
        between xs[i]..xs[i+1] and ys[j]..ys[j+1]."""
        xs, ys = _grid(self.regions, self.marks)
        owner = [
            [_owner(self.regions, (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2)
             for j in range(len(ys) - 1)]
            for i in range(len(xs) - 1)
        ]  # fmt: skip
        return xs, ys, owner

    def textmap(self) -> udmf.TextMap:
        tm = udmf.TextMap()
        index = {key: i for i, key in enumerate(self.sectors)}
        tm.sectors = [
            udmf.Sector(
                s.floor, s.ceiling if s.kind != "door" else s.floor,
                lightlevel=s.light, id=s.tag, damageamount=s.damage,
            )
            for s in self.sectors.values()
        ]  # fmt: skip
        vertex_index: dict[tuple[int, int], int] = {}
        for line in self.lines:
            for v in (line.v1, line.v2):
                if v not in vertex_index:
                    vertex_index[v] = len(vertex_index)
                    tm.vertices.append(udmf.Vertex(*v))
            front = len(tm.sidedefs)
            if line.back is None:
                tm.sidedefs.append(
                    udmf.Sidedef(index[line.front], texturemiddle=line.texture or TEX_WALL)
                )
                flags = frozenset({"blocking"})
                sideback = None
            else:
                back_kind = self.sectors[line.back].kind
                top = TEX_DOOR if back_kind == "door" else TEX_WALL
                bottom = TEX_LIFT if back_kind == "lift" else TEX_WALL
                # A pocket's mouth is a passable line drawn as a solid wall (missing_blocking).
                middle = TEX_WALL if back_kind == "pocket" else None
                tm.sidedefs.append(
                    udmf.Sidedef(index[line.front], texturemiddle=middle, texturetop=top,
                                 texturebottom=bottom)
                )  # fmt: skip
                tm.sidedefs.append(
                    udmf.Sidedef(index[line.back], texturemiddle=middle, texturetop=TEX_WALL,
                                 texturebottom=TEX_WALL)
                )  # fmt: skip
                flags = frozenset({"twosided"})
                sideback = front + 1
            special, args = 0, tuple[int, ...]()
            if line.special is not None:
                special, args = line.special.special, line.special.args
                flags |= line.special.flags
            tm.linedefs.append(
                udmf.Linedef(
                    vertex_index[line.v1], vertex_index[line.v2], front, sideback,
                    flags=flags, special=special, args=args,
                )
            )  # fmt: skip
        tm.things = list(self.things)
        return tm


def _owner(regions: Sequence[Region], x: float, y: float) -> str | None:
    found: Region | None = None
    for r in regions:
        if r.rect.contains(x, y) and (found is None or r.priority >= found.priority):
            found = r
    return found.sector if found is not None else None


def _opening(p: Portal, spec: MapSpec) -> tuple[Rect, Literal["x", "y"], int]:
    """Opening rect straddling the shared wall, plus the wall's axis and coordinate."""
    wall = shared_wall(spec.room(p.a), spec.room(p.b))
    assert wall is not None  # guaranteed by spec validation
    if p.at is not None:
        centre = p.at[1] if wall.axis == "x" else p.at[0]
    else:
        centre = (wall.lo + wall.hi) // 2
    lo = centre - p.width // 2
    hi = lo + p.width
    half = WALL // 2
    if wall.axis == "x":
        rect = Rect(wall.at - half, lo, wall.at + half, hi)
    else:
        rect = Rect(lo, wall.at - half, hi, wall.at + half)
    return rect, wall.axis, wall.at


def _room_sectors(spec: MapSpec) -> dict[str, SectorInfo]:
    parent = {r.id: r.id for r in spec.rooms}

    def root(i: str) -> str:
        while parent[i] != i:
            i = parent[i]
        return i

    for r in spec.rooms:
        if r.union_with is not None:
            parent[root(r.id)] = root(r.union_with)
    sectors: dict[str, SectorInfo] = {}
    for r in spec.rooms:
        key = root(r.id)
        if key in sectors:
            s = sectors[key]
            if (s.floor, s.ceiling) != (r.floor, r.ceiling):
                raise GeometryError(f"union rooms {s.rooms} and {r.id} differ in height")
            s.rooms = (*s.rooms, r.id)
            s.tag = s.tag or (r.tag or 0)
            continue
        sectors[key] = SectorInfo(
            key, "room", r.floor, r.ceiling, light=r.light, tag=r.tag or 0,
            damage=r.damage, rooms=(r.id,),
        )  # fmt: skip
    return sectors


def compile_map(spec: MapSpec, apply_defects: bool = True) -> CompiledMap:
    """Compile a spec. Its defects are injected unless `apply_defects` is False; the
    `fixed` build variant passes `spec.without_defects()` instead."""
    sectors = _room_sectors(spec)
    room_sector = {rid: s.key for s in sectors.values() for rid in s.rooms}
    regions = [
        Region(Rect(r.x0, r.y0, r.x1, r.y1), room_sector[r.id], 0) for r in spec.rooms
    ]
    used_tags = {s.tag for s in sectors.values() if s.tag}
    next_tag = max([FIRST_AUTO_TAG - 1, *used_tags]) + 1
    things: list[udmf.Thing] = []
    dests: dict[str, tuple[float, float]] = {}
    tids: dict[str, int] = {}

    # Void strips along every wall shared by two distinct room sectors.
    half = WALL // 2
    for i, a in enumerate(spec.rooms):
        for b in spec.rooms[i + 1 :]:
            wall = shared_wall(a, b)
            if wall is None or room_sector[a.id] == room_sector[b.id]:
                continue
            if wall.axis == "x":
                rect = Rect(wall.at - half, wall.lo, wall.at + half, wall.hi)
            else:
                rect = Rect(wall.lo, wall.at - half, wall.hi, wall.at + half)
            regions.append(Region(rect, None, 1))

    switched = {t.opens for t in spec.things if t.type == "switch"}
    door_tags: dict[str, int] = {}
    tid = 1
    for p in spec.portals:
        ra, rb = spec.room(p.a), spec.room(p.b)
        if p.kind == "teleport":
            assert p.at is not None
            pad = Rect(p.at[0] - PAD // 2, p.at[1] - PAD // 2, p.at[0] + PAD // 2, p.at[1] + PAD // 2)
            key = f"pad:{p.id}"
            sectors[key] = SectorInfo(
                key, "pad", ra.floor, ra.ceiling, light=ra.light, portal=p.id,
                face=Special(TELEPORT, (tid, 0), frozenset({"playercross", "repeatspecial"})),
            )  # fmt: skip
            regions.append(Region(pad, key, 2))
            cx, cy = Rect(rb.x0, rb.y0, rb.x1, rb.y1).centre
            things.append(udmf.Thing(cx, cy, udmf.THING_EDNUMS["teleport_dest"], id=tid))
            dests[p.id] = (cx, cy)
            tids[p.id] = tid
            tid += 1
            continue

        rect, axis, at = _opening(p, spec)
        if p.kind == "open":
            if abs(ra.floor - rb.floor) > STEP_MAX:
                raise GeometryError(f"portal {p.id}: open step of {abs(ra.floor - rb.floor)}")
            regions.append(Region(rect, room_sector[p.a], 2))
        elif p.kind in ("door", "locked_door"):
            if abs(ra.floor - rb.floor) > STEP_MAX:
                raise GeometryError(f"portal {p.id}: door step of {abs(ra.floor - rb.floor)}")
            key = f"door:{p.id}"
            face: Special | None
            tag = 0
            if p.id in switched:  # remote door: opened only by its switch
                tag, next_tag = next_tag, next_tag + 1
                door_tags[p.id] = tag
                face = None
            elif p.kind == "door":
                face = Special(DOOR_RAISE, (0, DOOR_SPEED, DOOR_DELAY), USE_FLAGS)
            else:
                assert p.requires is not None
                lock = LOCK_NUMBERS[p.requires]
                face = Special(DOOR_LOCKED_RAISE, (0, DOOR_SPEED, DOOR_DELAY, lock), USE_FLAGS)
            sectors[key] = SectorInfo(
                key, "door", max(ra.floor, rb.floor), min(ra.ceiling, rb.ceiling) - DOOR_OPEN_GAP,
                light=min(ra.light, rb.light), tag=tag, portal=p.id, face=face,
            )  # fmt: skip
            regions.append(Region(rect, key, 2))
        elif p.kind == "lift":
            lower, upper = (ra, rb) if ra.floor < rb.floor else (rb, ra)
            key = f"lift:{p.id}"
            tag, next_tag = next_tag, next_tag + 1
            sectors[key] = SectorInfo(
                key, "lift", upper.floor, upper.ceiling, light=upper.light, tag=tag,
                portal=p.id,
                face=Special(PLAT_DOWN_WAIT_UP_STAY, (tag, LIFT_SPEED, LIFT_DELAY), USE_FLAGS),
            )  # fmt: skip
            # Extend the platform into the lower room, away from the wall.
            lower_is_low_side = (lower.x1 if axis == "x" else lower.y1) == at
            if axis == "x":
                rect = (
                    Rect(rect.x0 - LIFT_DEPTH, rect.y0, rect.x1, rect.y1)
                    if lower_is_low_side
                    else Rect(rect.x0, rect.y0, rect.x1 + LIFT_DEPTH, rect.y1)
                )
            else:
                rect = (
                    Rect(rect.x0, rect.y0 - LIFT_DEPTH, rect.x1, rect.y1)
                    if lower_is_low_side
                    else Rect(rect.x0, rect.y0, rect.x1, rect.y1 + LIFT_DEPTH)
                )
            regions.append(Region(rect, key, 2))

    thing_index: dict[str, int] = {}
    for t in spec.things:
        if t.type in WALL_THING_TYPES:
            continue
        thing_index[t.id] = len(things)
        things.append(udmf.Thing(t.pos[0], t.pos[1], udmf.THING_EDNUMS[t.type]))

    _check_regions(spec, sectors, regions)

    # Switch linedefs need the merged walls to find the nearest one, so solve twice.
    marks: list[WallMark] = []
    for t in spec.things:
        if t.type == "exit_switch":
            special = Special(EXIT_NORMAL, (0,), frozenset({"playeruse"}))
        elif t.type == "switch":
            assert t.opens is not None
            special = Special(DOOR_OPEN, (door_tags[t.opens], DOOR_SPEED), frozenset({"playeruse"}))
        else:
            continue
        marks.append(_place_mark(regions, room_sector[t.room], t.pos, special, t.id))

    cm = CompiledMap(spec, sectors, regions, [], things, marks, dests, tids, thing_index)
    if apply_defects:
        from qatester.mapgen.defects import apply_defect  # defects builds on this module

        cm.defects = [apply_defect(cm, d) for d in spec.defects]
    cm.lines = _build_lines(cm.regions, cm.sectors, cm.marks)
    return cm


def _check_regions(
    spec: MapSpec, sectors: dict[str, SectorInfo], regions: list[Region]
) -> None:
    special = [r for r in regions if r.priority == 2]
    for i, a in enumerate(special):
        for b in special[i + 1 :]:
            w = min(a.rect.x1, b.rect.x1) - max(a.rect.x0, b.rect.x0)
            h = min(a.rect.y1, b.rect.y1) - max(a.rect.y0, b.rect.y0)
            if w > 0 and h > 0:
                raise GeometryError(f"portal regions {a.sector} and {b.sector} overlap")
    for r in special:
        info = sectors.get(r.sector or "")
        if info is None or info.kind not in ("lift", "pad"):
            continue
        # Lifts straddle the wall strip, so check only the part inside the lower room.
        room = spec.room(spec.portal(info.portal or "").a)
        if info.kind == "lift":
            p = spec.portal(info.portal or "")
            ra, rb = spec.room(p.a), spec.room(p.b)
            room = ra if ra.floor < rb.floor else rb
        inside = Rect(room.x0, room.y0, room.x1, room.y1)
        clip = Rect(
            max(r.rect.x0, inside.x0), max(r.rect.y0, inside.y0),
            min(r.rect.x1, inside.x1), min(r.rect.y1, inside.y1),
        )  # fmt: skip
        need = PLAYER_RADIUS * 2 + WALL // 2
        if info.kind == "pad" and clip != r.rect:
            raise GeometryError(f"{r.sector} does not fit inside room {room.id}")
        if info.kind == "lift" and min(inside.x1 - inside.x0, inside.y1 - inside.y0) < (
            LIFT_DEPTH + WALL + need
        ):
            raise GeometryError(f"{r.sector}: room {room.id} too small for a lift platform")


def _grid(regions: Sequence[Region], marks: Sequence[WallMark]) -> tuple[list[int], list[int]]:
    xs = {v for r in regions for v in (r.rect.x0, r.rect.x1)}
    ys = {v for r in regions for v in (r.rect.y0, r.rect.y1)}
    for m in marks:
        (xs if m.axis == "y" else ys).update((m.lo, m.hi))
        (xs if m.axis == "x" else ys).add(m.at)
    return sorted(xs), sorted(ys)


@dataclass(frozen=True)
class _Edge:
    axis: Literal["x", "y"]
    at: int
    lo: int
    hi: int
    low: str | None  # sector on the low-coordinate side (left of x-wall / below y-wall)
    high: str | None


def _edges(regions: Sequence[Region], marks: Sequence[WallMark]) -> list[_Edge]:
    xs, ys = _grid(regions, marks)
    nx, ny = len(xs) - 1, len(ys) - 1
    owner = [
        [_owner(regions, (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2) for j in range(ny)]
        for i in range(nx)
    ]
    edges = []
    for i in range(nx + 1):
        for j in range(ny):
            low = owner[i - 1][j] if i > 0 else None
            high = owner[i][j] if i < nx else None
            if low != high:
                edges.append(_Edge("x", xs[i], ys[j], ys[j + 1], low, high))
    for j in range(ny + 1):
        for i in range(nx):
            low = owner[i][j - 1] if j > 0 else None
            high = owner[i][j] if j < ny else None
            if low != high:
                edges.append(_Edge("y", ys[j], xs[i], xs[i + 1], low, high))
    return edges


def _front_back(e: _Edge, sectors: dict[str, SectorInfo]) -> tuple[str, str | None]:
    if e.low is None or e.high is None:
        front = e.high if e.low is None else e.low
        assert front is not None
        return front, None
    low_room = sectors[e.low].kind == "room"
    high_room = sectors[e.high].kind == "room"
    if low_room != high_room:
        return (e.low, e.high) if low_room else (e.high, e.low)
    return (e.low, e.high) if e.low < e.high else (e.high, e.low)


def _attrs(
    e: _Edge, front: str, back: str | None, sectors: dict[str, SectorInfo],
    marks: Sequence[WallMark],
) -> tuple[Special | None, str | None, str | None]:  # fmt: skip
    if back is not None:
        face = sectors[back].face
        if face is not None and sectors[front].kind == "room" and (
            front not in sectors[back].face_skip
        ):
            return face, None, sectors[back].portal
        return None, None, None
    for m in marks:
        if m.sector == front and m.covers(e.axis, e.at, e.lo, e.hi):
            return m.special, m.texture, m.source
    return None, None, None


@dataclass(frozen=True)
class _LineAttrs:
    low: str | None
    high: str | None
    front: str
    back: str | None
    special: Special | None
    texture: str | None
    source: str | None


def _build_lines(
    regions: Sequence[Region], sectors: dict[str, SectorInfo], marks: Sequence[WallMark]
) -> list[Line]:
    keyed = []
    for e in _edges(regions, marks):
        front, back = _front_back(e, sectors)
        special, texture, source = _attrs(e, front, back, sectors, marks)
        keyed.append((e, _LineAttrs(e.low, e.high, front, back, special, texture, source)))
    keyed.sort(key=lambda item: (item[0].axis, item[0].at, item[0].lo))

    merged: list[tuple[_Edge, _LineAttrs]] = []
    for e, attrs in keyed:
        if merged:
            prev, prev_attrs = merged[-1]
            if (prev.axis, prev.at, prev.hi) == (e.axis, e.at, e.lo) and prev_attrs == attrs:
                merged[-1] = (_Edge(e.axis, e.at, prev.lo, e.hi, e.low, e.high), attrs)
                continue
        merged.append((e, attrs))

    lines = []
    for e, at in merged:
        if e.axis == "x":  # (at, lo) -> (at, hi) has the high-x side on its right
            a, b = (e.at, e.lo), (e.at, e.hi)
        else:  # (lo, at) -> (hi, at) has the low-y side on its right
            a, b = (e.lo, e.at), (e.hi, e.at)
        front_is_right = (at.front == e.high) if e.axis == "x" else (at.front == e.low)
        v1, v2 = (a, b) if front_is_right else (b, a)
        lines.append(Line(v1, v2, at.front, at.back, at.special, at.texture, at.source))
    return lines


def _place_mark(
    regions: Sequence[Region], sector: str, pos: tuple[int, int], special: Special,
    source: str,
) -> WallMark:  # fmt: skip
    """Put a WALL_MARK-long special on the nearest one-sided wall run of `sector`."""
    runs: list[tuple[float, WallMark]] = []
    edges = sorted(
        (e for e in _edges(regions, ()) if (e.low is None) != (e.high is None)),
        key=lambda e: (e.axis, e.at, e.lo),
    )
    merged: list[_Edge] = []
    for e in edges:
        if merged and (merged[-1].axis, merged[-1].at, merged[-1].hi, merged[-1].low,
                       merged[-1].high) == (e.axis, e.at, e.lo, e.low, e.high):  # fmt: skip
            merged[-1] = _Edge(e.axis, e.at, merged[-1].lo, e.hi, e.low, e.high)
        else:
            merged.append(e)
    px, py = pos
    for e in merged:
        if sector not in (e.low, e.high) or e.hi - e.lo < WALL_MARK:
            continue
        along, across = (py, px) if e.axis == "x" else (px, py)
        centre = min(max(along, e.lo + WALL_MARK / 2), e.hi - WALL_MARK / 2)
        dist = abs(across - e.at) + abs(along - centre)
        lo = int(centre - WALL_MARK / 2)
        runs.append((dist, WallMark(e.axis, e.at, lo, lo + WALL_MARK, sector, special,
                                    TEX_SWITCH, source)))  # fmt: skip
    if not runs:
        raise GeometryError(f"{source}: no wall of {sector} is long enough for a switch")
    return min(runs, key=lambda r: r[0])[1]

