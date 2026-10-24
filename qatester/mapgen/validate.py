"""Correctness invariants asserted before a WAD is written (PLAN: Critical correctness
invariants). Checks run on the emitted TEXTMAP, not on the compiler's intermediate
structures, so a bug in emission is caught too.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection

from qatester.mapgen import udmf
from qatester.mapgen.geometry import (
    PLAYER_HEIGHT,
    PLAYER_RADIUS,
    CompiledMap,
    GeometryError,
)
from qatester.mapgen.spec import WALL_THING_TYPES

PROBE = 0.5  # how far either side of a linedef to probe for its sectors
MIN_VERTEX_GAP = 1.0


def _seg_dist(px: float, py: float, a: udmf.Vertex, b: udmf.Vertex) -> float:
    dx, dy = b.x - a.x, b.y - a.y
    t = max(0.0, min(1.0, ((px - a.x) * dx + (py - a.y) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (a.x + t * dx), py - (a.y + t * dy))


def check_sidedefs(tm: udmf.TextMap) -> list[str]:
    """1. Every linedef's front sidedef exists and references a valid sector."""
    out = []
    for i, ld in enumerate(tm.linedefs):
        for side in (ld.sidefront, ld.sideback):
            if side is None:
                continue
            if not 0 <= side < len(tm.sidedefs):
                out.append(f"linedef {i}: sidedef {side} does not exist")
            elif not 0 <= tm.sidedefs[side].sector < len(tm.sectors):
                out.append(f"linedef {i}: sidedef {side} has invalid sector")
    return out


def check_two_sided_flags(tm: udmf.TextMap, allow_blocking: Collection[int] = ()) -> list[str]:
    """2. Two-sided linedefs have `twosided` and not `blocking` (except listed defects)."""
    out = []
    for i, ld in enumerate(tm.linedefs):
        two = ld.sideback is not None
        if two != ("twosided" in ld.flags):
            out.append(f"linedef {i}: twosided flag does not match its sidedefs")
        if two and "blocking" in ld.flags and i not in allow_blocking:
            out.append(f"linedef {i}: two-sided linedef is blocking")
    return out


def check_vertices(tm: udmf.TextMap) -> list[str]:
    """3. No two vertices closer than 1 mu; no zero-length linedefs."""
    out = []
    vs = tm.vertices
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            if math.hypot(vs[i].x - vs[j].x, vs[i].y - vs[j].y) < MIN_VERTEX_GAP:
                out.append(f"vertices {i} and {j} are closer than {MIN_VERTEX_GAP} mu")
    for i, ld in enumerate(tm.linedefs):
        a, b = vs[ld.v1], vs[ld.v2]
        if (a.x, a.y) == (b.x, b.y):
            out.append(f"linedef {i} has zero length")
    return out


def check_winding(cm: CompiledMap, tm: udmf.TextMap) -> list[str]:
    """4. The front side faces the sector on the right of v1 -> v2, the back the left."""
    keys = list(cm.sectors)
    out = []
    for i, ld in enumerate(tm.linedefs):
        a, b = tm.vertices[ld.v1], tm.vertices[ld.v2]
        length = math.hypot(b.x - a.x, b.y - a.y)
        nx, ny = (b.y - a.y) / length, -(b.x - a.x) / length  # right-hand normal
        mx, my = (a.x + b.x) / 2, (a.y + b.y) / 2
        right = cm.owner_at(mx + PROBE * nx, my + PROBE * ny)
        left = cm.owner_at(mx - PROBE * nx, my - PROBE * ny)
        front = keys[tm.sidedefs[ld.sidefront].sector]
        back = None if ld.sideback is None else keys[tm.sidedefs[ld.sideback].sector]
        if right != front or left != back:
            out.append(
                f"linedef {i} ({a.x},{a.y})->({b.x},{b.y}): front={front} back={back} "
                f"but right={right} left={left}"
            )
    return out


def check_closed(tm: udmf.TextMap) -> list[str]:
    """5. Every sector is closed: its boundary edges, oriented with the sector on the
    right, enter and leave every vertex equally often."""
    flow: dict[int, Counter[int]] = {}
    for ld in tm.linedefs:
        sides = [(ld.sidefront, ld.v1, ld.v2), (ld.sideback, ld.v2, ld.v1)]
        for side, src, dst in sides:
            if side is None:
                continue
            c = flow.setdefault(tm.sidedefs[side].sector, Counter())
            c[src] += 1
            c[dst] -= 1
    out = [
        f"sector {s} is not closed at vertex {v}"
        for s, c in sorted(flow.items())
        for v, n in sorted(c.items())
        if n
    ]
    out += [f"sector {s} has no linedefs" for s in range(len(tm.sectors)) if s not in flow]
    return out


def check_heights(cm: CompiledMap) -> list[str]:
    """6. Ceiling exceeds floor by >= 56 mu wherever the player must stand (doors: open)."""
    return [
        f"sector {s.key}: only {s.ceiling - s.floor} mu between floor and ceiling"
        for s in cm.sectors.values()
        if s.ceiling - s.floor < PLAYER_HEIGHT
    ]


def check_things(cm: CompiledMap, tm: udmf.TextMap) -> list[str]:
    """7. No thing within 16 mu of a wall, and each thing stands in its own room."""
    walls = [
        (tm.vertices[ld.v1], tm.vertices[ld.v2]) for ld in tm.linedefs if ld.sideback is None
    ]
    placed = [
        (t.id, t.pos, cm.room_sector(t.room))
        for t in cm.spec.things
        if t.type not in WALL_THING_TYPES
    ]
    pads = [s.key for s in cm.sectors.values() if s.kind == "pad"]
    for portal_id, (x, y) in cm.teleport_dests.items():
        dest_room = cm.room_sector(cm.spec.portal(portal_id).b)
        placed.append((f"teleport_dest:{portal_id}", (int(x), int(y)), dest_room))
    out = []
    for name, (x, y), sector in placed:
        owner = cm.owner_at(x, y)
        if owner != sector:
            out.append(f"{name} at ({x},{y}) stands in {owner}, not {sector}")
        near = min((_seg_dist(x, y, a, b) for a, b in walls), default=math.inf)
        if near < PLAYER_RADIUS:
            out.append(f"{name} at ({x},{y}) is {near:.1f} mu from a wall")
        for pad in pads:
            r = next(rg.rect for rg in cm.regions if rg.sector == pad)
            if r.x0 - 2 * PLAYER_RADIUS < x < r.x1 + 2 * PLAYER_RADIUS and (
                r.y0 - 2 * PLAYER_RADIUS < y < r.y1 + 2 * PLAYER_RADIUS
            ):
                out.append(f"{name} at ({x},{y}) is on or next to teleport pad {pad}")
    return out


def problems(
    cm: CompiledMap, tm: udmf.TextMap, allow_blocking: Collection[int] = ()
) -> list[str]:
    out = check_sidedefs(tm)
    if out:  # later checks index sidedefs
        return out
    return (
        check_two_sided_flags(tm, allow_blocking)
        + check_vertices(tm)
        + check_winding(cm, tm)
        + check_closed(tm)
        + check_heights(cm)
        + check_things(cm, tm)
    )


def validate(cm: CompiledMap, tm: udmf.TextMap, allow_blocking: Collection[int] = ()) -> None:
    found = problems(cm, tm, allow_blocking)
    if found:
        raise GeometryError(f"{cm.spec.name}: " + "; ".join(found))
