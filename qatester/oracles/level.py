"""Geometry queries over the intended level, shared by the oracles.

Built from the intent-only ground truth, so nothing here can see injected defects.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qatester.mapgen.groundtruth import GroundTruth

if TYPE_CHECKING:
    from qatester.oracles.base import AttemptConfig

Rect = tuple[float, float, float, float]
Segment = tuple[float, float, float, float]


def rect_distance(x: float, y: float, r: Rect) -> float:
    """Distance from a point to a rectangle (0 inside)."""
    dx = max(r[0] - x, 0.0, x - r[2])
    dy = max(r[1] - y, 0.0, y - r[3])
    return math.hypot(dx, dy)


def segment_distance(x: float, y: float, s: Segment) -> float:
    ax, ay, bx, by = s
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(x - (ax + t * dx), y - (ay + t * dy))


def segments_cross(p: Segment, q: Segment) -> bool:
    """True when the two segments properly intersect or touch."""

    def orient(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    def on(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> bool:
        return min(ax, bx) <= cx <= max(ax, bx) and min(ay, by) <= cy <= max(ay, by)

    a, b, c, d = (p[0], p[1]), (p[2], p[3]), (q[0], q[1]), (q[2], q[3])
    d1 = orient(*c, *d, *a)
    d2 = orient(*c, *d, *b)
    d3 = orient(*a, *b, *c)
    d4 = orient(*a, *b, *d)
    if ((d1 > 0) != (d2 > 0)) and d1 and d2 and ((d3 > 0) != (d4 > 0)) and d3 and d4:
        return True
    return (
        (d1 == 0 and on(*c, *d, *a)) or (d2 == 0 and on(*c, *d, *b))
        or (d3 == 0 and on(*a, *b, *c)) or (d4 == 0 and on(*a, *b, *d))
    )  # fmt: skip


def facing(x: float, y: float, angle: float, tx: float, ty: float, cone: float) -> bool:
    """Whether a target point lies within +-`cone` degrees of the view direction."""
    want = math.degrees(math.atan2(ty - y, tx - x))
    return abs((want - angle + 180.0) % 360.0 - 180.0) <= cone


@dataclass(frozen=True)
class Portal:
    id: str
    a: str
    b: str
    kind: str
    requires: str | None
    switch: str | None
    area: Rect
    dest: tuple[float, float] | None


class Level:
    def __init__(self, truth: GroundTruth) -> None:
        d: dict[str, Any] = truth.data
        if "defects" in d:
            raise PermissionError("oracles must be given GroundTruth.intent_only()")
        self.truth = truth
        self.rooms: dict[str, Rect] = {r["id"]: tuple(r["rect"]) for r in d["rooms"]}
        self.room_sector: dict[str, int] = {r["id"]: r["sector"] for r in d["rooms"]}
        self.sectors: list[dict[str, Any]] = d["sectors"]
        self.walls: list[Segment] = [tuple(w) for w in d["walls"]]
        self.portals: dict[str, Portal] = {
            p["id"]: Portal(p["id"], p["a"], p["b"], p["kind"], p["requires"], p["switch"],
                            tuple(p["area"]), tuple(p["dest"]) if p["dest"] else None)
            for p in d["portals"]
        }  # fmt: skip
        self.switch_lines: dict[str, Segment] = {
            k: tuple(v) for k, v in d["switch_lines"].items()
        }
        self.things: list[dict[str, Any]] = d["things"]
        self.monster_count: int = d["monster_count"]
        self.spawn_room: str = d["spawn"]["room"]
        goal_room = d["goal"]["room"]
        self.goal_sector: int | None = self.room_sector[goal_room] if goal_room else None
        # portal id -> index of its own sector (door, lift or pad), if any
        self.portal_sector: dict[str, int] = {
            s["portal"]: s["index"] for s in self.sectors if s["portal"] is not None
        }

    def sector_at(self, x: float, y: float) -> int:
        return self.truth.sector_at(x, y)

    def is_room_sector(self, s: int) -> bool:
        return s >= 0 and self.sectors[s]["kind"] == "room"

    def rooms_of(self, s: int) -> list[str]:
        return list(self.sectors[s]["rooms"]) if s >= 0 else []

    def inside_level(self, x: float, y: float, margin: float) -> bool:
        return any(
            r[0] - margin <= x <= r[2] + margin and r[1] - margin <= y <= r[3] + margin
            for r in self.rooms.values()
        )

    def depth_outside(self, x: float, y: float) -> float:
        return min(rect_distance(x, y, r) for r in self.rooms.values())

    def portals_between(self, s_from: int, s_to: int) -> list[Portal]:
        """Portals leading from room sector `s_from` to room sector `s_to`."""
        out = []
        for p in self.portals.values():
            a, b = self.room_sector[p.a], self.room_sector[p.b]
            if (a, b) == (s_from, s_to) or (p.kind != "teleport" and (b, a) == (s_from, s_to)):
                out.append(p)
        return out

    def on_teleport_pad(self, x: float, y: float, margin: float) -> bool:
        return any(
            p.kind == "teleport" and rect_distance(x, y, p.area) <= margin
            for p in self.portals.values()
        )


class TraversalTracker:
    """Follows the player's room sector and reports the portals they pass through.

    Non-room sectors (door, lift, pad) and the void are transit: the player is still
    "in" the last room sector until they enter another one.
    """

    def __init__(self, level: Level) -> None:
        self.level = level
        self.current: int | None = None
        self.recent: deque[tuple[float, float]] = deque(maxlen=8)

    def reset(self) -> None:
        self.current = None
        self.recent.clear()

    def update(self, x: float, y: float) -> list[str]:
        s = self.level.sector_at(x, y)
        self.recent.append((x, y))
        if not self.level.is_room_sector(s):
            return []
        if self.current is None or s == self.current:
            self.current = s
            return []
        prev, self.current = self.current, s
        candidates = self.level.portals_between(prev, s)
        if not candidates:
            return []
        if len(candidates) == 1:
            return [candidates[0].id]
        # Several portals join the same rooms: pick the one the recent path passed nearest.
        best = min(
            candidates,
            key=lambda p: min(rect_distance(px, py, p.area) for px, py in self.recent),
        )
        return [best.id]


def ray_hits_rect(x: float, y: float, angle: float, length: float, r: Rect) -> bool:
    """Whether the view ray of `length` from (x, y) touches the rectangle."""
    ex = x + length * math.cos(math.radians(angle))
    ey = y + length * math.sin(math.radians(angle))
    if any(r[0] <= px <= r[2] and r[1] <= py <= r[3] for px, py in ((x, y), (ex, ey))):
        return True
    ray = (x, y, ex, ey)
    edges = [(r[0], r[1], r[2], r[1]), (r[2], r[1], r[2], r[3]),
             (r[2], r[3], r[0], r[3]), (r[0], r[3], r[0], r[1])]  # fmt: skip
    return any(segments_cross(ray, e) for e in edges)


def ray_hits_segment(x: float, y: float, angle: float, length: float, s: Segment) -> bool:
    ex = x + length * math.cos(math.radians(angle))
    ey = y + length * math.sin(math.radians(angle))
    return segments_cross((x, y, ex, ey), s)


@dataclass
class _Pending:
    tic: int
    sector: int | None  # the portal's own sector, whose movement proves it worked
    floor: float = 0.0
    ceiling: float = 0.0


class AttemptTracker:
    """Proper attempts to use an exit that visibly failed (docs/decisions.md D25).

    - door / lift: USE whose ray (Doom's 64 mu use range) hits the portal from outside
      it, holding the key for a locked door; then the portal's sector does not move
      within the window. (Standing on a lift and pressing USE does nothing in Doom, and
      is no attempt: you leave a raised lift by walking off it; docs/decisions.md D36.)
    - switch-operated door: USE whose ray hits its switch; the door does not move.
    - open portal: pushing forward through the opening itself (lined up inside it,
      facing it, within reach) for `throat_push_tics` without making progress.
    - teleport: standing on the pad without arriving within the window.
    Attempts on a portal that already moved this episode (an open door) are ignored.
    """

    def __init__(self, level: Level, cfg: AttemptConfig) -> None:
        self.level = level
        self.use_range = cfg.use_range_mu
        self.window = cfg.window_tics
        self.throat_mu = cfg.throat_mu
        self.throat_push_tics = cfg.throat_push_tics
        self.stall = cfg.stall_mu_per_tic
        self.cone = cfg.face_cone_deg
        self.radius = cfg.player_radius_mu
        self.reset()

    def reset(self) -> None:
        self.initial: tuple[tuple[float, ...], tuple[float, ...]] | None = None
        self.pending: dict[str, _Pending] = {}
        self.throat: dict[str, int] = {}
        self.traversed_at: dict[str, int] = {}  # portal -> tic of its latest traversal
        self.prev: tuple[int, float, float] | None = None

    def _moved(self, floors: tuple[float, ...], ceilings: tuple[float, ...], sector: int,
               floor: float, ceiling: float) -> bool:  # fmt: skip
        return abs(floors[sector] - floor) > 1.0 or abs(ceilings[sector] - ceiling) > 1.0

    def _already_moved(self, floors: tuple[float, ...], ceilings: tuple[float, ...],
                       sector: int) -> bool:  # fmt: skip
        if self.initial is None or sector >= len(floors):
            return False
        return self._moved(floors, ceilings, sector, self.initial[0][sector],
                           self.initial[1][sector])  # fmt: skip

    def _aligned(self, x: float, y: float, area: Rect) -> bool:
        if area[2] - area[0] <= area[3] - area[1]:  # vertical wall: opening spans y
            along = area[1] + self.radius <= y <= area[3] - self.radius
        else:
            along = area[0] + self.radius <= x <= area[2] - self.radius
        return along and rect_distance(x, y, area) <= self.throat_mu

    def update(self, step: Any, traversed_now: list[str]) -> list[str]:
        """Feed one StepRecord; returns the portals whose attempt failed at this step."""
        lv = self.level
        floors, ceilings = step.sector_floors, step.sector_ceilings
        if self.initial is None and floors:
            self.initial = (floors, ceilings)
        for pid in traversed_now:
            self.traversed_at[pid] = step.tic
            self.pending.pop(pid, None)
        failed: list[str] = []
        x, y, angle = step.x, step.y, step.angle

        if step.use:
            for p in lv.portals.values():
                if p.id in self.pending or p.kind in ("open", "teleport"):
                    continue
                if p.switch is not None:
                    if not ray_hits_segment(x, y, angle, self.use_range, lv.switch_lines[p.switch]):
                        continue
                elif rect_distance(x, y, p.area) == 0.0:
                    continue  # standing on the lift (or in the doorway): USE cannot work it
                elif not ray_hits_rect(x, y, angle, self.use_range, p.area):
                    continue
                elif p.requires is not None and p.requires not in step.keys:
                    continue  # no key: the door is right to stay shut
                sector = lv.portal_sector.get(p.id)
                if sector is not None and self._already_moved(floors, ceilings, sector):
                    continue
                pend = _Pending(step.tic, sector)
                if sector is not None and sector < len(floors):
                    pend.floor, pend.ceiling = floors[sector], ceilings[sector]
                self.pending[p.id] = pend

        here = lv.sector_at(x, y)
        moved = 0.0 if self.prev is None else math.hypot(x - self.prev[1], y - self.prev[2])
        dt = 1 if self.prev is None else max(1, step.tic - self.prev[0])
        for p in lv.portals.values():
            if p.kind == "teleport" and lv.portal_sector.get(p.id) == here and (
                p.id not in self.pending
            ):
                self.pending[p.id] = _Pending(step.tic, None)
            if p.kind != "open":
                continue
            cx, cy = (p.area[0] + p.area[2]) / 2, (p.area[1] + p.area[3]) / 2
            pushing = ("forward" in step.move_dirs and self._aligned(x, y, p.area)
                       and facing(x, y, angle, cx, cy, self.cone)
                       and moved < self.stall * dt)  # fmt: skip
            if not pushing:
                self.throat.pop(p.id, None)
                continue
            self.throat[p.id] = self.throat.get(p.id, 0) + dt
            if self.throat[p.id] >= self.throat_push_tics:
                failed.append(p.id)
                self.throat[p.id] = 0

        for pid, pend in list(self.pending.items()):
            # Only a traversal after the attempt proves it worked: a one-way lift was
            # traversed on the way down, which says nothing about the way back up.
            worked = self.traversed_at.get(pid, -1) >= pend.tic or (
                pend.sector is not None and pend.sector < len(floors)
                and self._moved(floors, ceilings, pend.sector, pend.floor, pend.ceiling)
            )  # fmt: skip
            if worked:
                del self.pending[pid]
            elif step.tic - pend.tic >= self.window:
                failed.append(pid)
                del self.pending[pid]
        self.prev = (step.tic, x, y)
        return failed
