"""Hand-written StepRecord traces for oracle tests (PLAN M4: synthetic traces).

A Trace walks a virtual player through a built map's intended geometry, emitting one
record per `frame_skip` tics with live sector heights like the engine would report:
doors start closed and open when told to, lifts sit at their top until lowered.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import replace

from qatester.mapgen.groundtruth import GroundTruth
from qatester.oracles.base import EpisodeContext, StepRecord

WALK = 8.0  # mu per tic


class Trace:
    def __init__(self, truth: GroundTruth, start: tuple[float, float] | None = None,
                 angle: float = 0.0, frame_skip: int = 4, health: float = 100.0) -> None:  # fmt: skip
        self.truth = truth
        self.frame_skip = frame_skip
        sectors = truth.data["sectors"]
        self.floors = [float(s["floor"]) for s in sectors]
        self.ceilings = [float(s["floor"] if s["kind"] == "door" else s["ceiling"])
                         for s in sectors]  # fmt: skip
        sx, sy = start if start is not None else truth.data["spawn"]["pos"]
        self.x, self.y, self.angle = float(sx), float(sy), angle
        self.tic = 0
        self.health = health
        self.keys: frozenset[str] = frozenset()
        self.records: list[StepRecord] = []

    # --- state ---------------------------------------------------------------------
    def _z(self) -> float:
        s = self.truth.sector_at(self.x, self.y)
        return self.floors[s] if s >= 0 else 0.0

    def _sector_of_portal(self, portal: str) -> int:
        return next(s["index"] for s in self.truth.data["sectors"] if s["portal"] == portal)

    def emit(self, move_dirs: Iterable[str] = (), use: bool = False, **extra: object) -> Trace:
        rec = StepRecord(
            tic=self.tic, x=self.x, y=self.y, z=self._z(), angle=self.angle,
            health=self.health, keys=self.keys, move_dirs=frozenset(move_dirs), use=use,
            sector_floors=tuple(self.floors), sector_ceilings=tuple(self.ceilings),
        )  # fmt: skip
        self.records.append(replace(rec, **extra) if extra else rec)  # type: ignore[arg-type]
        self.tic += self.frame_skip
        return self

    # --- actions -------------------------------------------------------------------
    def face(self, angle: float) -> Trace:
        self.angle = angle % 360.0
        return self

    def face_point(self, x: float, y: float) -> Trace:
        return self.face(math.degrees(math.atan2(y - self.y, x - self.x)))

    def walk_to(self, x: float, y: float, speed: float = WALK) -> Trace:
        self.face_point(x, y)
        step = speed * self.frame_skip
        while math.hypot(x - self.x, y - self.y) > 1e-6:
            d = math.hypot(x - self.x, y - self.y)
            k = min(1.0, step / d)
            self.x += (x - self.x) * k
            self.y += (y - self.y) * k
            self.emit({"forward"})
        return self

    def path(self, *points: tuple[float, float]) -> Trace:
        for p in points:
            self.walk_to(*p)
        return self

    def wait(self, tics: int) -> Trace:
        for _ in range(max(1, tics // self.frame_skip)):
            self.emit()
        return self

    def push(self, tics: int, dirs: tuple[str, ...] = ("forward",)) -> Trace:
        """Command movement that the world refuses: no displacement."""
        for i in range(max(1, tics // self.frame_skip)):
            self.emit({dirs[i % len(dirs)]})
        return self

    def use(self) -> Trace:
        return self.emit(use=True)

    def teleport(self, x: float, y: float, angle: float | None = None) -> Trace:
        self.x, self.y = x, y
        if angle is not None:
            self.angle = angle
        return self.emit({"forward"})

    def open_door(self, portal: str) -> Trace:
        s = self._sector_of_portal(portal)
        self.ceilings[s] = float(self.truth.data["sectors"][s]["ceiling"])
        return self

    def set_floor(self, sector: int, floor: float) -> Trace:
        self.floors[sector] = floor
        return self

    def pick_up(self, health: float | None = None, key: str | None = None) -> Trace:
        if health is not None:
            self.health = health
        if key is not None:
            self.keys = self.keys | {key}
        return self.emit()

    def damage(self, health: float) -> Trace:
        self.health = health
        return self.emit()

    def until(self, tic: int) -> Trace:
        while self.tic < tic:
            self.emit()
        return self

    def end(self, truncated: bool = True) -> list[StepRecord]:
        if not self.records:
            self.emit()
        self.records[-1] = replace(self.records[-1], done=True, truncated=truncated)
        return self.records


def episodes(truth: GroundTruth, *traces: list[StepRecord],
             frame_skip: int = 4) -> list[tuple[EpisodeContext, list[StepRecord]]]:  # fmt: skip
    """Wrap traces as consecutive episodes of one run, with the intent-only view."""
    view = truth.intent_only()
    return [
        (EpisodeContext(truth.data["map"], i, i, view, frame_skip), records)
        for i, records in enumerate(traces)
    ]


# --- routing through the intended level ---------------------------------------------

def room_rect(truth: GroundTruth, room: str) -> tuple[float, float, float, float]:
    r = next(r for r in truth.data["rooms"] if r["id"] == room)
    return tuple(r["rect"])  # type: ignore[return-value]


def portal_info(truth: GroundTruth, pid: str) -> dict[str, object]:
    return next(p for p in truth.data["portals"] if p["id"] == pid)


def approach(truth: GroundTruth, room: str, pid: str, dist: float = 48.0) -> tuple[float, float]:
    """A point `dist` inside `room` straight in front of portal `pid`'s opening."""
    p = portal_info(truth, pid)
    x0, y0, x1, y1 = p["area"]  # type: ignore[misc]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx0, ry0, rx1, ry1 = room_rect(truth, room)
    rcx, rcy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
    if x1 - x0 <= y1 - y0:  # thin in x: a vertical wall at x = cx
        return (cx - dist if rcx < cx else cx + dist), cy
    return cx, (cy - dist if rcy < cy else cy + dist)


def route(tr: Trace, frm: str, to: str) -> Trace:
    """Walk from room `frm` to room `to` through open portals and plain doors only."""
    portals = [p for p in tr.truth.data["portals"]
               if p["kind"] in ("open", "door") and p["switch"] is None]  # fmt: skip
    prev: dict[str, tuple[str, str] | None] = {frm: None}
    queue = [frm]
    while queue:
        room = queue.pop(0)
        for p in portals:
            for a, b in ((p["a"], p["b"]), (p["b"], p["a"])):
                if a == room and b not in prev:
                    prev[b] = (room, p["id"])
                    queue.append(b)
    if to not in prev:
        raise ValueError(f"no open/door route {frm} -> {to}")
    hops: list[tuple[str, str, str]] = []
    room = to
    while prev[room] is not None:
        before, pid = prev[room]  # type: ignore[misc]
        hops.append((before, pid, room))
        room = before
    for a, pid, b in reversed(hops):
        tr.walk_to(*approach(tr.truth, a, pid))
        p = portal_info(tr.truth, pid)
        if p["kind"] == "door":
            x0, y0, x1, y1 = p["area"]  # type: ignore[misc]
            tr.face_point((x0 + x1) / 2, (y0 + y1) / 2).use().open_door(pid).wait(40)
        tr.walk_to(*approach(tr.truth, b, pid))
    return tr
