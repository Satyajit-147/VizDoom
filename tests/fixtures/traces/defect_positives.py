"""A hand-authored positive trace for every injected defect (PLAN M4 acceptance: "every
injected defect detected in its hand-authored positive trace").

One generator per defect kind, driven by the defect's ground-truth entry, acting out what
the engine does on the buggy build (verified in tests/test_defects_engine.py). Fixtures
may read the defect list; oracles never can.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from tracebuilder import Trace, approach, portal_info, room_rect, route

from qatester.mapgen.groundtruth import GroundTruth
from qatester.oracles.base import StepRecord

Episodes = list[list[StepRecord]]
EPISODE_TICS = 10_500
LOOP_STANDOFF = 56.0  # matches qatester.mapgen.defects


def _room_of_thing(t: GroundTruth, thing: str) -> str:
    return str(next(x["room"] for x in t.data["things"] if x["id"] == thing))


def _sector(t: GroundTruth, room: str) -> int:
    return int(next(r["sector"] for r in t.data["rooms"] if r["id"] == room))


def _centre(rect: Any) -> tuple[float, float]:
    return (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2


def missing_blocking(t: GroundTruth, d: dict[str, Any]) -> Episodes:
    x0, y0, x1, y1 = room_rect(t, d["target"])
    lo, hi = d["params"]["segment"]
    mid = (lo + hi) / 2
    inward, outward = {
        "north": ((mid, y1 - 48), (mid, y1 + 112)), "south": ((mid, y0 + 48), (mid, y0 - 112)),
        "east": ((x1 - 48, mid), (x1 + 112, mid)), "west": ((x0 + 48, mid), (x0 - 112, mid)),
    }[d["params"]["wall"]]  # fmt: skip
    return [Trace(t, start=inward).path(outward, inward).end(False)]


def geometry_trap(t: GroundTruth, d: dict[str, Any]) -> Episodes:
    room = d["target"]
    p = next(p for p in t.data["portals"] if p["kind"] == "open" and room in (p["a"], p["b"]))
    outside = p["b"] if p["a"] == room else p["a"]
    tr = Trace(t, start=approach(t, outside, p["id"], 96))
    tr.set_floor(_sector(t, room), d["position"][2])
    return [tr.walk_to(*approach(t, room, p["id"], 48)).wait(200).end(False)]


def wrong_tag(t: GroundTruth, d: dict[str, Any]) -> Episodes:
    door = next(p for p in t.data["portals"] if p["switch"] == d["target"])
    near = door["a"] if door["b"] in d["affected_rooms"] else door["b"]
    switch_room = _room_of_thing(t, d["target"])
    sx0, sy0, sx1, sy1 = t.data["switch_lines"][d["target"]]
    rx0, ry0, _rx1, _ry1 = room_rect(t, switch_room)
    mx, my = (sx0 + sx1) / 2, (sy0 + sy1) / 2
    if sx0 == sx1:  # vertical wall
        stand = (mx + 48 if mx <= rx0 + 1 else mx - 48, my)
    else:
        stand = (mx, my + 48 if my <= ry0 + 1 else my - 48)
    runs = []
    for _ in range(6):
        tr = Trace(t, start=stand).face_point(mx, my).use().wait(60)
        route(tr, switch_room, near).walk_to(*approach(t, near, door["id"], 48))
        tr.face_point(*_centre(door["area"])).use().push(60)
        runs.append(tr.end(False))
    return runs


def wrong_special(t: GroundTruth, d: dict[str, Any]) -> Episodes:
    p = portal_info(t, d["target"])
    tr = Trace(t, start=approach(t, p["a"], p["id"], 56)).face_point(*_centre(p["area"]))
    tr.use().open_door(p["id"]).wait(40)
    return [tr.walk_to(*approach(t, p["b"], p["id"], 96)).end(False)]


def one_way_lift(t: GroundTruth, d: dict[str, Any]) -> Episodes:
    p = portal_info(t, d["target"])
    lower = d["affected_rooms"][0]
    ax0, ay0, ax1, ay1 = p["area"]  # type: ignore[misc]
    cx, cy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
    rcx, rcy = _centre(room_rect(t, lower))
    if ax1 - ax0 >= ay1 - ay0:  # horizontal wall: the lift reaches along y
        stand = (cx, ay1 + 40 if rcy > cy else ay0 - 40)
    else:
        stand = (ax1 + 40 if rcx > cx else ax0 - 40, cy)
    runs = []
    for _ in range(2):
        tr = Trace(t, start=_centre(room_rect(t, lower))).walk_to(*stand)
        tr.until(int(EPISODE_TICS * 0.65)).face_point(cx, cy).push(300).use().push(200)
        runs.append(tr.until(EPISODE_TICS).end())
    return runs


def teleport_loop(t: GroundTruth, d: dict[str, Any]) -> Episodes:
    a, b = portal_info(t, d["target"]), portal_info(t, d["params"]["partner"])

    def arrival(p: dict[str, Any]) -> tuple[float, float]:
        px, py = _centre(p["area"])
        rx, ry = _centre(room_rect(t, p["a"]))
        n = math.hypot(rx - px, ry - py)
        return px + LOOP_STANDOFF * (rx - px) / n, py + LOOP_STANDOFF * (ry - py) / n

    def nudge(p: dict[str, Any], frm: tuple[float, float]) -> tuple[float, float]:
        px, py = _centre(p["area"])
        n = math.hypot(px - frm[0], py - frm[1])
        return frm[0] + 20 * (px - frm[0]) / n, frm[1] + 20 * (py - frm[1]) / n

    start = arrival(a)
    tr = Trace(t, start=start).walk_to(*nudge(a, start))
    pads = [b, a]
    for i in range(6):  # crossing one pad lands in front of the other
        dest = arrival(pads[i % 2])
        tr.teleport(*dest).walk_to(*nudge(pads[i % 2], dest))
    return [tr.end(False)]


def bad_pickup(t: GroundTruth, d: dict[str, Any]) -> Episodes:
    x, y, _ = d["position"]
    tr = Trace(t, start=(x - 64, y)).walk_to(x, y).pick_up(health=200)
    return [tr.wait(40).end(False)]


GENERATORS: dict[str, Callable[[GroundTruth, dict[str, Any]], Episodes]] = {
    "missing_blocking": missing_blocking,
    "geometry_trap": geometry_trap,
    "wrong_tag": wrong_tag,
    "wrong_special": wrong_special,
    "one_way_lift": one_way_lift,
    "teleport_loop": teleport_loop,
    "bad_pickup": bad_pickup,
}
