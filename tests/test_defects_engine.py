"""M3: every injected defect really happens in the engine, and its fixed twin does not.

Each defect kind has one scripted scenario, run on the buggy and the fixed build of the
same map. The scenario returns True when the defect shows. Scenarios are generic over the
defect's target, so they also cover maps generated later.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from doomdriver import Driver, approach, inside_room

from qatester.mapgen.cli import BuildResult, build
from qatester.mapgen.geometry import EXIT_NORMAL, LIFT_DEPTH, WALL, CompiledMap
from qatester.mapgen.spec import Defect, MapSpec, Room, shared_wall

REPO = Path(__file__).resolve().parents[1]
PLAYER_RADIUS = 16


@pytest.fixture(scope="module")
def builds(tmp_path_factory: pytest.TempPathFactory) -> dict[str, BuildResult]:
    out = tmp_path_factory.mktemp("build")
    return {"buggy": build(REPO / "maps", out, "buggy"), "fixed": build(REPO / "maps", out, "fixed")}


def _cases(builds: dict[str, BuildResult]) -> Iterator[tuple[str, Defect]]:
    for lump, cm in builds["buggy"].compiled.items():
        for d in cm.spec.defects:
            yield lump, d


def _outside_level(spec: MapSpec, x: float, y: float) -> bool:
    return not any(
        r.x0 - PLAYER_RADIUS <= x <= r.x1 + PLAYER_RADIUS
        and r.y0 - PLAYER_RADIUS <= y <= r.y1 + PLAYER_RADIUS
        for r in spec.rooms
    )


def _neighbour_via_open(spec: MapSpec, room: Room) -> Room:
    p = next(p for p in spec.portals if p.kind == "open" and room.id in (p.a, p.b))
    return spec.room(p.b if p.a == room.id else p.a)


# --- one scenario per defect kind: True = the defect showed ---------------------------

def leaks(d: Driver, cm: CompiledMap, defect: Defect) -> bool:
    room = cm.spec.room(defect.target)
    lo, hi = (int(v) for v in defect.params["segment"])  # type: ignore[attr-defined]
    wall = str(defect.params["wall"])
    axis, at = {"north": ("y", room.y1), "south": ("y", room.y0),
                "east": ("x", room.x1), "west": ("x", room.x0)}[wall]  # fmt: skip
    x, y, angle = approach(room, axis, at, (lo + hi) / 2, 48)
    d.reset()
    d.warp(x, y)
    d.face(angle)
    d.forward(60)
    return _outside_level(cm.spec, *d.pos()[:2])


def switch_dead(d: Driver, cm: CompiledMap, defect: Defect) -> bool:
    switch = cm.spec.thing(defect.target)
    assert switch.opens is not None
    mark = next(m for m in cm.marks if m.source == switch.id)
    room = cm.spec.room(switch.room)
    x, y, angle = approach(room, mark.axis, mark.at, (mark.lo + mark.hi) / 2, 40)
    door = cm.spec.portal(switch.opens)
    ra, rb = cm.spec.room(door.a), cm.spec.room(door.b)
    wall = shared_wall(ra, rb)
    assert wall is not None
    dx, dy, dangle = approach(ra, wall.axis, wall.at, (wall.lo + wall.hi) / 2, 56)
    d.reset()
    d.warp(x, y)
    d.face(angle)
    d.use()
    d.idle(70)
    d.warp(dx, dy)
    d.face(dangle)
    return not d.walk_until(inside_room(rb), 140)


def opens_without_key(d: Driver, cm: CompiledMap, defect: Defect) -> bool:
    door = cm.spec.portal(defect.target)
    ra, rb = cm.spec.room(door.a), cm.spec.room(door.b)
    wall = shared_wall(ra, rb)
    assert wall is not None
    x, y, angle = approach(ra, wall.axis, wall.at, (wall.lo + wall.hi) / 2, 56)
    d.reset()
    d.warp(x, y)
    d.face(angle)
    d.forward(20)
    d.use()
    d.idle(35)
    return d.walk_until(inside_room(rb), 140)


def traps(d: Driver, cm: CompiledMap, defect: Defect) -> bool:
    """Walk into the room from its neighbour, then try to walk and jump back out."""
    room = cm.spec.room(defect.target)
    out = _neighbour_via_open(cm.spec, room)
    wall = shared_wall(room, out)
    assert wall is not None
    along = (wall.lo + wall.hi) / 2
    x, y, angle = approach(out, wall.axis, wall.at, along, 64)
    d.reset()
    d.warp(x, y)
    d.face(angle)
    assert d.walk_until(inside_room(room), 120), "could not even enter the room"
    d.forward(10)
    d.face(angle + 180)
    escaped = inside_room(out)
    for _ in range(4):  # several runs at the ledge, jumping all the way
        d.forward(40, jump=True)
        if escaped(*d.pos()[:2]):
            return False
    return not escaped(*d.pos()[:2])


def lift_stranded(d: Driver, cm: CompiledMap, defect: Defect) -> bool:
    """From the lower room, call the lift and try to ride it up."""
    p = cm.spec.portal(defect.target)
    ra, rb = cm.spec.room(p.a), cm.spec.room(p.b)
    lower, upper = (ra, rb) if ra.floor < rb.floor else (rb, ra)
    wall = shared_wall(lower, upper)
    assert wall is not None
    x, y, angle = approach(lower, wall.axis, wall.at, (wall.lo + wall.hi) / 2,
                           WALL // 2 + LIFT_DEPTH + 40)  # fmt: skip
    d.reset()
    d.warp(x, y)
    d.face(angle)
    d.forward(20)
    d.use()
    d.idle(35)
    return not d.walk_until(inside_room(upper), 300)


def teleport_bounces(d: Driver, cm: CompiledMap, defect: Defect) -> bool:
    p = cm.spec.portal(defect.target)
    assert p.at is not None
    ra = cm.spec.room(p.a)
    cx, cy = (ra.x0 + ra.x1) / 2, (ra.y0 + ra.y1) / 2
    vx, vy = cx - p.at[0], cy - p.at[1]
    norm = math.hypot(vx, vy)
    d.reset()
    d.warp(p.at[0] + 96 * vx / norm, p.at[1] + 96 * vy / norm)
    d.face_point(*p.at)
    jumps, last = 0, d.pos()
    for _ in range(140):
        d.forward(1)
        now = d.pos()
        if math.hypot(now[0] - last[0], now[1] - last[1]) > 200:
            jumps += 1
        last = now
    return jumps >= 3


def overheals(d: Driver, cm: CompiledMap, defect: Defect) -> bool:
    thing = cm.spec.thing(defect.target)
    d.reset()
    d.warp(thing.pos[0] - 64, thing.pos[1])
    d.face(0)
    d.forward(30)
    return d.health() > 100


SCENARIOS: dict[str, Callable[[Driver, CompiledMap, Defect], bool]] = {
    "missing_blocking": leaks,
    "wrong_tag": switch_dead,
    "wrong_special": opens_without_key,
    "geometry_trap": traps,
    "one_way_lift": lift_stranded,
    "teleport_loop": teleport_bounces,
    "bad_pickup": overheals,
}


@pytest.mark.parametrize("variant", ["buggy", "fixed"])
def test_defects_show_only_in_buggy_build(builds: dict[str, BuildResult], variant: str) -> None:
    results = []
    for lump, defect in _cases(builds):
        build_result = builds[variant]
        d = Driver(build_result.cfg_paths[lump])
        try:
            shown = SCENARIOS[defect.kind](d, build_result.compiled[lump], defect)
        finally:
            d.close()
        results.append((lump, defect.id, shown))
    expected = variant == "buggy"
    wrong = [f"{lump} {did}: shown={shown}" for lump, did, shown in results if shown != expected]
    assert len(results) == 15
    assert not wrong, f"{variant}: " + "; ".join(wrong)


def test_exits_still_work_in_buggy_build(builds: dict[str, BuildResult]) -> None:
    """Defects must not break the level's own goal mechanism."""
    for lump, cm in builds["buggy"].compiled.items():
        m = next(m for m in cm.marks if m.special.special == EXIT_NORMAL)
        room = cm.spec.room(cm.spec.thing(m.source).room)
        x, y, angle = approach(room, m.axis, m.at, (m.lo + m.hi) / 2, 40)
        d = Driver(builds["buggy"].cfg_paths[lump])
        try:
            d.reset()
            d.warp(x, y)
            d.face(angle)
            d.use()
            d.idle(2)
            assert d.game.is_episode_finished(), f"{lump}: exit broken by a defect"
        finally:
            d.close()
