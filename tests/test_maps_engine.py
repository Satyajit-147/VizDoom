"""M2: every compiled map loads in ViZDoom and plays as intended.

Checks that the engine sees exactly the geometry the compiler emitted, the player spawns
where the spec says, random play stays inside the level, and every portal kind works:
doors open, locked doors need their key, remote doors need their switch, lifts carry the
player up and let them walk back down, teleports arrive, exit switches end the episode.

Runs on the `fixed` variant, which must stay fully playable once M3 adds defects.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import vizdoom as vzd
from doomdriver import Driver, approach, inside_room

from qatester.mapgen.cli import BuildResult, build
from qatester.mapgen.geometry import LIFT_DEPTH, WALL, CompiledMap
from qatester.mapgen.spec import Portal, shared_wall

REPO = Path(__file__).resolve().parents[1]
MAPS = [f"MAP{i:02d}" for i in range(1, 9)]
PLAYER_RADIUS = 16


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> BuildResult:
    return build(REPO / "maps", tmp_path_factory.mktemp("build"), "fixed")


@pytest.fixture(scope="module", params=MAPS)
def driver(request: pytest.FixtureRequest, built: BuildResult) -> Iterator[tuple[Driver, CompiledMap]]:
    d = Driver(built.cfg_paths[request.param])
    yield d, built.compiled[request.param]
    d.close()


def cross(d: Driver, cm: CompiledMap, p: Portal, src: str, dst: str, use: bool, tics: int = 140,
          start_dist: float = 56) -> bool:  # fmt: skip
    rs, rd = cm.spec.room(src), cm.spec.room(dst)
    wall = shared_wall(rs, rd)
    assert wall is not None
    along = (wall.lo + wall.hi) / 2
    if p.at is not None:
        along = p.at[1] if wall.axis == "x" else p.at[0]
    x, y, angle = approach(rs, wall.axis, wall.at, along, start_dist)
    d.reset()
    d.warp(x, y)
    d.face(angle)
    if use:
        d.forward(20)  # bump into the door or lift face
        d.use()
        d.idle(35)
    return d.walk_until(inside_room(rd), tics)


# --- static checks --------------------------------------------------------------------

def test_engine_geometry_matches_compiler(driver: tuple[Driver, CompiledMap]) -> None:
    d, cm = driver
    d.reset()
    sectors = d.game.get_state().sectors
    tm = cm.textmap()
    assert [(round(s.floor_height), round(s.ceiling_height)) for s in sectors] == [
        (s.heightfloor, s.heightceiling) for s in tm.sectors
    ]
    engine = {
        frozenset({(round(ln.x1), round(ln.y1)), (round(ln.x2), round(ln.y2))})
        for s in sectors
        for ln in s.lines
    }
    compiled = {frozenset({line.v1, line.v2}) for line in cm.lines}
    assert engine == compiled


def test_spawn_position_and_walls_render(driver: tuple[Driver, CompiledMap]) -> None:
    d, cm = driver
    d.reset()
    spawn = next(t for t in cm.spec.things if t.type == "player_start")
    x, y, z = d.pos()
    assert (x, y) == pytest.approx(spawn.pos)
    assert z == pytest.approx(cm.spec.room(spawn.room).floor)
    depth = d.game.get_state().depth_buffer
    assert depth is not None and depth.max() > 0, "nothing renders: reversed winding?"


def test_random_play_stays_in_bounds(
    driver: tuple[Driver, CompiledMap], built: BuildResult
) -> None:
    _, cm = driver
    g = vzd.DoomGame()
    g.load_config(str(built.cfg_paths[cm.spec.map_lump]))
    g.set_seed(1)
    g.init()
    try:
        rng = np.random.default_rng(1)
        n = g.get_available_buttons_size()
        g.new_episode()
        for _ in range(500):
            if g.is_episode_finished():
                g.new_episode()
            g.make_action(rng.integers(0, 2, size=n).tolist(), 4)
            x = g.get_game_variable(vzd.GameVariable.POSITION_X)
            y = g.get_game_variable(vzd.GameVariable.POSITION_Y)
            assert math.isfinite(x) and math.isfinite(y)
            assert any(
                r.x0 - PLAYER_RADIUS <= x <= r.x1 + PLAYER_RADIUS
                and r.y0 - PLAYER_RADIUS <= y <= r.y1 + PLAYER_RADIUS
                for r in cm.spec.rooms
            ), f"left the level at ({x:.0f}, {y:.0f})"
    finally:
        g.close()


# --- portals --------------------------------------------------------------------------

def _portals(cm: CompiledMap, *kinds: str) -> list[Portal]:
    return [p for p in cm.spec.portals if p.kind in kinds]


def _remote(cm: CompiledMap, p: Portal) -> bool:
    return any(t.opens == p.id for t in cm.spec.things)


def test_open_portals_both_ways(driver: tuple[Driver, CompiledMap]) -> None:
    d, cm = driver
    for p in _portals(cm, "open"):
        assert cross(d, cm, p, p.a, p.b, use=False), f"{p.id}: {p.a} -> {p.b}"
        assert cross(d, cm, p, p.b, p.a, use=False), f"{p.id}: {p.b} -> {p.a}"


def test_doors_closed_until_used_then_open_both_ways(
    driver: tuple[Driver, CompiledMap],
) -> None:
    d, cm = driver
    for p in _portals(cm, "door"):
        if _remote(cm, p):
            continue
        assert not cross(d, cm, p, p.a, p.b, use=False, tics=70), f"{p.id} starts open"
        assert cross(d, cm, p, p.a, p.b, use=True), f"{p.id}: {p.a} -> {p.b}"
        assert cross(d, cm, p, p.b, p.a, use=True), f"{p.id}: {p.b} -> {p.a}"


def test_locked_doors_need_their_key(driver: tuple[Driver, CompiledMap]) -> None:
    d, cm = driver
    for p in _portals(cm, "locked_door"):
        assert p.requires is not None
        assert not cross(d, cm, p, p.a, p.b, use=True, tics=70), f"{p.id} opened without key"
        rs, rd = cm.spec.room(p.a), cm.spec.room(p.b)
        wall = shared_wall(rs, rd)
        assert wall is not None
        x, y, angle = approach(rs, wall.axis, wall.at, (wall.lo + wall.hi) / 2, 56)
        d.reset()
        d.give(p.requires)
        d.warp(x, y)
        d.face(angle)
        d.forward(20)
        d.use()
        d.idle(35)
        assert d.walk_until(inside_room(rd), 140), f"{p.id} stays shut with {p.requires}"


def test_remote_doors_open_only_from_their_switch(driver: tuple[Driver, CompiledMap]) -> None:
    d, cm = driver
    for p in _portals(cm, "door"):
        if not _remote(cm, p):
            continue
        assert not cross(d, cm, p, p.a, p.b, use=True, tics=70), f"{p.id} opens by hand"
        mark = next(m for m in cm.marks if m.special.args[:1] == (cm.sectors[f"door:{p.id}"].tag,))
        switch_room = cm.spec.room(next(t.room for t in cm.spec.things if t.opens == p.id))
        x, y, angle = approach(switch_room, mark.axis, mark.at, (mark.lo + mark.hi) / 2, 40)
        rs, rd = cm.spec.room(p.a), cm.spec.room(p.b)
        wall = shared_wall(rs, rd)
        assert wall is not None
        dx, dy, dangle = approach(rs, wall.axis, wall.at, (wall.lo + wall.hi) / 2, 56)
        d.reset()
        d.warp(x, y)
        d.face(angle)
        d.use()
        d.idle(70)
        d.warp(dx, dy)
        d.face(dangle)
        assert d.walk_until(inside_room(rd), 140), f"switch {mark.source} did not open {p.id}"


def test_lifts_carry_up_and_walk_down(driver: tuple[Driver, CompiledMap]) -> None:
    d, cm = driver
    for p in _portals(cm, "lift"):
        ra, rb = cm.spec.room(p.a), cm.spec.room(p.b)
        lower, upper = (ra, rb) if ra.floor < rb.floor else (rb, ra)
        # Stand clear of the platform, which reaches LIFT_DEPTH into the lower room.
        up = cross(d, cm, p, lower.id, upper.id, use=True, tics=300,
                   start_dist=WALL // 2 + LIFT_DEPTH + 40)  # fmt: skip
        assert up, f"{p.id}: lift does not carry {lower.id} -> {upper.id}"
        assert d.pos()[2] == pytest.approx(upper.floor), f"{p.id}: not at the upper floor"
        down = cross(d, cm, p, upper.id, lower.id, use=False, tics=140)
        assert down, f"{p.id}: cannot walk off {upper.id} -> {lower.id}"


def test_teleports_arrive(driver: tuple[Driver, CompiledMap]) -> None:
    d, cm = driver
    for p in _portals(cm, "teleport"):
        assert p.at is not None
        ra = cm.spec.room(p.a)
        cx, cy = (ra.x0 + ra.x1) / 2, (ra.y0 + ra.y1) / 2
        vx, vy = cx - p.at[0], cy - p.at[1]
        norm = math.hypot(vx, vy)
        d.reset()
        d.warp(p.at[0] + 96 * vx / norm, p.at[1] + 96 * vy / norm)
        d.face_point(*p.at)
        dest_x, dest_y = cm.teleport_dests[p.id]
        def near_dest(x: float, y: float, dx: float = dest_x, dy: float = dest_y) -> bool:
            return math.hypot(x - dx, y - dy) < 32

        arrived = d.walk_until(near_dest, 70)
        assert arrived, f"{p.id}: did not arrive at ({dest_x}, {dest_y})"


def test_exit_switch_ends_episode(driver: tuple[Driver, CompiledMap]) -> None:
    d, cm = driver
    exits = [m for m in cm.marks if m.special.special == 243]
    assert len(exits) == 1
    m = exits[0]
    room = cm.spec.room(cm.spec.thing(m.source).room)
    x, y, angle = approach(room, m.axis, m.at, (m.lo + m.hi) / 2, 40)
    d.reset()
    d.warp(x, y)
    d.face(angle)
    d.use()
    d.idle(2)
    assert d.game.is_episode_finished(), "exit switch did not end the episode"
