"""M5 pieces that need no engine: actions, observation, reward rules, coverage counts,
Parquet traces and the supervisor's crash / hang / retry handling."""

from __future__ import annotations

import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pytest

from qatester.coverage import CoverageTracker
from qatester.env.actions import ACTIONS, BUTTONS, INDEX, N_ACTIONS
from qatester.env.observation import (
    N_SCALARS,
    DepthProcessor,
    FrameStack,
    ScalarInputs,
    observation_space,
    scalars,
)
from qatester.env.reward import EpisodeReward, RewardConfig
from qatester.env.supervisor import Supervisor
from qatester.env.trace import read_trace, write_trace
from qatester.mapgen.groundtruth import GroundTruth
from qatester.oracles.base import Finding, StepRecord

# --- actions -----------------------------------------------------------------------


def test_twelve_actions_as_the_plan_lists_them() -> None:
    assert N_ACTIONS == 12
    expected = [
        set(), {"MOVE_FORWARD"}, {"MOVE_BACKWARD"}, {"MOVE_LEFT"}, {"MOVE_RIGHT"},
        {"MOVE_FORWARD", "TURN_LEFT"}, {"MOVE_FORWARD", "TURN_RIGHT"}, {"USE"},
        {"MOVE_FORWARD", "USE"}, {"ATTACK"}, {"JUMP"}, {"MOVE_FORWARD", "JUMP"},
    ]  # fmt: skip
    assert [set(a.pressed) for a in ACTIONS] == expected
    assert all(len(a.buttons) == len(BUTTONS) == 9 for a in ACTIONS)
    assert any(a.use for a in ACTIONS) and any(a.jump for a in ACTIONS)
    assert ACTIONS[INDEX["strafe_left"]].move_dirs == {"left"}
    assert ACTIONS[INDEX["forward_turn_right"]].move_dirs == {"forward"}


# --- observation -------------------------------------------------------------------


def test_depth_is_centre_cropped_and_resized() -> None:
    depth = np.zeros((120, 160), np.uint8)
    depth[:, 20:140] = 200  # the centre square only
    out = DepthProcessor()(depth)
    assert out.shape == (64, 64) and out.dtype == np.uint8
    assert (out == 200).all(), "the crop must drop the 20-column side bands"
    with pytest.raises(ValueError):
        DepthProcessor()(np.zeros((64, 64), np.uint8))


def test_frame_stack_returns_fresh_arrays() -> None:
    fs = FrameStack()
    a = fs.fill(np.full((64, 64), 1, np.uint8))
    b = fs.push(np.full((64, 64), 2, np.uint8))
    assert a[-1, 0, 0] == 1, "earlier observations must not change under later steps"
    assert list(b[:, 0, 0]) == [1, 1, 1, 2]


def _inputs(**kw: object) -> ScalarInputs:
    base: dict[str, object] = {
        "angle": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0, "on_ground": True, "moved_mu": 0.0,
        "commanded": False, "tics": 4, "health": 100.0, "armour": 0.0, "ammo": 50.0,
        "keys": frozenset(), "elapsed": 0.0, "novelty": 1.0, "neighbour_novelty": 1.0,
        "interactive_in_view": 0, "use_target": False,
    }
    base.update(kw)
    return ScalarInputs(**base)  # type: ignore[arg-type]


def test_scalars_are_egocentric_and_bounded() -> None:
    space = observation_space()["scalars"]
    # Facing north and moving north at 16 mu/tic is "forward" at 0.5.
    v = scalars(_inputs(angle=90.0, vy=16.0))
    assert v.shape == (N_SCALARS,) and v.dtype == np.float32
    assert v[0] == pytest.approx(1.0) and v[1] == pytest.approx(0.0, abs=1e-6)
    assert v[2] == pytest.approx(0.5) and v[3] == pytest.approx(0.0, abs=1e-6)
    # Moving east while facing north is to the agent's right: negative "left".
    assert scalars(_inputs(angle=90.0, vx=16.0))[3] == pytest.approx(-0.5)
    extreme = scalars(_inputs(health=200.0, armour=400.0, ammo=500.0, vx=999.0,
                              interactive_in_view=50))  # fmt: skip
    assert space.contains(extreme)
    keys = scalars(_inputs(keys=frozenset({"key_blue"})))
    assert list(keys[10:13]) == [0.0, 1.0, 0.0]


def test_speed_ratio_is_the_stuck_signal() -> None:
    assert scalars(_inputs(commanded=True, moved_mu=0.0))[6] == 0.0  # pushing a wall
    assert scalars(_inputs(commanded=True, moved_mu=30.0))[6] == pytest.approx(1.0)
    assert scalars(_inputs(commanded=False))[6] == 1.0  # not trying is not being stuck


# --- reward ------------------------------------------------------------------------


def _finding(oracle: str, kind: str, x: float, y: float) -> Finding:
    return Finding(oracle, kind, (x, y, 0.0), (0, 0), 1.0, {"episode": 0})


def test_oracle_reward_once_per_family_cluster_per_episode() -> None:
    r = EpisodeReward(RewardConfig(alpha=0.0))
    paid = [r.step(findings=[f])[1]["oracle"] for f in (
        _finding("stuck", "wedge", 0, 0),
        _finding("stuck", "wedge", 50, 50),  # within 96 mu: same cluster
        _finding("stuck", "vertical_trap", 0, 0),  # same family, same place
        _finding("bounds", "outside_level", 0, 0),  # another family
        _finding("stuck", "wedge", 500, 0),  # another place
        _finding("invariant", "health_cap", 0, 0),
        _finding("invariant", "ammo_nonneg", 0, 0),  # invariants split by name
    )]  # fmt: skip
    assert paid == [3.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0]
    r.reset()
    assert r.step(findings=[_finding("stuck", "wedge", 0, 0)])[1]["oracle"] == 3.0


def test_cells_rearm_only_after_moving_away() -> None:
    r = EpisodeReward()
    assert r.enter_cell((0, 0)) and r.enter_cell((1, 0))
    assert not r.enter_cell((0, 0)), "bouncing back across a boundary does not count"
    assert not r.enter_cell((1, 0))
    assert r.enter_cell((2, 0))
    assert r.enter_cell((3, 0))  # now two cells clear of (1, 0): it re-arms
    assert r.enter_cell((1, 0))


def test_firsts_are_per_episode_and_alpha_mixes() -> None:
    cfg = RewardConfig(alpha=0.25)
    r = EpisodeReward(cfg)
    total, parts = r.step(progress={"key:key_red"}, entities={"redkey"}, portals={"p1"},
                          cell_count=3)  # fmt: skip
    assert parts["progress"] == 1.0 and parts["entity"] == 1.0 and parts["portal"] == 1.0
    assert parts["cell_novelty"] == pytest.approx(0.3 / 2)
    valid = 1.0 + cfg.time
    adv = 1.0 + 1.0 + 0.15
    assert total == pytest.approx(0.25 * valid + 0.75 * adv)
    _, again = r.step(progress={"key:key_red"}, entities={"redkey"}, portals={"p1"})
    assert again["progress"] == again["entity"] == again["portal"] == 0.0
    _, died = r.step(died=True)
    assert died["death"] == -5.0
    with pytest.raises(ValueError):
        RewardConfig(alpha=1.5)


# --- coverage ----------------------------------------------------------------------


def _bump(tracker: CoverageTracker, cell: tuple[int, int], n: int) -> None:
    for _ in range(n):
        tracker.visit_cell(cell)
    tracker.close()


def test_coverage_counts_are_shared_across_processes(
    truth: dict[tuple[str, str], GroundTruth],
) -> None:
    gt = truth["MAP01", "fixed"]
    cov = CoverageTracker(gt, state_buckets=1024)
    try:
        cell = gt.cell_of(256, 224)
        assert cov.visit_cell(cell) == 0
        ctx = mp.get_context("spawn")
        p = ctx.Process(target=_bump, args=(cov, cell, 5))
        p.start()
        p.join(60)
        assert p.exitcode == 0
        assert cov.cell_count(cell) == 6, "the child's visits must land in the same counts"
        assert cov.visit_cell((-5, -5)) == 0 and cov.cell_count((-5, -5)) == 0
        cov.visit_state((*cell, 1, 0))
        cov.use_thing("redkey")
        cov.traverse_portal("p_start_hall")
        s = cov.summary()
        assert s["cells_visited"] == 1 and s["abstract_states"] == 1
        assert 0 < s["cell_coverage"] < 1
        assert s["entity_coverage"] == pytest.approx(1 / 3)  # redkey of 3 interactive
        assert s["connection_coverage"] == pytest.approx(1 / 3)
        assert 0.0 < cov.neighbour_novelty(cell) <= 1.0
    finally:
        cov.close()


# --- traces ------------------------------------------------------------------------


def test_trace_round_trip(tmp_path: Path) -> None:
    recs = [
        StepRecord(tic=4 * i, x=1.5 * i, y=-2.0, z=0.0, keys=frozenset({"key_red"}),
                   move_dirs=frozenset({"forward", "left"}), use=i % 2 == 0,
                   sector_floors=(0.0, -64.0), sector_ceilings=(128.0, 128.0),
                   notifications=("Picked up a medikit.",) if i == 1 else (),
                   error="boom" if i == 2 else None, done=i == 2, truncated=i == 2)
        for i in range(3)
    ]  # fmt: skip
    path = tmp_path / "t" / "ep.parquet"
    write_trace(path, recs, {"map": "MAP01", "episode": 7})
    back, meta = read_trace(path)
    assert meta == {"map": "MAP01", "episode": 7}
    assert back == recs


# --- supervisor --------------------------------------------------------------------


class EngineError(Exception):
    pass


class FakeGame:
    def __init__(self) -> None:
        self.closed = False

    def step(self, mode: str) -> str:
        if mode == "crash":
            raise EngineError("engine died")
        if mode == "hang":
            time.sleep(1.5)
        return "ok"

    def close(self) -> None:
        self.closed = True


def test_supervisor_reports_crash_and_hang_without_raising() -> None:
    sup = Supervisor(FakeGame, step_timeout_s=0.4, errors=(EngineError,))
    sup.start()
    try:
        ok = sup.call(sup.game.step, "fine")
        assert (ok.value, ok.error, ok.hung) == ("ok", None, False)
        crash = sup.call(sup.game.step, "crash")
        assert crash.error is not None and "engine died" in crash.error and not crash.hung
        old = sup.game
        sup.restart()
        assert old.closed and sup.game is not old and sup.restarts == 1
        hang = sup.call(sup.game.step, "hang")
        assert hang.hung and hang.error is None and hang.wall_s >= 0.4
        assert not sup.call(sup.game.step, "fine").hung, "the hang flag must not stick"
    finally:
        sup.close()


def test_supervisor_retries_init_with_backoff() -> None:
    attempts: list[int] = []
    slept: list[float] = []

    def flaky() -> FakeGame:
        attempts.append(1)
        if len(attempts) < 3:
            raise EngineError("no shared memory yet")
        return FakeGame()

    sup = Supervisor(flaky, errors=(EngineError,), sleep=slept.append)
    sup.start()
    sup.close()
    assert len(attempts) == 3 and slept == [2.0, 4.0]

    def dead() -> FakeGame:
        raise EngineError("never")

    sup = Supervisor(dead, errors=(EngineError,), sleep=slept.append, retries=5)
    with pytest.raises(EngineError):
        sup.start()
    assert slept[2:] == [2.0, 4.0, 8.0, 16.0]
    assert math.isclose(sum(slept[2:]), 30.0)
