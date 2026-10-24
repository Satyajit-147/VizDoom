"""M5 acceptance: QADoomEnv against Gymnasium's own env checker, plus the engine-level
contract the training loop depends on (keys, reward wiring, oracles in the loop,
traces, demos, save/load, crash and hang supervision)."""

from __future__ import annotations

import os
import signal
from pathlib import Path

import numpy as np
import pytest
import vizdoom as vzd
from gymnasium.utils.env_checker import check_env

from qatester.coverage import CoverageTracker
from qatester.env.actions import INDEX
from qatester.env.trace import read_trace
from qatester.env.vizdoom_env import QADoomEnv
from qatester.mapgen.cli import BuildResult
from qatester.mapgen.groundtruth import GroundTruth
from qatester.oracles.base import CrashConfig, OracleConfig

FWD = INDEX["forward"]


def make_env(built: dict[str, BuildResult], truth: dict[tuple[str, str], GroundTruth],
             lump: str, variant: str = "fixed", **kw: object) -> QADoomEnv:  # fmt: skip
    return QADoomEnv(built[variant].cfg_paths[lump], truth[lump, variant], **kw)  # type: ignore[arg-type]


@pytest.fixture
def env01(built: dict[str, BuildResult],
          truth: dict[tuple[str, str], GroundTruth]) -> QADoomEnv:  # fmt: skip
    env = make_env(built, truth, "MAP01", cheats=True)
    yield env  # type: ignore[misc]
    env.close()


def test_gymnasium_env_checker(built: dict[str, BuildResult],
                               truth: dict[tuple[str, str], GroundTruth]) -> None:  # fmt: skip
    # Without a coverage tracker: shared visit counts make novelty depend on history,
    # by design, which the checker's same-seed-same-observation test would reject.
    env = make_env(built, truth, "MAP01")
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_env_never_sees_the_defects(built: dict[str, BuildResult],
                                    truth: dict[tuple[str, str], GroundTruth]) -> None:  # fmt: skip
    env = make_env(built, truth, "MAP03", variant="buggy")
    try:
        assert "defects" not in env.truth.data
        with pytest.raises(PermissionError):
            _ = env.truth.defects
    finally:
        env.close()


def test_observations_stay_in_their_spaces(built: dict[str, BuildResult],
                                           truth: dict[tuple[str, str], GroundTruth]) -> None:  # fmt: skip
    gt = truth["MAP07", "fixed"]
    cov = CoverageTracker(gt)
    env = QADoomEnv(built["fixed"].cfg_paths["MAP07"], gt, cov, seed=3)
    try:
        obs, _ = env.reset()
        rng = np.random.default_rng(0)
        for _ in range(400):
            assert env.observation_space.contains(obs)
            obs, r, term, trunc, info = env.step(int(rng.integers(12)))
            assert np.isfinite(r)
            assert set(info["reward_parts"]) >= {"time", "cell_novelty", "oracle"}
            if term or trunc:
                obs, _ = env.reset()
        assert obs["depth"].std() > 0, "the depth buffer is blank"
        assert cov.summary()["cells_visited"] > 3
    finally:
        env.close()
        cov.close()


def test_key_pickup_is_seen_and_rewarded_once(env01: QADoomEnv) -> None:
    # MAP01: the red key at (256, 700) in the loft, which the goal requires.
    env01.reset(options={"warp": (256, 620, 90.0)})
    got = None
    for _ in range(12):
        _, _, _, _, info = env01.step(FWD)
        if info["keys"]:
            got = info
            break
    assert got is not None and got["keys"] == ["key_red"]
    assert got["reward_parts"]["progress"] == 1.0 and got["reward_parts"]["entity"] == 1.0
    obs, _, _, _, info = env01.step(FWD)
    assert list(obs["scalars"][10:13]) == [1.0, 0.0, 0.0]
    assert info["reward_parts"]["progress"] == 0.0 and info["reward_parts"]["entity"] == 0.0


def test_use_target_flag_in_front_of_a_door(env01: QADoomEnv) -> None:
    # MAP01: the start-loft door is on y = 448, x in [208, 304]. Face it from 40 mu.
    obs, _ = env01.reset(options={"warp": (256, 448 - 16 - 40, 90.0)})
    assert obs["scalars"][17] == 1.0
    obs, _ = env01.reset(options={"warp": (256, 448 - 16 - 40, 270.0)})
    assert obs["scalars"][17] == 0.0, "facing away from the door"


def test_exit_terminates_with_the_goal_bonus(env01: QADoomEnv) -> None:
    # MAP01: the exit switch is on the east wall of the vault at (1200, 700).
    env01.reset(options={"warp": (1150, 700, 0.0)})
    for _ in range(20):
        _, _, term, trunc, info = env01.step(INDEX["forward_use"])
        if term or trunc:
            break
    assert term and not trunc and info["outcome"] == "goal"
    assert info["reward_parts"]["goal"] == 5.0
    with pytest.raises(RuntimeError):
        env01.step(FWD)  # the episode is over until reset()


def test_oracles_run_live_inside_the_env(built: dict[str, BuildResult],
                                         truth: dict[tuple[str, str], GroundTruth],
                                         tmp_path: Path) -> None:  # fmt: skip
    """MAP03 buggy: walking through the missing wall is reported on that very step,
    paid as an oracle reward, put on the queue, and kept as a trace plus a demo."""
    import queue as q

    findings_q: q.Queue[object] = q.Queue()
    env = make_env(built, truth, "MAP03", variant="buggy", cheats=True,
                   oracle_queue=findings_q, record_dir=tmp_path)  # fmt: skip
    try:
        # The hall's north wall is at y = -96; its leak spans x in [192, 320].
        # An excursion is reported when it ends: walk in, then back out.
        env.reset(options={"warp": (256, -150, 90.0)})
        hit = None
        for a in [FWD] * 10 + [INDEX["backward"]] * 10:
            _, _, _, _, info = env.step(a)
            if any(f.oracle == "out_of_bounds" for f in info["findings"]):
                hit = info
                break
        assert hit is not None, "walking through the leak was not reported"
        assert hit["reward_parts"]["oracle"] == 3.0
        f = next(f for f in hit["findings"] if f.oracle == "out_of_bounds")
        assert f.demo_path is not None and f.demo_path.parent == tmp_path / "demos"
        assert not findings_q.empty()
        env.reset()  # ends the episode: its trace is kept because it has a finding
        traces = sorted((tmp_path / "traces").glob("*.parquet"))
        assert len(traces) == 1
        records, meta = read_trace(traces[0])
        assert meta["map"] == "MAP03" and meta["episode"] == 0 and len(records) > 2
        assert f.demo_path.exists()
    finally:
        env.close()


def test_demo_replays_the_episode(built: dict[str, BuildResult],
                                  truth: dict[tuple[str, str], GroundTruth],
                                  tmp_path: Path) -> None:  # fmt: skip
    env = make_env(built, truth, "MAP01", record_dir=tmp_path, seed=5)
    rng = np.random.default_rng(1)
    try:
        env.reset()
        for _ in range(150):
            _, _, term, trunc, info = env.step(int(rng.choice([1, 1, 1, 5, 6, 8])))
            if term or trunc:
                break
        end = info["pos"]
        demo = env.demo_path
        env.reset()  # closes the recording
    finally:
        env.close()
    assert demo is not None and demo.exists()
    g = vzd.DoomGame()
    g.load_config(str(built["fixed"].cfg_paths["MAP01"]))
    g.set_window_visible(False)
    g.init()
    try:
        g.replay_episode(str(demo))
        while not g.is_episode_finished():
            g.advance_action()
        pos = (g.get_game_variable(vzd.GameVariable.POSITION_X),
               g.get_game_variable(vzd.GameVariable.POSITION_Y))  # fmt: skip
    finally:
        g.close()
    assert pos == pytest.approx(end[:2], abs=1.0), "the demo does not reproduce the run"


def test_same_seed_same_trajectory(built: dict[str, BuildResult],
                                   truth: dict[tuple[str, str], GroundTruth]) -> None:  # fmt: skip
    actions = np.random.default_rng(2).integers(0, 12, 120)
    ends = []
    for _ in range(2):
        env = make_env(built, truth, "MAP02", seed=11)
        try:
            env.reset(seed=11)
            for a in actions:
                *_, info = env.step(int(a))
            ends.append(info["pos"])
        finally:
            env.close()
    assert ends[0] == ends[1]


def test_save_and_load_state(env01: QADoomEnv, tmp_path: Path) -> None:
    env01.reset()
    for _ in range(20):
        env01.step(INDEX["forward_turn_left"])
    for _ in range(15):  # come to rest: a save lands 2 tics late (docs/decisions.md D29)
        obs, *_ = env01.step(0)
    pos = env01.last.x, env01.last.y  # type: ignore[union-attr]
    episode = env01.episode
    env01.save_state(tmp_path / "s.sav")
    for _ in range(30):
        env01.step(INDEX["forward_turn_left"])
    restored = env01.load_state(tmp_path / "s.sav")
    assert (env01.last.x, env01.last.y) == pytest.approx(pos, abs=0.5)  # type: ignore[union-attr]
    assert env01.episode == episode + 1, "a load starts a new oracle episode"
    assert (restored["depth"][-1] == obs["depth"][-1]).all()
    _, _, _, _, info = env01.step(0)
    assert not info["findings"], "loading must not look like a wall traversal"


def test_save_while_moving_drifts_at_most_two_tics(env01: QADoomEnv, tmp_path: Path) -> None:
    env01.reset()
    for _ in range(25):
        env01.step(FWD)
    pos = env01.last.x, env01.last.y  # type: ignore[union-attr]
    env01.save_state(tmp_path / "m.sav")
    env01.load_state(tmp_path / "m.sav")
    drift = float(np.hypot(env01.last.x - pos[0], env01.last.y - pos[1]))  # type: ignore[union-attr]
    assert drift <= 2 * 8.5


def _engine_pid(env: QADoomEnv) -> int:
    (pid,) = env.supervisor.engine_pids
    return pid


def test_engine_crash_is_a_finding_and_the_env_recovers(env01: QADoomEnv) -> None:
    env01.reset()
    env01.step(FWD)
    os.kill(_engine_pid(env01), signal.SIGKILL)
    _, r, term, trunc, info = env01.step(FWD)
    assert (term, trunc, info["outcome"]) == (False, True, "crash")
    assert [f.kind for f in info["findings"]] == ["crash"]
    assert r > 0, "a crash is a finding, paid like one"
    assert env01.supervisor.restarts == 1
    env01.reset()
    _, _, _, _, info = env01.step(FWD)
    assert info["tic"] > 0


def test_engine_hang_is_killed_and_reported(built: dict[str, BuildResult],
                                            truth: dict[tuple[str, str], GroundTruth]) -> None:  # fmt: skip
    cfg = OracleConfig(crash=CrashConfig(step_timeout_s=2.0))
    env = make_env(built, truth, "MAP01", oracle_cfg=cfg)
    try:
        env.reset()
        env.step(FWD)
        os.kill(_engine_pid(env), signal.SIGSTOP)  # frozen, not dead
        _, _, term, trunc, info = env.step(FWD)
        assert (term, trunc, info["outcome"]) == (False, True, "hang")
        assert [f.kind for f in info["findings"]] == ["hang"]
        env.reset()
        _, _, _, _, info = env.step(FWD)
        assert info["tic"] > 0
    finally:
        env.close()
