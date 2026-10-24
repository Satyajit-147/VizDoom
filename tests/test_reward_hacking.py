"""M5 acceptance: the three predictable reward hacks, each driven by a scripted
adversarial policy against the real env, with total return held under a bound
(PLAN: Three predictable reward-hacking failures, and the countermeasures)."""

from __future__ import annotations

import math

import pytest

from qatester.coverage import CoverageTracker
from qatester.env.actions import INDEX
from qatester.env.reward import RewardConfig
from qatester.env.vizdoom_env import QADoomEnv
from qatester.mapgen.cli import BuildResult
from qatester.mapgen.groundtruth import GroundTruth

FWD, BACK, NOOP = INDEX["forward"], INDEX["backward"], INDEX["noop"]
ALPHA = 0.5


def test_oracle_farming_pays_one_cluster_once(built: dict[str, BuildResult],
                                              truth: dict[tuple[str, str], GroundTruth]) -> None:  # fmt: skip
    """MAP03 buggy: step in and out of the leak in the hall's north wall, over and
    over. Every excursion is a fresh finding; only the first is paid."""
    env = QADoomEnv(built["buggy"].cfg_paths["MAP03"], truth["MAP03", "buggy"], cheats=True,
                    reward_cfg=RewardConfig(alpha=ALPHA))  # fmt: skip
    try:
        env.reset(options={"warp": (256, -150, 90.0)})
        raw, oracle_paid, total = 0, 0.0, 0.0
        for _ in range(12):
            for a in [FWD] * 8 + [BACK] * 8:
                _, r, term, trunc, info = env.step(a)
                total += r
                raw += sum(f.oracle == "out_of_bounds" for f in info["findings"])
                oracle_paid += info["reward_parts"]["oracle"]
                assert not (term or trunc)
    finally:
        env.close()
    assert raw >= 10, f"the policy did not actually farm the leak ({raw} findings)"
    assert oracle_paid == 3.0
    # Everything else this policy earns is a few novelty / portal crumbs.
    assert total <= (1 - ALPHA) * (3.0 + 4.0)


def test_novelty_churn_across_a_cell_boundary(built: dict[str, BuildResult],
                                              truth: dict[tuple[str, str], GroundTruth]) -> None:  # fmt: skip
    """MAP01: rock back and forth across the x = 256 cell boundary for 400 steps."""
    gt = truth["MAP01", "fixed"]
    cov = CoverageTracker(gt, state_buckets=4096)
    env = QADoomEnv(built["fixed"].cfg_paths["MAP01"], gt, cov, cheats=True,
                    reward_cfg=RewardConfig(alpha=ALPHA))  # fmt: skip
    try:
        env.reset(options={"warp": (240, 224, 0.0)})
        crossings, novelty, total = 0, 0.0, 0.0
        cells: set[tuple[int, int]] = set()
        prev = None
        x = 240.0
        for _ in range(400):  # facing east: forward while west of the line, back after
            _, r, _, _, info = env.step(FWD if x < 256 else BACK)
            x = info["pos"][0]
            total += r
            novelty += info["reward_parts"]["cell_novelty"] + info["reward_parts"]["state_novelty"]
            cells.add(info["cell"])
            crossings += prev is not None and info["cell"] != prev
            prev = info["cell"]
    finally:
        env.close()
        cov.close()
    assert crossings >= 50, f"the policy did not churn ({crossings} crossings)"
    # Each distinct cell (and its abstract state) pays at most once: re-entry needs a
    # move of more than one cell width away first.
    assert novelty <= (0.3 + 0.5) * len(cells) + 1e-9
    assert total <= (1 - ALPHA) * (0.3 + 0.5) * len(cells)


def _episode_return(env: QADoomEnv, kill_at: int | None) -> tuple[float, dict[str, object]]:
    env.reset()
    total, i = 0.0, 0
    while True:
        if kill_at is not None and i == kill_at:
            env.game.send_game_command("kill")
        _, r, term, trunc, info = env.step(NOOP)
        total += r
        i += 1
        if term or trunc:
            return total, {**info, "terminated": term, "truncated": trunc}


def test_suicide_is_worse_than_waiting_out_the_clock(built: dict[str, BuildResult],
                                                     truth: dict[tuple[str, str], GroundTruth]) -> None:  # fmt: skip
    """MAP01: dying early must never beat simply idling to the time limit, and the time
    limit truncates rather than terminates."""
    env = QADoomEnv(built["fixed"].cfg_paths["MAP01"], truth["MAP01", "fixed"], cheats=True,
                    reward_cfg=RewardConfig(alpha=ALPHA))  # fmt: skip
    try:
        died, d_info = _episode_return(env, kill_at=10)
        waited, w_info = _episode_return(env, kill_at=None)
    finally:
        env.close()
    assert d_info["outcome"] == "death" and d_info["terminated"] and not d_info["truncated"]
    assert d_info["reward_parts"]["death"] == -5.0  # type: ignore[index]
    assert w_info["outcome"] == "timeout" and w_info["truncated"] and not w_info["terminated"]
    steps = math.ceil(10500 / 4)
    assert waited == pytest.approx(ALPHA * -0.0005 * steps, abs=0.05)
    assert died < waited - 1.5
    assert died <= ALPHA * -5.0
