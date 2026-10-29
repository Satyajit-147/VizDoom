"""Potential-based shaping toward the goal (docs/decisions.md D35)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qatester.env.actions import INDEX, N_ACTIONS
from qatester.env.reward import RewardConfig
from qatester.env.shaping import GoalDistance
from qatester.env.vizdoom_env import QADoomEnv
from qatester.mapgen.cli import BuildResult
from qatester.mapgen.groundtruth import GroundTruth

MAPS = [f"MAP{i:02d}" for i in range(1, 9)]


def test_map01_route(truth: dict[tuple[str, str], GroundTruth]) -> None:
    g = GoalDistance(truth["MAP01", "fixed"])
    # Spawn -> loft door (256, 448) -> key (256, 700), then key -> ... -> exit (1200, 700).
    to_key = 224 + 252
    assert g.remaining(256, 224, []) == pytest.approx(to_key + g.remaining(256, 700, ["key_red"]))
    # Picking the key up does not move the potential: the rest of the route was counted.
    assert g.remaining(256, 700, []) == pytest.approx(g.remaining(256, 700, ["key_red"]))
    assert g.remaining(1190, 700, ["key_red"]) < 20
    # Without the key the vault's locked door is shut, so the route goes via the loft.
    assert g.remaining(600, 224, []) > g.remaining(600, 224, ["key_red"]) + 400


@pytest.mark.parametrize("lump", MAPS)
def test_every_map_has_a_finite_route_from_spawn(
    truth: dict[tuple[str, str], GroundTruth], lump: str
) -> None:
    g = GoalDistance(truth[lump, "fixed"])
    sx, sy = truth[lump, "fixed"].data["spawn"]["pos"]
    d = g.remaining(sx, sy, [])
    assert 100 < d < 10_000


def test_shaping_telescopes_so_it_cannot_be_farmed(
    built: dict[str, BuildResult], truth: dict[tuple[str, str], GroundTruth]
) -> None:
    env = QADoomEnv(built["fixed"].cfg_paths["MAP01"], truth["MAP01", "fixed"],
                    reward_cfg=RewardConfig(alpha=1.0, shaping=1.0))  # fmt: skip
    g = GoalDistance(truth["MAP01", "fixed"])
    rng = np.random.default_rng(0)
    try:
        env.reset()
        start = env.last
        assert start is not None
        total = 0.0
        for i in range(600):  # random play, then rocking back and forth
            a = int(rng.integers(N_ACTIONS)) if i < 300 else (
                INDEX["forward"] if (i // 3) % 2 else INDEX["backward"])
            *_, info = env.step(a)
            total += info["reward_parts"]["shaping"]
        end = env.last
        assert end is not None
    finally:
        env.close()
    expected = (g.remaining(start.x, start.y, start.keys)
                - g.remaining(end.x, end.y, end.keys)) / 1000.0  # fmt: skip
    assert math.isclose(total, expected, abs_tol=1e-6)
