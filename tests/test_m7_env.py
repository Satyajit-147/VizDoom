"""M7 hooks in the env: the Bug Book reward and the novelty schedule."""

from __future__ import annotations

import pytest

from qatester.env.actions import INDEX
from qatester.env.reward import RewardConfig
from qatester.env.vizdoom_env import QADoomEnv
from qatester.mapgen.cli import BuildResult
from qatester.mapgen.groundtruth import GroundTruth

FWD, BACK = INDEX["forward"], INDEX["backward"]


def test_bugbook_reward_once_per_entry_per_episode(
    built: dict[str, BuildResult], truth: dict[tuple[str, str], GroundTruth]
) -> None:
    env = QADoomEnv(built["fixed"].cfg_paths["MAP01"], truth["MAP01", "fixed"], cheats=True,
                    reward_cfg=RewardConfig(alpha=0.0))  # fmt: skip
    try:
        # A known bug 150 mu east of the start position, in the start room.
        env.set_bugbook_targets("MAP01", [("map01-stuck-0001", 400.0, 224.0)])
        env.reset(options={"warp": (250, 224, 0.0)})
        paid = []
        for a in [FWD] * 12 + [BACK] * 12 + [FWD] * 12:  # past it, back, past it again
            *_, info = env.step(a)
            paid.append(info["reward_parts"]["bugbook"])
        assert sum(paid) == pytest.approx(1.5), "paid once per entry per episode"
        env.reset(options={"warp": (250, 224, 0.0)})
        again = sum(env.step(FWD)[4]["reward_parts"]["bugbook"] for _ in range(12))
        assert again == pytest.approx(1.5), "a new episode pays again"
        env.set_bugbook_targets("MAP01", [])
        env.reset(options={"warp": (250, 224, 0.0)})
        assert sum(env.step(FWD)[4]["reward_parts"]["bugbook"] for _ in range(12)) == 0.0
    finally:
        env.close()


def test_novelty_beta_can_be_scheduled(
    built: dict[str, BuildResult], truth: dict[tuple[str, str], GroundTruth]
) -> None:
    env = QADoomEnv(built["fixed"].cfg_paths["MAP01"], truth["MAP01", "fixed"])
    try:
        assert env.reward.cfg.state_novelty == 0.5
        env.set_novelty_beta(0.1)
        assert env.reward.cfg.state_novelty == pytest.approx(0.1)
        assert env.reward_cfg.state_novelty == pytest.approx(0.1)
    finally:
        env.close()
