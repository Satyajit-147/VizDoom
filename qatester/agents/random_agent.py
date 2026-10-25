"""T1: the uniform random policy, the floor every other tier is measured against."""

from __future__ import annotations

import numpy as np

from qatester.agents.base import Obs
from qatester.env.actions import N_ACTIONS


class RandomAgent:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def reset(self) -> None:
        pass

    def act(self, obs: Obs) -> int:
        return int(self.rng.integers(N_ACTIONS))
