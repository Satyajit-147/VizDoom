"""Map curriculum (PLAN: Training curriculum): before each episode, pick a map by the
weights of the current stage and switch the env to it.

Stages are keyed on the run's global step count. Each worker estimates it as its own
steps times the number of workers, plus the steps done before a resume; workers step in
lockstep under a vector env, so the estimate is close.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from qatester.coverage import CoverageTracker
from qatester.env.vizdoom_env import Obs, QADoomEnv
from qatester.mapgen.groundtruth import GroundTruth


@dataclass(frozen=True)
class MapEntry:
    name: str  # MAP01, or GEN0000123 for a generated map
    cfg_path: Path
    groundtruth_path: Path
    spec_path: Path


@dataclass(frozen=True)
class Stage:
    until: int  # global env steps
    weights: Mapping[str, float]  # map name -> sampling weight


class CurriculumEnv(gym.Wrapper[Obs, np.int64, Obs, np.int64]):
    def __init__(self, env: QADoomEnv, maps: Mapping[str, MapEntry],
                 stages: Sequence[Stage], coverage: Mapping[str, CoverageTracker] | None,
                 n_envs: int, rng: np.random.Generator, initial_steps: int = 0) -> None:  # fmt: skip
        super().__init__(env)
        if not stages:
            raise ValueError("a curriculum needs at least one stage")
        for st in stages:
            missing = set(st.weights) - set(maps)
            if missing:
                raise ValueError(f"stage until {st.until}: unknown maps {sorted(missing)}")
        self.qa = env
        self.maps = dict(maps)
        self.stages = list(stages)
        self.coverage = dict(coverage or {})
        self.n_envs = n_envs
        self.rng = rng
        self.initial_steps = initial_steps
        self._truth: dict[str, GroundTruth] = {}

    @property
    def global_steps(self) -> int:
        return self.initial_steps + self.qa.total_steps * self.n_envs

    def stage(self) -> Stage:
        steps = self.global_steps
        return next((s for s in self.stages if steps < s.until), self.stages[-1])

    def pick(self) -> str:
        weights = self.stage().weights
        names = sorted(weights)
        p = np.array([weights[n] for n in names], float)
        return names[int(self.rng.choice(len(names), p=p / p.sum()))]

    def truth(self, name: str) -> GroundTruth:
        if name not in self._truth:
            self._truth[name] = GroundTruth.load(self.maps[name].groundtruth_path).intent_only()
        return self._truth[name]

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[Obs, dict[str, Any]]:  # fmt: skip
        name = self.pick()
        entry = self.maps[name]
        self.qa.set_map(entry.cfg_path, self.truth(name), self.coverage.get(name), name)
        obs, info = self.qa.reset(seed=seed, options=options)
        info["map_name"] = name
        return obs, info
