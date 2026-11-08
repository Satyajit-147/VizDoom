"""T4: Go-Explore over ViZDoom save / load (PLAN: T4: the Go-Explore loop; M8).

    loop:
      with probability p_reset (and a non-empty archive for this map): load a cell's save
      otherwise: a new episode
      explore for up to `horizon` steps; every new cell, or a cell reached earlier than
      before, is saved into the archive; new cells are credited to the reset cell

The explorer between resets is a sticky random policy (repeat the last action with
probability `repeat`), as in Ecoffet et al.'s exploration phase, not the PPO policy the
plan's T4 runs: it needs no GPU, and its coverage comes from the archive, which is what
M8 measures (docs/decisions.md D46). Each map has its own archive. Every reset is an
episode boundary for the env and its oracles.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from qatester.agents.base import Obs
from qatester.archive import Archive, Entry, Heuristic, cell_of
from qatester.env.actions import INDEX, N_ACTIONS

USE_IN_FRONT = 17  # observation scalar: something usable is within reach ahead


class StickyRandom:
    """Uniform random actions, each repeated with probability `repeat`, plus USE when
    the observation says something usable is in reach ahead (the scripted tester's
    rule): a door must be pressed while faced, which random play almost never does."""

    def __init__(self, rng: np.random.Generator, repeat: float = 0.9,
                 p_use: float = 0.5) -> None:  # fmt: skip
        self.rng = rng
        self.repeat = repeat
        self.p_use = p_use
        self.last = 0

    def reset(self) -> None:
        self.last = int(self.rng.integers(N_ACTIONS))

    def act(self, obs: Obs) -> int:
        if obs["scalars"][USE_IN_FRONT] > 0.5 and self.last != INDEX["use"] and (
            self.rng.random() < self.p_use
        ):  # fmt: skip
            self.last = INDEX["use"]
            return self.last
        if self.last == INDEX["use"] or self.rng.random() >= self.repeat:
            self.last = int(self.rng.integers(N_ACTIONS))
        return self.last


def save_root() -> Path:
    """Save files go to local disk: /kaggle/temp on Kaggle (PLAN: Kaggle rules)."""
    base = Path("/kaggle/temp")
    return Path(tempfile.mkdtemp(prefix="ge", dir=base if base.is_dir() else None))


@dataclass
class GoExplore:
    p_reset: float = 0.5
    horizon: int = 500
    heuristic: Heuristic = "mixed"
    guard: bool = True
    max_cells: int = 20_000
    repeat: float = 0.9
    root: Path = field(default_factory=save_root)
    archives: dict[str, Archive] = field(default_factory=dict)
    curve_every: int = 10_000  # worker steps between coverage samples
    patience: float = 0.0  # share of steps for runs that play to the episode's end
    patience_min_resets: int = 2
    patience_max_steps: int = 6_000
    steps: int = 0
    iterations: int = 0
    loads: int = 0
    patient_runs: int = 0
    patient_steps: int = 0
    patient_timeouts: int = 0  # patient runs that ended by the time limit (softlock-checkable)
    dead_starts: int = 0  # archived starts that loaded into an already ended episode
    curve: list[dict[str, Any]] = field(default_factory=list)

    def archive(self, name: str) -> Archive:
        if name not in self.archives:
            self.archives[name] = Archive(self.root / name, self.max_cells, self.heuristic,
                                          self.guard)  # fmt: skip
        return self.archives[name]

    def walk(self, env: Any, rng: np.random.Generator) -> Iterator[tuple[dict[str, Any], bool]]:
        """Go-Explore, one env step at a time, forever: yields (step info, episode over).
        `env` is a CurriculumEnv (it picks the map and takes `options["map"]`) or a plain
        QADoomEnv on one map (the demo)."""
        explorer = StickyRandom(rng, self.repeat)
        while True:
            name = env.pick() if hasattr(env, "pick") else env.map_name
            arch = self.archive(name)
            start: Entry | None = None
            # Patience first (only when on: with patience 0 no extra random draw is made,
            # so M8 runs reproduce exactly). A patient run is not cut at the horizon: only
            # an episode that times out lets the softlock family confirm "stuck forever".
            patient = False
            if self.patience > 0 and len(arch) and self.patient_steps < self.patience * self.steps:
                start = arch.sample_confined(rng, self.patience_min_resets)
                patient = start is not None
            if start is None and len(arch) and rng.random() < self.p_reset:
                start = arch.sample(rng)
            if start is not None:
                obs, info = env.reset(options={"map": name, "load_state": start.save_path})
                self.loads += 1
                game = getattr(env.unwrapped, "game", None)
                if game is not None and game.is_episode_finished():
                    # Saved in the last tics before the time limit: it loads into an ended
                    # episode, which can be neither explored nor saved again (ViZDoom
                    # writes no file). Useless as a start, so drop it and pick again.
                    arch.discard(start.cell)
                    self.dead_starts += 1
                    continue
            else:
                obs, info = env.reset(options={"map": name})
            explorer.reset()
            self.iterations += 1
            self.patient_runs += patient
            here = None
            for _ in range(self.patience_max_steps if patient else self.horizon):
                x, y, z = info["pos"]
                c = cell_of(x, y, z, info["keys"], info["doors"])
                if c != here:
                    here = c
                    new = arch.visit(c, int(info["tic"]), self.steps, env.unwrapped.save_state)
                    if start is not None:
                        start.new_cells_after_reset += new
                        start.max_reach = max(start.max_reach, abs(c[0] - start.cell[0]),
                                              abs(c[1] - start.cell[1]))  # fmt: skip
                obs, _, term, trunc, info = env.step(np.int64(explorer.act(obs)))
                self.steps += 1
                self.patient_steps += patient
                yield info, term or trunc
                if term or trunc:
                    self.patient_timeouts += patient and bool(trunc)
                    break

    def run(self, env: Any, steps: int, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Explore for `steps` env steps on a CurriculumEnv; returns finished episodes'
        stats, and samples every map's coverage every `curve_every` steps."""
        episodes: list[dict[str, Any]] = []
        for info, over in self.walk(env, rng):
            if self.steps % self.curve_every == 0:
                self.curve.append({"step": self.steps, "cell": {
                    m: c.summary()["cell_coverage"] for m, c in env.coverage.items()}})
            if over:
                episodes.append(info["episode_stats"])
            if self.steps >= steps:
                break
        return episodes

    def summary(self) -> dict[str, Any]:
        return {"steps": self.steps, "iterations": self.iterations, "loads": self.loads,
                "patient_runs": self.patient_runs, "patient_steps": self.patient_steps,
                "patient_timeouts": self.patient_timeouts, "dead_starts": self.dead_starts,
                "archives": {k: a.summary() for k, a in self.archives.items()},
                "curve": self.curve}  # fmt: skip

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
