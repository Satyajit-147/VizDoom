"""Run a non-learning tester (T1 random, T2 scripted) for a step budget across worker
processes. Findings and coverage gaps arrive on a shared queue; episode stats come back
from each worker. Learning tiers use ppo_runner instead, with the same env factory."""

from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from qatester.agents.base import Agent
from qatester.agents.random_agent import RandomAgent
from qatester.agents.scripted_agent import WallFollower
from qatester.config import RunConfig
from qatester.coverage import CoverageTracker
from qatester.env.curriculum import MapEntry, Stage
from qatester.env.factory import make_env
from qatester.env.supervisor import exit_on_sigterm


def make_agent(kind: str, rng: np.random.Generator) -> Agent:
    if kind == "random":
        return RandomAgent(rng)
    if kind == "scripted":
        return WallFollower(rng)
    raise ValueError(f"no rollout agent {kind!r}")


@dataclass
class RolloutResult:
    steps: int
    wall_s: float
    episodes: list[dict[str, Any]] = field(default_factory=list)
    queue_items: list[dict[str, Any]] = field(default_factory=list)


def _worker(cfg: RunConfig, entries: Mapping[str, MapEntry], st: list[Stage],
            coverage: Mapping[str, CoverageTracker], queue: Any, rank: int, steps: int,
            record_dir: Path) -> list[dict[str, Any]]:  # fmt: skip
    exit_on_sigterm()
    env = make_env(cfg, entries, st, coverage, queue, rank, record_dir)
    agent = make_agent(cfg.agent, np.random.default_rng([cfg.seed, rank, 1]))
    episodes: list[dict[str, Any]] = []
    try:
        obs, _ = env.reset()
        agent.reset()
        for _ in range(steps):
            obs, _, term, trunc, info = env.step(np.int64(agent.act(obs)))
            if term or trunc:
                episodes.append({"rank": rank, **info["episode_stats"]})
                obs, _ = env.reset()
                agent.reset()
    finally:
        env.close()  # finalizes the cross-episode oracles onto the queue
    return episodes


def run_rollout(cfg: RunConfig, entries: Mapping[str, MapEntry], st: list[Stage],
                coverage: Mapping[str, CoverageTracker], record_dir: Path) -> RolloutResult:  # fmt: skip
    ctx = mp.get_context("spawn")
    n = cfg.n_envs
    share = [cfg.total_steps // n + (1 if r < cfg.total_steps % n else 0) for r in range(n)]
    t0 = time.perf_counter()
    with ctx.Manager() as manager:
        queue = manager.Queue()
        with ctx.Pool(n) as pool:
            outs = pool.starmap(_worker, [
                (cfg, entries, st, coverage, queue, r, share[r], record_dir) for r in range(n)
            ])  # fmt: skip
        items = []
        while not queue.empty():
            items.append(queue.get())
    result = RolloutResult(cfg.total_steps, time.perf_counter() - t0, queue_items=items)
    result.episodes = [e for out in outs for e in out]
    return result
