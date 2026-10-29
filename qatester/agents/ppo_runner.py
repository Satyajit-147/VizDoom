"""T3a/T3b training with sb3-contrib's RecurrentPPO (PLAN: The training loop, concretely;
Kaggle operating rules). Runs on Kaggle only.

- n_envs workers in subprocesses (SubprocVecEnv, spawn), each a curriculum env.
- VecNormalize on rewards only; the depth observation is never normalised.
- Learning rate linear to 0, clip range 0.2 -> 0.1, entropy 0.01 -> 0.003 after 500k.
- Every `ckpt_every` steps: model, VecNormalize stats, coverage counts and a pointer to
  them. A new session with the same run id resumes from the newest checkpoint.
- Findings stream from the workers on a queue and are appended to findings.jsonl as
  they arrive, so nothing is lost when a session dies.
- Every `metrics_every` steps a line goes to metrics.jsonl; every `eval.every` steps the
  policy plays `eval.episodes` episodes on each eval map (fixed variant), and the goal
  rate goes to eval.jsonl.
- The run stops cleanly before `max_session_hours`.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecMonitor,
    VecNormalize,
)

from qatester.agents.policy import policy_kwargs
from qatester.config import RunConfig
from qatester.coverage import CoverageTracker
from qatester.env.curriculum import MapEntry, Stage
from qatester.env.factory import eval_env, make_env, prepare_maps
from qatester.report.cluster import cluster_findings
from qatester.runner import append_jsonl, coverage_summary, queue_rows, read_jsonl


def _latest(ckpt: Path) -> dict[str, Any] | None:
    p = ckpt / "latest.json"
    return json.loads(p.read_text()) if p.exists() else None


def _schedule_lr(cfg: RunConfig) -> Any:
    lr = cfg.ppo.learning_rate
    return (lambda p: lr * p) if cfg.ppo.lr_schedule == "linear" else lr


def _schedule_clip(cfg: RunConfig) -> Any:
    a, b = cfg.ppo.clip_range, cfg.ppo.clip_range_end
    return lambda p: b + (a - b) * p


def play_episode(model: RecurrentPPO, env: Any, deterministic: bool) -> dict[str, Any]:
    obs, _ = env.reset()
    state = None
    start = np.ones((1,), dtype=bool)
    while True:
        action, state = model.predict(obs, state=state, episode_start=start,
                                      deterministic=deterministic)  # fmt: skip
        start[:] = False
        obs, _, term, trunc, info = env.step(int(action))
        if term or trunc:
            return dict(info["episode_stats"])


class QACallback(BaseCallback):  # type: ignore[misc]
    """Drains findings, logs metrics, schedules entropy, evaluates and checkpoints."""

    def __init__(self, cfg: RunConfig, run_dir: Path, queue: Any,
                 coverage: Mapping[str, CoverageTracker], eval_entries: Mapping[str, MapEntry],
                 initial_steps: int) -> None:  # fmt: skip
        super().__init__()
        self.cfg = cfg
        self.run_dir = run_dir
        self.ckpt = run_dir / "ckpt"
        self.queue = queue
        self.coverage = coverage
        self.eval_entries = eval_entries
        self.initial_steps = initial_steps
        self.started = time.time()
        self.findings: list[dict[str, Any]] = read_jsonl(run_dir / "findings.jsonl")
        self.episodes: list[dict[str, Any]] = []
        self.next_metrics = self.next_ckpt = self.next_eval = 0
        self.last_t, self.last_steps = time.time(), initial_steps
        self.stopped_early = False

    def _on_training_start(self) -> None:
        n = self.model.num_timesteps
        self.next_metrics = n + self.cfg.metrics_every
        self.next_ckpt = n + self.cfg.ckpt_every
        self.next_eval = n + self.cfg.eval.every
        self.last_steps = n

    def drain(self) -> None:
        items = []
        while True:
            try:
                items.append(self.queue.get_nowait())
            except Exception:  # noqa: BLE001 - queue.Empty through the manager proxy
                break
        findings, gaps = queue_rows(items, self.cfg.n_envs, self.initial_steps)
        append_jsonl(self.run_dir / "findings.jsonl", findings)
        append_jsonl(self.run_dir / "coverage_gaps.jsonl", gaps)
        self.findings += findings

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode_stats" in info:
                row = {"step": self.num_timesteps, **info["episode_stats"]}
                self.episodes.append(row)
                append_jsonl(self.run_dir / "episodes.jsonl", [row])
        n = self.num_timesteps
        p = self.cfg.ppo
        if n >= p.ent_decay_after:
            self.model.ent_coef = p.ent_coef_end
        if n >= self.next_metrics:
            self.next_metrics += self.cfg.metrics_every
            self.drain()
            self.log_metrics()
        if n >= self.next_eval:
            self.next_eval += self.cfg.eval.every
            self.evaluate(self.cfg.eval.episodes)
        if n >= self.next_ckpt:
            self.next_ckpt += self.cfg.ckpt_every
            self.checkpoint()
        if time.time() - self.started > self.cfg.max_session_hours * 3600:
            print(f"session limit reached at {n} steps: checkpointing and stopping", flush=True)
            self.checkpoint()
            self.stopped_early = True
            return False
        return True

    def log_metrics(self) -> None:
        n = self.num_timesteps
        now = time.time()
        eps, self.episodes = self.episodes, []
        clusters: dict[str, dict[str, int]] = {}
        by_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for f in self.findings:
            by_map[f["map"]].append(f)
        for m, fs in by_map.items():
            fam: dict[str, int] = defaultdict(int)
            for c in cluster_findings(fs, self.cfg.report.dbscan_eps_mu,
                                      self.cfg.report.dbscan_min_samples):  # fmt: skip
                fam[c.oracle] += 1
            clusters[m] = dict(fam)

        def mean(k: str) -> float | None:
            return float(np.mean([e[k] for e in eps])) if eps else None

        row = {
            "step": n, "wall_s": round(now - self.started, 1),
            "steps_per_s": round((n - self.last_steps) / max(1e-9, now - self.last_t), 1),
            "episodes": len(eps), "return": mean("return"), "validity": mean("validity"),
            "adversarial": mean("adversarial"), "length": mean("steps"),
            "death_rate": (sum(e["outcome"] == "death" for e in eps) / len(eps)) if eps else None,
            "goal_rate": (sum(e["outcome"] == "goal" for e in eps) / len(eps)) if eps else None,
            "coverage": {m: coverage_summary(c) for m, c in self.coverage.items()},
            "clusters_by_family": clusters, "raw_findings": len(self.findings),
            "ent_coef": float(self.model.ent_coef),
        }  # fmt: skip
        self.last_t, self.last_steps = now, n
        append_jsonl(self.run_dir / "metrics.jsonl", [row])
        print(f"[{n:>9,}] {row['steps_per_s']:6.0f} steps/s  episodes {len(eps):3d}  "
              f"return {row['return'] if eps else float('nan'):7.2f}  goal "
              f"{row['goal_rate'] if eps else float('nan'):.2f}  findings "
              f"{len(self.findings)}", flush=True)  # fmt: skip

    def evaluate(self, episodes: int) -> dict[str, Any]:
        rows = {}
        for name in self.cfg.eval.maps:
            env = eval_env(self.cfg, self.eval_entries, name, seed=10_000 + self.num_timesteps)
            try:
                eps = [play_episode(self.model, env, self.cfg.eval.deterministic)
                       for _ in range(episodes)]  # fmt: skip
            finally:
                env.close()
            rows[name] = {"episodes": episodes,
                          "goal_rate": sum(e["outcome"] == "goal" for e in eps) / episodes,
                          "mean_return": float(np.mean([e["return"] for e in eps])),
                          "mean_steps": float(np.mean([e["steps"] for e in eps]))}  # fmt: skip
        row = {"step": self.num_timesteps, "maps": rows}
        append_jsonl(self.run_dir / "eval.jsonl", [row])
        print(f"[{self.num_timesteps:>9,}] eval " + ", ".join(
            f"{m}: goal {r['goal_rate']:.2f} in {r['mean_steps']:.0f} steps"
            for m, r in rows.items()), flush=True)  # fmt: skip
        return row

    def checkpoint(self) -> None:
        self.drain()
        self.ckpt.mkdir(parents=True, exist_ok=True)
        n = self.num_timesteps
        self.model.save(self.ckpt / "model.zip")
        venv = self.model.get_vec_normalize_env()
        if venv is not None:
            venv.save(str(self.ckpt / "vecnormalize.pkl"))
        for name, c in self.coverage.items():
            c.save(self.ckpt / f"coverage_{name}.npy")
        tmp = self.ckpt / "latest.json.tmp"
        tmp.write_text(json.dumps({"steps": n, "time": time.time()}))
        tmp.replace(self.ckpt / "latest.json")
        print(f"[{n:>9,}] checkpoint", flush=True)


def train(cfg: RunConfig, run_id: str, run_dir: Path, entries: Mapping[str, MapEntry],
          st: list[Stage], coverage: Mapping[str, CoverageTracker]) -> int:  # fmt: skip
    """Train (or resume) PPO; returns the env steps done so far."""
    th.set_num_threads(1)
    ckpt = run_dir / "ckpt"
    latest = _latest(ckpt)
    initial = int(latest["steps"]) if latest else 0
    if latest:
        for name, c in coverage.items():
            saved = ckpt / f"coverage_{name}.npy"
            if saved.exists():
                c.restore(saved)
        print(f"resuming {run_id} from {initial:,} steps", flush=True)
    if initial >= cfg.total_steps:
        return initial
    eval_entries = prepare_maps(cfg.maps, run_dir / "build_eval", variant=cfg.eval.variant)

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    queue = manager.Queue()
    fns = [partial(make_env, cfg, entries, st, coverage, queue, r, run_dir, initial)
           for r in range(cfg.n_envs)]  # fmt: skip
    base: VecEnv = SubprocVecEnv(fns, start_method="spawn") if cfg.n_envs > 1 else DummyVecEnv(fns)
    base = VecMonitor(base)
    if latest and (ckpt / "vecnormalize.pkl").exists():
        venv = VecNormalize.load(str(ckpt / "vecnormalize.pkl"), base)
    else:
        venv = VecNormalize(base, norm_obs=False, norm_reward=True, gamma=cfg.ppo.gamma)
    p = cfg.ppo
    if latest:
        model = RecurrentPPO.load(ckpt / "model.zip", env=venv, device=cfg.device)
    else:
        model = RecurrentPPO(
            "MultiInputLstmPolicy", venv, n_steps=p.n_steps, batch_size=p.batch_size,
            n_epochs=p.n_epochs, learning_rate=_schedule_lr(cfg), gamma=p.gamma,
            gae_lambda=p.gae_lambda, clip_range=_schedule_clip(cfg), ent_coef=p.ent_coef,
            vf_coef=p.vf_coef, max_grad_norm=p.max_grad_norm, seed=cfg.seed,
            device=cfg.device, policy_kwargs=policy_kwargs(p.lstm_hidden_size), verbose=0,
        )  # fmt: skip
    print(f"PPO on {model.device}: {sum(t.numel() for t in model.policy.parameters()):,} "
          f"parameters, {cfg.n_envs} envs", flush=True)  # fmt: skip
    cb = QACallback(cfg, run_dir, queue, coverage, eval_entries, initial)
    try:
        model.learn(total_timesteps=cfg.total_steps - initial, callback=cb,
                    reset_num_timesteps=not latest)  # fmt: skip
        if not cb.stopped_early:
            cb.checkpoint()
            cb.evaluate(cfg.eval.final_episodes)
    finally:
        venv.close()  # workers finalize their oracles onto the queue
        time.sleep(1.0)
        cb.drain()
        manager.shutdown()
    return int(model.num_timesteps)
