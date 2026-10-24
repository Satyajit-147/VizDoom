"""Env throughput (PLAN: Robustness and throughput; M5 acceptance: > 400 steps/s with
one worker). Run it on the machine you train on: a Kaggle GPU session.

    python scripts/bench_env.py --workers 1 2 4 16 --seconds 20

For each worker count, that many processes each step their own QADoomEnv (random
actions, all oracles live, one shared CoverageTracker) for `--seconds` after a warm-up.
Also reports a bare-ViZDoom loop (make_action + get_state, no env) for one worker, and
the share of each env step spent inside the engine, to separate engine cost from ours.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from qatester.coverage import CoverageTracker
from qatester.mapgen.cli import artefact_name, build
from qatester.mapgen.groundtruth import GroundTruth


def _paths(build_dir: Path, lump: str, variant: str) -> tuple[Path, Path]:
    v = variant  # type: ignore[assignment]
    return (build_dir / artefact_name(lump, v, "cfg"),  # type: ignore[arg-type]
            build_dir / artefact_name(lump, v, "groundtruth.json"))  # type: ignore[arg-type]  # fmt: skip


def _env_worker(cfg: str, gt_path: str, cov: CoverageTracker, seed: int, warmup: int,
                seconds: float, barrier: Any) -> dict[str, float]:  # fmt: skip
    from qatester.env.vizdoom_env import QADoomEnv

    env = QADoomEnv(Path(cfg), GroundTruth.load(Path(gt_path)), cov, seed=seed)
    rng = np.random.default_rng(seed)
    env.reset()
    for _ in range(warmup):
        *_, term, trunc, _ = env.step(int(rng.integers(12)))
        if term or trunc:
            env.reset()
    barrier.wait()
    steps, engine_s, resets = 0, 0.0, 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        *_, term, trunc, _ = env.step(int(rng.integers(12)))
        steps += 1
        engine_s += env.last.step_wall_s if env.last is not None else 0.0
        if term or trunc:
            env.reset()
            resets += 1
    wall = time.perf_counter() - t0
    env.close()
    return {"steps": steps, "wall_s": wall, "engine_s": engine_s, "resets": resets}


def _raw_worker(cfg: str, seed: int, warmup: int, seconds: float) -> dict[str, float]:
    import vizdoom as vzd

    from qatester.env.actions import ACTIONS, BUTTONS

    g = vzd.DoomGame()
    g.load_config(cfg)
    g.set_available_buttons([getattr(vzd.Button, b) for b in BUTTONS])
    g.set_seed(seed)
    g.init()
    rng = np.random.default_rng(seed)
    for _ in range(warmup):
        g.make_action(ACTIONS[int(rng.integers(12))].buttons, 4)
    steps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        if g.is_episode_finished():
            g.new_episode()
        g.make_action(ACTIONS[int(rng.integers(12))].buttons, 4)
        g.get_state()
        steps += 1
    wall = time.perf_counter() - t0
    g.close()
    return {"steps": steps, "wall_s": wall}


def run(args: argparse.Namespace) -> dict[str, Any]:
    build_dir = Path(args.build)
    cfg, gt_path = _paths(build_dir, args.map, args.variant)
    if not cfg.exists():
        build(Path(args.maps), build_dir, args.variant)
    ctx = mp.get_context("spawn")
    gt = GroundTruth.load(gt_path)
    results: dict[str, Any] = {
        "map": args.map, "variant": args.variant, "seconds": args.seconds,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "cpus": os.cpu_count(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "runs": [],
    }  # fmt: skip
    with ctx.Pool(1) as pool:
        raw = pool.apply(_raw_worker, (str(cfg), 0, args.warmup, args.seconds))
    results["raw_vizdoom_steps_per_s"] = raw["steps"] / raw["wall_s"]
    print(f"bare ViZDoom, 1 worker: {results['raw_vizdoom_steps_per_s']:7.1f} steps/s")
    for n in args.workers:
        cov = CoverageTracker(gt)
        barrier = ctx.Manager().Barrier(n)
        try:
            with ctx.Pool(n) as pool:
                outs = pool.starmap(_env_worker, [
                    (str(cfg), str(gt_path), cov, 1000 + i, args.warmup, args.seconds, barrier)
                    for i in range(n)
                ])  # fmt: skip
        finally:
            summary = cov.summary()
            cov.close()
        per = [o["steps"] / o["wall_s"] for o in outs]
        engine_share = sum(o["engine_s"] for o in outs) / sum(o["wall_s"] for o in outs)
        row = {"workers": n, "steps_per_s_per_worker": float(np.mean(per)),
               "steps_per_s_total": float(np.sum(per)), "engine_share": engine_share,
               "episodes_finished": int(sum(o["resets"] for o in outs)),
               "coverage": summary}  # fmt: skip
        results["runs"].append(row)
        print(f"QADoomEnv, {n:2d} worker(s): {row['steps_per_s_per_worker']:7.1f} steps/s each,"
              f" {row['steps_per_s_total']:7.1f} total, {100 * engine_share:4.1f}% in the engine")
    single = next((r for r in results["runs"] if r["workers"] == 1), None)
    if single is not None:
        ok = single["steps_per_s_per_worker"] > args.target
        results["m5_target_met"] = ok
        print(f"M5 target (> {args.target:.0f} steps/s, 1 worker): {'MET' if ok else 'NOT MET'}")
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--maps", default=str(REPO / "maps"))
    ap.add_argument("--build", default=str(REPO / "build"))
    ap.add_argument("--map", default="MAP07", help="the largest map by default")
    ap.add_argument("--variant", default="buggy", choices=("buggy", "fixed"))
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--target", type=float, default=400.0)
    ap.add_argument("--out", default=None, help="write the results as JSON here")
    args = ap.parse_args(argv)
    results = run(args)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
