"""Evaluate a saved PPO checkpoint: N episodes on a map, then the goal rate.

    python scripts/eval_policy.py --config configs/t3a_map01.yaml \\
        --ckpt runs/t3a_map01_seed0/ckpt --map MAP01 --episodes 20 --novelty both

`--novelty snapshot` gives the policy the visit counts it trained with (the checkpoint's
coverage_<map>.npy); `off` runs without counts, as evaluation did before D39; `both`
runs each and prints them side by side. Inference only, on CPU by default.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sb3_contrib import RecurrentPPO

from qatester.agents.ppo_runner import play_episode
from qatester.config import load_config
from qatester.env.factory import eval_env, prepare_maps, snapshot_coverage


def evaluate(model: RecurrentPPO, cfg: Any, entries: Any, name: str, episodes: int,
             counts: Path | None, seed: int) -> dict[str, Any]:  # fmt: skip
    snap = snapshot_coverage(entries, name, counts)
    env = eval_env(cfg, entries, name, seed=seed, coverage=snap)
    t0 = time.time()
    try:
        eps = [play_episode(model, env, deterministic=False) for _ in range(episodes)]
    finally:
        env.close()
        if snap is not None:
            snap.close()
    return {
        "map": name, "novelty": "snapshot" if counts else "off", "episodes": episodes,
        "goal_rate": sum(e["outcome"] == "goal" for e in eps) / episodes,
        "key_rate": sum(e["parts"].get("progress", 0) >= 1 for e in eps) / episodes,
        "door_rate": sum(e["parts"].get("progress", 0) >= 2 for e in eps) / episodes,
        "mean_steps": sum(e["steps"] for e in eps) / episodes,
        "mean_return": sum(e["return"] for e in eps) / episodes,
        "wall_s": round(time.time() - t0, 1),
    }  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True, help="folder with model.zip")
    ap.add_argument("--map", default="MAP01")
    ap.add_argument("--variant", default="fixed", choices=("buggy", "fixed"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--novelty", default="both", choices=("snapshot", "off", "both"))
    ap.add_argument("--build", type=Path, default=Path("/tmp/qa_eval_build"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    entries = prepare_maps(cfg.maps, args.build, variant=args.variant)
    model = RecurrentPPO.load(args.ckpt / "model.zip", device=args.device)
    counts = args.ckpt / f"coverage_{args.map}.npy"
    modes = ["snapshot", "off"] if args.novelty == "both" else [args.novelty]
    rows = []
    for mode in modes:
        row = evaluate(model, cfg, entries, args.map, args.episodes,
                       counts if mode == "snapshot" else None, args.seed)  # fmt: skip
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1) + "\n")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
