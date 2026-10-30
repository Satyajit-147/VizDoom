"""Run one tester configuration end to end and print where its reports went.

    python scripts/train.py --config configs/t1_random_map03.yaml
    python scripts/train.py --config configs/t3a_map01.yaml --seed 1 \\
        --runs-dir /kaggle/working/runs --results-dir /kaggle/working/results

Every tier goes through here: T1/T2 run without learning, T3a+ train PPO (Kaggle only).
"""

from __future__ import annotations

import os

# Before anything imports torch or vizdoom (PLAN: Kaggle operating rule 6).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import faulthandler
import signal
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qatester.config import load_config
from qatester.env.supervisor import exit_on_sigterm
from qatester.runner import run, summarise


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--total-steps", type=int)
    ap.add_argument("--n-envs", type=int)
    ap.add_argument("--runs-dir", type=Path)
    ap.add_argument("--results-dir", type=Path)
    ap.add_argument("--run-id")
    args = ap.parse_args(argv)
    exit_on_sigterm()
    faulthandler.register(signal.SIGUSR1, all_threads=True)  # `kill -USR1` dumps stacks
    overrides: dict[str, Any] = {
        k: v for k, v in {"seed": args.seed, "total_steps": args.total_steps,
                          "n_envs": args.n_envs, "runs_dir": args.runs_dir,
                          "results_dir": args.results_dir}.items() if v is not None
    }  # fmt: skip
    cfg = load_config(args.config, overrides)
    reports = run(cfg, args.run_id)
    print(summarise(reports))
    print("RUN COMPLETE", flush=True)
    # Everything is written. Leave now: a stuck finalizer in a library must not keep a
    # finished run alive for hours (docs/decisions.md D38).
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
