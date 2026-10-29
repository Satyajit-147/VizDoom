"""T3a wiring on a tiny budget: the network's shapes, a short PPO run that checkpoints,
logs, evaluates and reports, and a second session that resumes it. Needs torch and
sb3-contrib, so it only runs on Kaggle (it is skipped elsewhere)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("sb3_contrib")

import torch as th

from qatester.agents.policy import QAFeatures
from qatester.config import load_config
from qatester.env.observation import observation_space
from qatester.runner import read_jsonl, run

REPO = Path(__file__).resolve().parents[1]


def test_encoder_shapes() -> None:
    space = observation_space()
    enc = QAFeatures(space)
    obs = {"depth": th.rand(3, 4, 64, 64), "scalars": th.rand(3, 18)}
    assert enc(obs).shape == (3, 320)
    assert enc.cnn(th.zeros(1, 4, 64, 64)).shape == (1, 1024)  # 64 x 4 x 4, as planned


def _cfg(tmp: Path, total: int) -> object:
    return load_config(REPO / "configs" / "t3a_map01.yaml", {
        "total_steps": total, "runs_dir": tmp / "runs", "results_dir": tmp / "results",
        "ckpt_every": 1024, "metrics_every": 512,
        "ppo": {"n_steps": 256, "batch_size": 128, "n_epochs": 1},
        "eval": {"every": 10**9, "episodes": 1, "final_episodes": 1, "maps": ["MAP01"],
                 "variant": "fixed"},
    })  # fmt: skip


def test_ppo_trains_checkpoints_and_resumes(tmp_path: Path) -> None:
    reports = run(_cfg(tmp_path, 2048))  # type: ignore[arg-type]
    run_dir = tmp_path / "runs" / "t3a_map01_seed0"
    latest = json.loads((run_dir / "ckpt" / "latest.json").read_text())
    assert latest["steps"] >= 2048
    assert (run_dir / "ckpt" / "model.zip").exists()
    assert (run_dir / "ckpt" / "vecnormalize.pkl").exists()
    assert (run_dir / "ckpt" / "coverage_MAP01.npy").exists()
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    assert metrics and metrics[-1]["coverage"]["MAP01"]["cell"] > 0
    assert read_jsonl(run_dir / "eval.jsonl")[-1]["maps"]["MAP01"]["episodes"] == 1
    assert reports["MAP01"].exists()

    reports = run(_cfg(tmp_path, 4096))  # type: ignore[arg-type]  # a "new session"
    latest = json.loads((run_dir / "ckpt" / "latest.json").read_text())
    assert latest["steps"] >= 4096
    steps = [m["step"] for m in read_jsonl(run_dir / "metrics.jsonl")]
    assert steps == sorted(steps) and steps[-1] >= 4096, "the resumed run restarted at 0"
