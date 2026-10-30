"""M6 end to end without learning: config -> maps -> T1/T2 rollout -> reports, plus the
curriculum's map switching. (PPO is tested on Kaggle: tests/test_ppo_smoke.py.)"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qatester.config import load_config
from qatester.env.curriculum import CurriculumEnv, Stage
from qatester.env.factory import make_coverage, make_env, prepare_maps, stages
from qatester.runner import run

REPO = Path(__file__).resolve().parents[1]


def _cfg(name: str, tmp: Path, **over: object) -> object:
    return load_config(REPO / "configs" / name, {"runs_dir": tmp / "runs",
                                                 "results_dir": tmp / "results", **over})  # fmt: skip


def test_t1_random_on_map03_reports_the_leak_with_a_working_demo(tmp_path: Path) -> None:
    cfg = _cfg("t1_random_map03.yaml", tmp_path, total_steps=16_000, n_envs=2)
    reports = run(cfg)  # type: ignore[arg-type]
    rep = json.loads(reports["MAP03"].with_name("report.json").read_text())
    oob = [f for f in rep["findings"] if f["oracle"] == "out_of_bounds"]
    assert oob, rep["findings"]
    demo = reports["MAP03"].parent / oob[0]["repro"]["demo"]
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "replay.py"), str(demo), "--wad",
         str(reports["MAP03"].parent / rep["wad"]), "--map", rep["map_lump"], "--headless"],
        capture_output=True, text=True, timeout=300, check=False,
        env={**os.environ, "SDL_VIDEODRIVER": "dummy"},
    )  # fmt: skip
    assert out.returncode == 0 and "replayed" in out.stdout, out.stderr
    run_dir = tmp_path / "runs" / "t1_random_map03_seed0"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["env_steps"] == 16_000 and manifest["agent"] == "random"
    assert len(manifest["wad_sha256"]) == 1
    assert (run_dir / "episodes.jsonl").read_text().count("\n") >= 4


def test_t2_scripted_on_clean_map01_finds_nothing(tmp_path: Path) -> None:
    cfg = _cfg("t2_scripted_map03.yaml", tmp_path, total_steps=8_000, n_envs=2,
               maps={"variant": "fixed",
                     "curriculum": [{"until": 8_000, "maps": ["MAP01"], "weights": [1.0]}]})  # fmt: skip
    reports = run(cfg)  # type: ignore[arg-type]
    rep = json.loads(reports["MAP01"].with_name("report.json").read_text())
    assert rep["findings"] == [], "a false alarm on the clean map"
    assert rep["coverage"]["cell"] > 0.05


def test_curriculum_switches_maps_within_and_across_wads(tmp_path: Path) -> None:
    cfg = load_config(REPO / "configs" / "t1_random_map03.yaml", {
        "maps": {"generated": {"split": "train", "count": 2},
                 "curriculum": [{"until": 10, "maps": ["MAP01", "MAP02"], "weights": [1, 1],
                                 "generated_weight": 1.0}]}})  # fmt: skip
    entries = prepare_maps(cfg.maps, tmp_path / "build")
    gen = sorted(n for n in entries if n.startswith("GEN"))
    assert len(gen) == 2 and {"MAP01", "MAP08"} <= set(entries)
    st = stages(cfg.maps, entries)
    assert st[0].weights[gen[0]] == pytest.approx(0.5)
    cov = make_coverage(entries, {"MAP01", "MAP02", *gen})
    env = make_env(cfg, entries, st, cov, None, 0, tmp_path / "rec")
    try:
        qa = env.qa
        seen = []
        for name in ["MAP01", "MAP02", gen[0], gen[1], "MAP01"]:
            env.pick = lambda name=name: name  # type: ignore[method-assign]
            _, info = env.reset()
            spawn = env.truth(name).data["spawn"]["pos"]
            assert info["map_name"] == name and qa.map_name == name
            assert info["pos"][:2] == pytest.approx(spawn, abs=1.0)
            for _ in range(5):
                env.step(1)
            seen.append(qa.supervisor.restarts)
        # The env starts on the first map by name, a generated one, so going to MAP01 is
        # already a WAD change. MAP01 -> MAP02 shares a WAD (no restart); corpus <->
        # generated pack does not.
        assert seen == [1, 1, 2, 2, 3]
        env.reset()
        demos = sorted(p.name for p in (tmp_path / "rec" / "demos").glob("*.lmp"))
        assert len(demos) == 5 and demos[0].startswith("GEN")  # moved out of the temp dir
    finally:
        env.close()
        for c in cov.values():
            c.close()


def test_curriculum_stage_follows_global_steps() -> None:
    class Fake:
        total_steps = 0

    st = [Stage(100, {"A": 1.0}), Stage(200, {"B": 1.0})]
    env = CurriculumEnv.__new__(CurriculumEnv)
    env.qa, env.stages, env.n_envs, env.initial_steps = Fake(), st, 2, 0  # type: ignore[assignment]
    env.rng = np.random.default_rng(0)
    assert env.pick() == "A"
    Fake.total_steps = 50  # x 2 envs = 100: the second stage
    assert env.pick() == "B"
    Fake.total_steps = 10_000  # past the last stage: stay on it
    assert env.pick() == "B"


def test_evaluation_sees_a_private_copy_of_the_training_counts(tmp_path: Path) -> None:
    """D39: evaluation must show the policy the novelty inputs it trained with, without
    changing the training counts."""
    from qatester.env.factory import snapshot_coverage

    cfg = load_config(REPO / "configs" / "t3a_map01.yaml")
    entries = prepare_maps(cfg.maps, tmp_path / "build")
    (train,) = make_coverage(entries, {"MAP01"}).values()
    snap = None
    try:
        cell = (4, 3)
        for _ in range(7):
            train.visit_cell(cell)
        snap = snapshot_coverage(entries, "MAP01", train)
        assert snap is not None and snap.cell_count(cell) == 7
        snap.visit_cell(cell)
        assert train.cell_count(cell) == 7, "evaluation changed the training counts"
        train.save(tmp_path / "c.npy")
        again = snapshot_coverage(entries, "MAP01", tmp_path / "c.npy")
        assert again is not None and again.cell_count(cell) == 7
        again.close()
        assert snapshot_coverage(entries, "MAP01", None) is None
    finally:
        train.close()
        if snap is not None:
            snap.close()
