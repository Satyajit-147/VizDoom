"""M8: the Go-Explore archive (qatester/archive.py) and loop (qatester/agents/goexplore.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qatester.archive import Archive, cell_of
from qatester.config import load_config
from qatester.runner import run

REPO = Path(__file__).resolve().parents[1]


def _saver(calls: list[Path]):  # type: ignore[no-untyped-def]
    def save(path: Path) -> None:
        calls.append(path)
        path.write_bytes(b"s")

    return save


def test_cells_separate_floors_keys_and_doors() -> None:
    base = cell_of(100, 100, 0, [])
    assert cell_of(120, 90, 10, []) == base  # same 64 mu cell and z bucket
    assert cell_of(100, 100, 64, []) != base  # a balcony above
    assert cell_of(100, 100, 0, ["key_red"]) != base
    assert cell_of(100, 100, 0, [], doors=1) != base  # the moment a door is open (D46)


def test_archive_saves_new_cells_and_earlier_arrivals(tmp_path: Path) -> None:
    a = Archive(tmp_path)
    calls: list[Path] = []
    c = cell_of(0, 0, 0, [])
    assert a.visit(c, depth=100, step=1, save=_saver(calls)) is True
    assert a.visit(c, depth=200, step=2, save=_saver(calls)) is False  # later: kept
    assert len(calls) == 1 and a.entries[c].visits == 2
    assert a.visit(c, depth=50, step=3, save=_saver(calls)) is False  # earlier: re-saved
    assert len(calls) == 2 and a.entries[c].depth == 50 and a.entries[c].best_return == -50


def test_lru_eviction_removes_the_save_file(tmp_path: Path) -> None:
    a = Archive(tmp_path, max_cells=2)
    calls: list[Path] = []
    cells = [cell_of(64 * i, 0, 0, []) for i in range(3)]
    a.visit(cells[0], 1, 1, _saver(calls))
    a.visit(cells[1], 1, 2, _saver(calls))
    a.visit(cells[0], 1, 3, _saver(calls))  # touched: cells[1] is now least recent
    a.visit(cells[2], 1, 4, _saver(calls))
    assert set(a.entries) == {cells[0], cells[2]} and a.evicted == 1
    assert not calls[1].exists()


def test_heuristics_and_productivity_guard(tmp_path: Path) -> None:
    a = Archive(tmp_path, heuristic="count")
    calls: list[Path] = []
    inner = [cell_of(64 * i, 64 * j, 0, []) for i in range(5) for j in range(5)]
    for c in inner:
        a.visit(c, 1, 1, _saver(calls))
    middle, corner = cell_of(128, 128, 0, []), cell_of(0, 0, 0, [])
    for _ in range(8):
        a.visit(middle, 1, 2, _saver(calls))
    w = dict(zip(a.entries, a.weights("count"), strict=True))
    assert w[middle] < w[corner]  # visited more: sampled less
    w = dict(zip(a.entries, a.weights("frontier"), strict=True))
    assert w[corner] > w[middle]  # the corner borders unreached cells; the middle does not
    assert set(a.weights("uniform")) == {1.0}
    a.entries[corner].resets_from = 20  # reset to 20 times, never paid off
    w = dict(zip(a.entries, a.weights("uniform"), strict=True))
    assert w[corner] < 0.1 * w[middle]
    a.guard = False
    assert set(a.weights("uniform")) == {1.0}
    e = a.sample(np.random.default_rng(0))
    assert e.resets_from >= 1


def test_goexplore_run_loads_saves_and_explores_with_the_clock_running(tmp_path: Path) -> None:
    """End to end through the runner on MAP01. Regression (D46): in a process that had
    recorded a demo, the game clock froze after every load, so each loaded stretch was
    wasted and the crash oracle reported tic stalls."""
    cfg = load_config(REPO / "configs" / "t4_goexplore_map07.yaml", {
        "runs_dir": tmp_path / "runs", "results_dir": tmp_path / "results",
        "total_steps": 6_000, "n_envs": 1,
        "maps": {"variant": "fixed",
                 "curriculum": [{"until": 6_000, "maps": ["MAP01"], "weights": [1.0]}]},
        "goexplore": {"horizon": 200},
    })  # fmt: skip
    reports = run(cfg)
    rep = json.loads(reports["MAP01"].with_name("report.json").read_text())
    assert rep["findings"] == [], "a false alarm on the clean map (a tic stall after a load?)"
    run_dir = tmp_path / "runs" / "t4_goexplore_map07_seed0"
    summary = json.loads((run_dir / "goexplore.jsonl").read_text().splitlines()[0])["summary"]
    assert summary["steps"] == 6_000 and summary["loads"] >= 5
    assert summary["archives"]["MAP01"]["cells"] >= 20
    assert summary["archives"]["MAP01"]["productive_resets"] >= 1
    assert rep["coverage"]["cell"] > 0.2
