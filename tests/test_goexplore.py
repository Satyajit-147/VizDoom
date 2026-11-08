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


class _Corridor:
    """A fake env, no engine: the player random-walks along x one 64 mu cell per step and
    the episode times out `limit` steps after its reset (also after a load). A load puts
    the player where the save was made. Records each episode's length."""

    map_name = "M"

    def __init__(self, limit: int, dead_after: int | None = None) -> None:
        self.limit, self.x, self.t = limit, 0.0, 0
        self.saves: dict[Path, float] = {}
        self.late: set[Path] = set()  # saves that load into an already ended episode
        self.dead_after, self.finished = dead_after, False
        self.lengths: list[int] = []
        self.unwrapped = self
        self.game = self

    def is_episode_finished(self) -> bool:
        return self.finished

    def save_state(self, path: Path) -> None:
        assert not self.finished, "saved a finished episode (ViZDoom would write nothing)"
        self.saves[Path(path)] = self.x
        if self.dead_after is not None and self.t >= self.dead_after:
            self.late.add(Path(path))
        Path(path).write_bytes(b"s")

    def _info(self) -> dict:  # type: ignore[type-arg]
        return {"pos": (self.x, 0.0, 0.0), "keys": [], "doors": 0, "tic": 4 * self.t,
                "episode_stats": {}}  # fmt: skip

    def reset(self, options: dict | None = None):  # type: ignore[no-untyped-def,type-arg]
        load = (options or {}).get("load_state")
        self.x = self.saves[Path(load)] if load else 0.0
        self.finished = load is not None and Path(load) in self.late
        self.t = 0
        self.lengths.append(0)
        return {"scalars": np.zeros(18)}, self._info()

    def step(self, action):  # type: ignore[no-untyped-def]
        self.x += 64.0 if int(action) % 2 else -64.0
        self.t += 1
        self.lengths[-1] += 1
        return {"scalars": np.zeros(18)}, 0.0, False, self.t >= self.limit, self._info()


def _walk(ge, env, steps: int) -> None:  # type: ignore[no-untyped-def]
    for _ in ge.walk(env, np.random.default_rng(0)):
        if ge.steps >= steps:
            return


def test_patience_off_keeps_every_run_within_the_horizon(tmp_path: Path) -> None:
    from qatester.agents.goexplore import GoExplore

    env = _Corridor(limit=1_000)
    ge = GoExplore(horizon=10, root=tmp_path)
    _walk(ge, env, 3_000)
    assert max(env.lengths) <= 10 and ge.patient_runs == 0 and ge.loads > 0


def test_patient_runs_play_to_the_time_limit_and_keep_to_their_share(tmp_path: Path) -> None:
    """CONTRIBUTING 'Go-Explore patience': a softlock is confirmed only by an episode that
    times out, which a run cut every `horizon` steps never does."""
    from qatester.agents.goexplore import GoExplore

    env = _Corridor(limit=200)
    ge = GoExplore(horizon=10, root=tmp_path, patience=0.2)
    _walk(ge, env, 20_000)
    assert ge.patient_runs >= 5 and ge.patient_timeouts == ge.patient_runs
    assert max(env.lengths) == 200  # patient runs are not cut at the horizon
    share = ge.patient_steps / ge.steps
    assert 0.1 < share <= 0.2 + 200 / 20_000  # its budget share, overshooting by one run at most
    assert ge.summary()["patient_timeouts"] == ge.patient_timeouts


def test_a_save_that_loads_into_an_ended_episode_is_dropped_not_saved_again(
    tmp_path: Path,
) -> None:
    """Regression (MAP07 patience seed 3 crashed): a cell first reached in the last tics
    before the time limit loads into an already ended episode, and ViZDoom writes no file
    when asked to save one, so the next save crashed the run. Such starts are dropped."""
    from qatester.agents.goexplore import GoExplore

    env = _Corridor(limit=40, dead_after=30)  # saves made after tic step 30 are "late"
    ge = GoExplore(horizon=40, root=tmp_path, p_reset=0.9, patience=0.3)
    _walk(ge, env, 20_000)  # the corridor asserts that no finished episode is saved
    assert env.late and ge.dead_starts > 0
    live = {e.save_path for a in ge.archives.values() for e in a.entries.values()}
    loaded_late = env.late - live
    assert loaded_late, "no late save was ever dropped"
    assert all(not p.exists() for p in loaded_late)  # their files are gone too


def test_confined_cells_are_chosen_for_patience(tmp_path: Path) -> None:
    a = Archive(tmp_path)
    calls: list[Path] = []
    trap, open_, young = cell_of(0, 0, 0, []), cell_of(640, 0, 0, []), cell_of(1280, 0, 0, [])
    for c in (trap, open_, young):
        a.visit(c, 1, 1, _saver(calls))
    assert a.sample_confined(np.random.default_rng(0)) is None  # nothing restarted yet
    a.entries[trap].resets_from, a.entries[trap].max_reach = 3, 0  # restarts never left it
    a.entries[open_].resets_from, a.entries[open_].max_reach = 3, 9
    a.entries[young].resets_from = 1  # too few restarts to judge
    rng = np.random.default_rng(0)
    picks = [a.sample_confined(rng, min_resets=2).cell for _ in range(2_000)]  # type: ignore[union-attr]
    assert picks.count(young) == 0
    assert picks.count(trap) > 30 * picks.count(open_)


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
