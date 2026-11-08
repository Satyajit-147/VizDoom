"""The detect -> fix -> re-test loop (qatester/fixer/loop.py), with a fake QA.

The fake QA reports a bug while the level still has its defect (checked on the TextMap
the loop wrote), so these tests pin the loop's logic: an attempt is kept only when QA no
longer reports the target and nothing got worse; a failed attempt is undone; a level
with no reports is left alone. The real QA runs in scripts/fix_loop.py.
"""

from __future__ import annotations

from pathlib import Path

from qatester.fixer.edits import SetFlag, SetFloor
from qatester.fixer.loop import (
    QAResult,
    candidate_fixes,
    fix_level,
    fix_until_quiet,
    reward,
    same_bug,
)
from qatester.fixer.menu import Report
from qatester.mapgen import udmf
from qatester.mapgen.wad import read_wad

CFG = "doom_scenario_path = x.wad\ndoom_map = map01\n"
PIT = Report("stuck", "vertical_trap", (32.0, 32.0, -64.0))


def _level() -> udmf.TextMap:
    """Sector 0 (floor 0) with a pit, sector 1 (floor -64), joined by one open line."""
    tm = udmf.TextMap()
    tm.vertices = [udmf.Vertex(0, 0), udmf.Vertex(64, 0), udmf.Vertex(64, 64), udmf.Vertex(0, 64),
                   udmf.Vertex(128, 0), udmf.Vertex(128, 64)]  # fmt: skip
    tm.sectors = [udmf.Sector(0, 128), udmf.Sector(-64, 128)]
    tm.sidedefs = [udmf.Sidedef(0), udmf.Sidedef(1), udmf.Sidedef(0), udmf.Sidedef(1),
                   udmf.Sidedef(0), udmf.Sidedef(1), udmf.Sidedef(1)]  # fmt: skip
    tm.linedefs = [udmf.Linedef(1, 2, 0, 1, flags=frozenset({"twosided"})),
                   udmf.Linedef(0, 1, 2, None, flags=frozenset({"blocking"})),
                   udmf.Linedef(2, 3, 3, None, flags=frozenset({"blocking"})),
                   udmf.Linedef(3, 0, 4, None, flags=frozenset({"blocking"})),
                   udmf.Linedef(1, 4, 5, None, flags=frozenset({"blocking"})),
                   udmf.Linedef(4, 5, 6, None, flags=frozenset({"blocking"}))]  # fmt: skip
    return tm


def _fake_qa(calls: list[Path]):  # type: ignore[no-untyped-def]
    """Reports the pit while sector 1's floor is below 0; reads the level the loop wrote."""

    def qa(cfg: Path, seed: int) -> QAResult:
        calls.append(cfg)
        wad = cfg.with_suffix(".wad")
        text = next(lump.data for lump in read_wad(wad) if lump.name == "TEXTMAP").decode()
        pit = "heightfloor = -64" in text
        return QAResult((PIT,) if pit else (), 0.9)

    return qa


def test_reward_follows_the_plan() -> None:
    before = QAResult((PIT,), 0.9)
    assert reward(PIT, before, QAResult((), 0.9), 1)[0] == 1.0 - 0.05
    assert reward(PIT, before, QAResult((PIT,), 0.9), 1)[1] is False  # still there
    r, gone, damaged, _ = reward(PIT, before, QAResult((), 0.5), 1)  # walled off
    assert gone and damaged and r < 0
    other = Report("out_of_bounds", "wall_traversal", (900.0, 900.0, 0.0))
    assert reward(PIT, before, QAResult((other,), 0.9), 1)[3] == 1  # a new bug


def test_same_bug_needs_the_family_and_128_mu() -> None:
    assert same_bug(PIT, Report("stuck", "wedge", (100.0, 100.0, 0.0)))
    assert not same_bug(PIT, Report("softlock", "confined", (32.0, 32.0, 0.0)))
    assert not same_bug(PIT, Report("stuck", "wedge", (300.0, 32.0, 0.0)))


def test_the_loop_keeps_the_fix_qa_confirms(tmp_path: Path) -> None:
    calls: list[Path] = []
    res = fix_level(_level(), "MAP01", CFG, _fake_qa(calls), tmp_path, log=lambda s: None)
    assert res.initial.reports == (PIT,) and res.final.reports == ()
    assert len(res.fixes) == 1 and res.fixes[0].edits == (SetFloor(1, 0),)
    assert res.final_level.sectors[1].heightfloor == 0
    assert all(not a.accepted for a in res.attempts[:-1])  # anything tried before was undone
    assert len(calls) == 1 + len(res.attempts)  # one QA run per attempt, plus the first


def test_a_level_without_reports_is_left_alone(tmp_path: Path) -> None:
    calls: list[Path] = []
    tm = _level()
    tm.sectors[1] = udmf.Sector(0, 128)
    res = fix_level(tm, "MAP01", CFG, _fake_qa(calls), tmp_path, log=lambda s: None)
    assert res.attempts == [] and len(calls) == 1 and res.final_level.to_text() == tm.to_text()


def test_rounds_continue_down_the_candidate_list_on_the_same_level(tmp_path: Path) -> None:
    """One candidate per round: round 1 tries the first (it fails and is undone), round 2
    goes on with the second instead of repeating the first; then two quiet rounds stop it."""
    calls: list[Path] = []
    res, rounds = fix_until_quiet(_level(), "MAP01", CFG, lambda r: _fake_qa(calls), tmp_path,
                                  max_attempts=1, quiet=2, max_rounds=10, log=lambda s: None)  # fmt: skip
    assert [a.edits for a in res.attempts] == [(SetFloor(0, -64),), (SetFloor(1, 0),)]
    assert [len(x.attempts) for x in rounds] == [1, 1, 0, 0]
    assert res.initial.reports == (PIT,) and res.final.reports == ()
    assert res.final_level.sectors[1].heightfloor == 0


def test_a_bug_qa_misses_at_first_is_fixed_in_a_later_round(tmp_path: Path) -> None:
    """QA spots the pit only from round 2 on (as a longer run with a new seed may); the
    loop keeps going on the same level and repairs it there."""
    calls: list[Path] = []
    seeds: list[int] = []

    def qa_for_round(r: int):  # type: ignore[no-untyped-def]
        inner = _fake_qa(calls)

        def qa(cfg: Path, seed: int) -> QAResult:
            seeds.append(seed)
            found = inner(cfg, seed)
            return found if r >= 1 else QAResult((), found.coverage)

        return qa

    res, rounds = fix_until_quiet(_level(), "MAP01", CFG, qa_for_round, tmp_path, seed=5,
                                  quiet=2, max_rounds=10, log=lambda s: None)  # fmt: skip
    assert rounds[0].initial.reports == () and rounds[1].initial.reports == (PIT,)
    assert res.final_level.sectors[1].heightfloor == 0 and res.final.reports == ()
    assert len(rounds) == 4  # quiet, fixed, quiet, quiet
    assert seeds[0] == 5 and len(set(seeds)) == 4  # a new QA seed every round


def test_rounds_stop_at_the_cap(tmp_path: Path) -> None:
    """A report the Fixer keeps trying never makes QA quiet; max_rounds ends it."""

    def stubborn(cfg: Path, seed: int) -> QAResult:
        return QAResult((PIT,), 0.9)

    res, rounds = fix_until_quiet(_level(), "MAP01", CFG, lambda r: stubborn, tmp_path,
                                  max_attempts=1, quiet=2, max_rounds=2, log=lambda s: None)  # fmt: skip
    assert len(rounds) == 2 and not res.fixes and res.final_level.to_text() == _level().to_text()


def test_candidates_never_wall_off_an_open_line() -> None:
    """Line 0 is an open passage (no middle texture): the menu may raise the pit's floor
    but must not offer to block the passage, which would hide the pit, not repair it."""
    fixes = candidate_fixes(_level(), PIT)
    assert (SetFloor(1, 0),) in fixes
    assert not [f for f in fixes if isinstance(f[0], SetFlag) and f[0].on]


def test_candidates_group_edits_that_belong_together() -> None:
    """Every walk-through wall line between the same two sectors is one attempt."""
    tm = _level()
    tm.sidedefs = [udmf.Sidedef(0, texturemiddle="STARTAN2"), udmf.Sidedef(1, texturemiddle="STARTAN2"),
                   *tm.sidedefs[2:]]  # fmt: skip  (line 0 is now drawn as a wall)
    blocking = [f for f in candidate_fixes(tm, PIT) if isinstance(f[0], SetFlag)]
    assert blocking == [(SetFlag(0, "blocking", True),)]
