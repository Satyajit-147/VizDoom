"""F1 (docs/FIXER_PLAN.md): the edit menu contains the right fix.

On 50 generated maps with random planted bugs, a report is placed at each bug's recorded
position and the menu is built from the buggy level alone. The bug's reference patch
(test_fixer_edits, every edit of it) must be in the menu for >= 95% of bugs (the plan's
bar); with the report moved up to 128 mu (the radius QA reports are scored with), for
>= 90% (bar set before the first run). Menus stay within the option picker's limits.

Teleport destinations: the true fix puts a destination at its room's centre, which the
level alone cannot pin to the unit: the room's sector stops 8 mu short where a wall strip
meets a neighbour and reaches 8 mu further where it owns an open doorway. So a MoveThing
counts when it is for the same thing, facing the same way, within 16 mu of the true fix
(the first run demanded an exact match and missed all 14 loops by these few units), and
`test_the_menus_own_teleport_moves_repair_the_loop_in_the_engine` checks that the menu's
own moves really end the loop in the engine.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from doomdriver import Driver
from test_defects_engine import SCENARIOS
from test_fixer_edits import reference_patch

from qatester.fixer.edits import Edit, MoveThing, SetFlag, SetFloor, apply, write_level
from qatester.fixer.menu import MAX_EDITS, MAX_ENTITIES, Report, build_menu
from qatester.mapgen.cli import build, render_cfg
from qatester.mapgen.geometry import CompiledMap
from qatester.mapgen.procgen import ProcgenConfig, generate, seeds, spec_yaml
from qatester.mapgen.spec import Defect

KINDS = ["missing_blocking", "wrong_tag", "wrong_special", "geometry_trap", "one_way_lift",
         "teleport_loop", "bad_pickup"]  # fmt: skip
N_MAPS = 50


@pytest.fixture(scope="module")
def bugs(tmp_path_factory: pytest.TempPathFactory) -> list[tuple[CompiledMap, CompiledMap, Defect]]:
    out = tmp_path_factory.mktemp("f1")
    rng = np.random.default_rng(7)
    found: list[tuple[CompiledMap, CompiledMap, Defect]] = []
    maps = 0
    for seed in seeds("train", 400):
        if maps == N_MAPS:
            break
        kinds = list(rng.choice(KINDS, size=int(rng.integers(1, 4)), replace=False))
        try:
            spec = generate(seed, ProcgenConfig(), slot=1, kinds=kinds)
        except Exception:  # noqa: BLE001, S112 - this seed cannot host these kinds
            continue
        d = out / f"s{seed}"
        d.mkdir()
        (d / "gen.yaml").write_text(spec_yaml(spec), encoding="utf-8")
        buggy, fixed = build(d, d / "b", "buggy"), build(d, d / "f", "fixed")
        cm = buggy.compiled["MAP01"]
        found += [(cm, fixed.compiled["MAP01"], defect) for defect in cm.spec.defects]
        maps += 1
    assert maps == N_MAPS
    return found


MOVE_TOLERANCE_MU = 16.0  # a wall strip's width: see the module docstring


def _record(cm: CompiledMap, defect: Defect) -> tuple[float, float, float]:
    return next(r.position for r in cm.defects if r.id == defect.id)


def _match(ref: Edit, offered: list[Edit]) -> Edit | None:
    """The menu edit that equals `ref` (MoveThing: same thing and angle, within 16 mu)."""
    if isinstance(ref, MoveThing):
        moves = [m for m in offered if isinstance(m, MoveThing) and m.thing == ref.thing
                 and m.angle == ref.angle
                 and math.hypot(m.x - ref.x, m.y - ref.y) <= MOVE_TOLERANCE_MU]  # fmt: skip
        return min(moves, key=lambda m: math.hypot(m.x - ref.x, m.y - ref.y), default=None)
    return ref if ref in offered else None


def _coverage(bugs: list[tuple[CompiledMap, CompiledMap, Defect]],
              jitter: float) -> tuple[float, Counter[str]]:  # fmt: skip
    rng = np.random.default_rng(0)
    missed: Counter[str] = Counter()
    for cm, fixed, defect in bugs:
        x, y, z = _record(cm, defect)
        if jitter:
            a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0, jitter)
            x, y = x + r * math.cos(a), y + r * math.sin(a)
        menu = build_menu(cm.textmap(), Report(defect.expected_oracle, "", (x, y, z)))
        offered = menu.all_edits()
        if any(_match(e, offered) is None for e in reference_patch(cm, fixed, defect)):
            missed[defect.kind] += 1
    return 1 - sum(missed.values()) / len(bugs), missed


def test_f1_the_fix_is_in_the_menu(bugs: list[tuple[CompiledMap, CompiledMap, Defect]]) -> None:
    assert {d.kind for _, _, d in bugs} == set(KINDS), "not every kind was planted"
    cov, missed = _coverage(bugs, jitter=0.0)
    print(f"\nF1 exact position: {cov:.1%} of {len(bugs)} bugs; missed by kind {dict(missed)}")
    assert cov >= 0.95, f"menu coverage {cov:.1%} < 95%; missed {dict(missed)}"


def test_f1_the_fix_is_in_the_menu_for_an_inexact_report(
    bugs: list[tuple[CompiledMap, CompiledMap, Defect]],
) -> None:
    cov, missed = _coverage(bugs, jitter=128.0)
    print(f"\nF1 report within 128 mu: {cov:.1%}; missed by kind {dict(missed)}")
    assert cov >= 0.90, f"menu coverage {cov:.1%} < 90%; missed {dict(missed)}"


def test_menus_fit_the_option_picker(bugs: list[tuple[CompiledMap, CompiledMap, Defect]]) -> None:
    for cm, _, defect in bugs[:40]:
        menu = build_menu(cm.textmap(), Report(defect.expected_oracle, "", _record(cm, defect)))
        assert 0 < len(menu.entities) <= MAX_ENTITIES
        assert all(0 < len(e.edits) <= MAX_EDITS for e in menu.entities)
        assert [e.distance for e in menu.entities] == sorted(e.distance for e in menu.entities)


def test_the_menus_own_teleport_moves_repair_the_loop_in_the_engine(
    bugs: list[tuple[CompiledMap, CompiledMap, Defect]], tmp_path: Path,
) -> None:
    """The 16 mu tolerance is only fair if the menu's moves work: apply the menu's own
    MoveThing edits (not the reference positions) and replay the loop scenario."""
    loops = [b for b in bugs if b[2].kind == "teleport_loop"][:4]
    assert loops
    for n, (cm, fixed, defect) in enumerate(loops):
        menu = build_menu(cm.textmap(), Report(defect.expected_oracle, "", _record(cm, defect)))
        patch = [_match(e, menu.all_edits()) for e in reference_patch(cm, fixed, defect)]
        assert all(p is not None for p in patch)
        cfg_text = render_cfg(cm.spec, "buggy")
        for name, tm, want in (("bug", cm.textmap(), True),
                               ("fix", apply(cm.textmap(), patch), False)):  # type: ignore[arg-type]
            d = Driver(write_level(tm, "MAP01", cfg_text, tmp_path, f"{name}{n}"))
            try:
                assert SCENARIOS["teleport_loop"](d, cm, defect) is want, f"{name}{n}"
            finally:
                d.close()


def test_the_menu_never_offers_to_wall_off_an_opening_or_a_trigger(
    bugs: list[tuple[CompiledMap, CompiledMap, Defect]],
) -> None:
    """Regression (first loop batch, level 1302): the Fixer made a teleport loop 'go
    away' by blocking the teleport pad's four edges, which also cut off a room hiding a
    trap. 'Make blocking' is offered only on lines drawn as a wall and with no special."""
    for cm, _, defect in bugs:
        tm = cm.textmap()
        menu = build_menu(tm, Report(defect.expected_oracle, "", _record(cm, defect)))
        for e in menu.all_edits():
            if isinstance(e, SetFlag) and e.flag == "blocking" and e.on:
                ld = tm.linedefs[e.linedef]
                assert ld.special == 0, f"would block trigger line {e.linedef}"
                assert ld.sideback is not None and any(
                    tm.sidedefs[s].texturemiddle for s in (ld.sidefront, ld.sideback)
                ), f"would wall off open line {e.linedef}"


def test_the_menu_never_flattens_a_lift_or_the_rooms_it_joins(
    bugs: list[tuple[CompiledMap, CompiledMap, Defect]],
) -> None:
    """Regression (first loop batch, level 403): a one-way lift was 'repaired' by raising
    its lower room to the top of the lift. A lift sector and its neighbours get no floor
    edits; the repair is the missing trigger (CopySpecial), still in the menu (F1)."""
    checked = 0
    for cm, _, defect in bugs:
        if defect.kind != "one_way_lift":
            continue
        p = cm.spec.portal(defect.target)
        lift = list(cm.sectors).index(f"lift:{p.id}")
        rooms = {list(cm.sectors).index(cm.room_sector(r)) for r in (p.a, p.b)}
        menu = build_menu(cm.textmap(), Report(defect.expected_oracle, "", _record(cm, defect)))
        floors = {e.sector for e in menu.all_edits() if isinstance(e, SetFloor)}
        assert not floors & ({lift} | rooms), f"{defect.id}: would flatten {floors & rooms}"
        checked += 1
    assert checked


def test_the_menu_depends_only_on_the_level_and_the_report(
    bugs: list[tuple[CompiledMap, CompiledMap, Defect]],
) -> None:
    cm, _, defect = bugs[0]
    r = Report(defect.expected_oracle, "", _record(cm, defect))
    assert build_menu(cm.textmap(), r) == build_menu(cm.textmap(), r)
