"""F0 (docs/FIXER_PLAN.md): the Fixer's six edit operations can repair every bug kind.

Unit tests check each op and its guard rails. The engine test is the milestone's
acceptance test: for every planted bug on the hand-made maps, plus a generated map with
the one kind they lack (teleport_loop), a patch made ONLY of the six ops is applied to
the buggy level, the level is rewritten, and the bug's own engine scenario from
test_defects_engine (proven to show on the buggy build and not on the fixed one) must no
longer show; the exit must still work. The patches are written by hand from the answer
key, as the scenarios are: this proves the op set CAN express every fix, not that a
learned Fixer finds them (that is F1-F5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from doomdriver import Driver, approach
from test_defects_engine import SCENARIOS

from qatester.fixer.edits import (
    CopySpecial,
    Edit,
    EditError,
    MoveThing,
    SetArg,
    SetFlag,
    SetFloor,
    SetThingType,
    apply,
    describe,
    write_level,
)
from qatester.mapgen import udmf
from qatester.mapgen.cli import build
from qatester.mapgen.geometry import (
    DOOR_LOCKED_RAISE,
    EXIT_NORMAL,
    LOCK_NUMBERS,
    PLAT_DOWN_WAIT_UP_STAY,
    CompiledMap,
)
from qatester.mapgen.procgen import ProcgenConfig, generate, spec_yaml
from qatester.mapgen.spec import Defect

REPO = Path(__file__).resolve().parents[1]
MEDIKIT, SOULSPHERE = 2012, 2013


# --- unit tests: the ops and their guard rails --------------------------------------

def _tiny() -> udmf.TextMap:
    tm = udmf.TextMap()
    tm.vertices = [udmf.Vertex(0, 0), udmf.Vertex(64, 0), udmf.Vertex(64, 64)]
    tm.sectors = [udmf.Sector(0, 128), udmf.Sector(-64, 128)]
    tm.sidedefs = [udmf.Sidedef(0), udmf.Sidedef(1)]
    tm.linedefs = [udmf.Linedef(0, 1, 0, 1, flags=frozenset({"twosided"})),
                   udmf.Linedef(1, 2, 0, None, flags=frozenset({"blocking", "playeruse"}),
                                special=PLAT_DOWN_WAIT_UP_STAY, args=(5, 16, 105))]  # fmt: skip
    tm.things = [udmf.Thing(32, 32, SOULSPHERE), udmf.Thing(10, 10, 14, angle=90),
                 udmf.Thing(20, 20, 3001)]  # fmt: skip
    return tm


def test_apply_returns_a_new_level_and_leaves_the_input_alone() -> None:
    tm = _tiny()
    before = tm.to_text()
    out = apply(tm, [SetFlag(0, "blocking", True), SetFloor(1, 0), SetThingType(0, MEDIKIT),
                     MoveThing(1, 40, 40, 450), SetArg(1, 0, 9), CopySpecial(1, 0)])  # fmt: skip
    assert tm.to_text() == before
    assert "blocking" in out.linedefs[0].flags and out.sectors[1].heightfloor == 0
    assert out.things[0].type == MEDIKIT and (out.things[1].x, out.things[1].angle) == (40, 90)
    assert out.linedefs[0].special == PLAT_DOWN_WAIT_UP_STAY and out.linedefs[0].args[0] == 9
    assert "playeruse" in out.linedefs[0].flags  # activation flags travel with the special


@pytest.mark.parametrize("edit,why", [
    (SetFlag(0, "twosided", False), "structure"),
    (SetFlag(9, "blocking", True), "no linedef"),
    (SetArg(0, 0, 3), "no special"),
    (SetArg(1, 5, 3), "arguments 0-4"),
    (SetFloor(0, 100), "crush"),
    (SetThingType(0, 3001), "class"),  # a health item may not become a monster
    (SetThingType(2, 3002), "class"),  # a monster has no class: never retyped
    (MoveThing(0, 1, 1, 0), "teleport destination"),
    (CopySpecial(0, 1), "no special"),
])  # fmt: skip
def test_edits_that_do_not_apply_are_refused(edit: Edit, why: str) -> None:
    with pytest.raises(EditError):
        apply(_tiny(), [edit])


def test_describe_names_every_op() -> None:
    for e in (SetFlag(1, "blocking", True), SetArg(1, 0, 2), CopySpecial(1, 2),
              SetFloor(3, 0), SetThingType(0, MEDIKIT), MoveThing(1, 5, 5, 0)):  # fmt: skip
        assert str(e.__dict__[next(iter(e.__dict__))]) in describe(e)


def test_write_level_points_the_config_at_the_new_wad(tmp_path: Path) -> None:
    cfg = "doom_scenario_path = qa_scenarios_buggy.wad\ndoom_map = map03\n"
    path = write_level(_tiny(), "MAP03", cfg, tmp_path, "fix1")
    assert "doom_scenario_path = fix1.wad" in path.read_text() and "map03" in path.read_text()
    from qatester.mapgen.wad import read_wad

    names = [lump.name for lump in read_wad(tmp_path / "fix1.wad")]
    assert names == ["MAP03", "TEXTMAP", "ENDMAP"]


# --- the reference patches (hand-written from the answer key, like the scenarios) ----

def _sector(cm: CompiledMap, key: str) -> int:
    return list(cm.sectors).index(key)


def _lines_between(tm: udmf.TextMap, a: int, b: int) -> list[int]:
    return [i for i, ld in enumerate(tm.linedefs) if ld.sideback is not None and
            {tm.sidedefs[ld.sidefront].sector, tm.sidedefs[ld.sideback].sector} == {a, b}]  # fmt: skip


def reference_patch(cm: CompiledMap, fixed: CompiledMap, d: Defect) -> list[Edit]:
    tm, spec = cm.textmap(), cm.spec
    if d.kind == "missing_blocking":
        pocket, room = _sector(cm, f"pocket:{d.id}"), _sector(cm, cm.room_sector(d.target))
        return [SetFlag(i, "blocking", True) for i in _lines_between(tm, pocket, room)]
    if d.kind == "wrong_tag":
        switch = spec.thing(d.target)
        assert switch.opens is not None
        mark = next(m for m in cm.marks if m.source == switch.id).special
        tag = cm.sectors[f"door:{switch.opens}"].tag
        return [SetArg(i, 0, tag) for i, ld in enumerate(tm.linedefs)
                if (ld.special, ld.args) == (mark.special, mark.args)]  # fmt: skip
    if d.kind == "wrong_special":
        door = _sector(cm, f"door:{d.target}")
        lock = LOCK_NUMBERS[str(spec.portal(d.target).requires)]
        return [SetArg(i, 3, lock) for i, ld in enumerate(tm.linedefs)
                if ld.special == DOOR_LOCKED_RAISE and door in
                {tm.sidedefs[ld.sidefront].sector,
                 tm.sidedefs[ld.sideback].sector if ld.sideback is not None else -1}]  # fmt: skip
    if d.kind == "geometry_trap":
        room = spec.room(d.target)
        return [SetFloor(_sector(cm, cm.room_sector(room.id)), room.floor)]
    if d.kind == "one_way_lift":
        p = spec.portal(d.target)
        ra, rb = spec.room(p.a), spec.room(p.b)
        lower, upper = (ra, rb) if ra.floor < rb.floor else (rb, ra)
        lift = _sector(cm, f"lift:{p.id}")
        up = _lines_between(tm, lift, _sector(cm, cm.room_sector(upper.id)))
        src = next(i for i in up if tm.linedefs[i].special == PLAT_DOWN_WAIT_UP_STAY)
        low = _lines_between(tm, lift, _sector(cm, cm.room_sector(lower.id)))
        return [CopySpecial(src, i) for i in low if tm.linedefs[i].special == 0]
    if d.kind == "teleport_loop":
        ft = fixed.textmap().things
        return [MoveThing(i, ft[i].x, ft[i].y, ft[i].angle) for i, t in enumerate(tm.things)
                if t.type == udmf.THING_EDNUMS["teleport_dest"] and (t.x, t.y) != (ft[i].x, ft[i].y)]  # fmt: skip
    if d.kind == "bad_pickup":
        return [SetThingType(cm.thing_index[d.target], MEDIKIT)]
    raise AssertionError(d.kind)


# --- the engine acceptance test -------------------------------------------------------

@dataclass
class Case:
    lump: str
    defect: Defect
    buggy: CompiledMap
    fixed: CompiledMap
    cfg_text: str


@pytest.fixture(scope="module")
def cases(tmp_path_factory: pytest.TempPathFactory) -> list[Case]:
    out = tmp_path_factory.mktemp("f0")
    # the 8 hand-made maps: 15 planted bugs of 6 kinds
    sources = [REPO / "maps"]
    # a generated map for the 7th kind, which no hand-made map has
    last: Exception | None = None
    for seed in range(1, 60):
        try:
            spec = generate(seed, ProcgenConfig(), slot=1, kinds=["teleport_loop"])
            break
        except Exception as e:  # noqa: BLE001 - this seed cannot host it: try the next
            last = e
    else:
        raise RuntimeError(f"no seed in 1..59 hosts a teleport_loop: {last}")
    gen = out / "gen"
    gen.mkdir()
    (gen / "gen.yaml").write_text(spec_yaml(spec), encoding="utf-8")
    sources.append(gen)
    found = []
    for i, src in enumerate(sources):
        buggy, fixed = build(src, out / f"b{i}", "buggy"), build(src, out / f"f{i}", "fixed")
        for lump, cm in buggy.compiled.items():
            for d in cm.spec.defects:
                found.append(Case(lump, d, cm, fixed.compiled[lump],
                                  buggy.cfg_paths[lump].read_text()))  # fmt: skip
    return found


def test_every_kind_has_a_reference_patch_made_of_the_six_ops(cases: list[Case]) -> None:
    assert {c.defect.kind for c in cases} == set(SCENARIOS)  # all 7 kinds are covered
    for c in cases:
        patch = reference_patch(c.buggy, c.fixed, c.defect)
        assert patch, f"{c.lump} {c.defect.id}: empty patch"
        apply(c.buggy.textmap(), patch)  # every edit is valid on the buggy level


def test_f0_the_six_ops_repair_every_planted_bug_in_the_engine(cases: list[Case],
                                                               tmp_path: Path) -> None:  # fmt: skip
    """The milestone F0 acceptance test (docs/FIXER_PLAN.md section 8)."""
    still_buggy, broken_exit, not_shown = [], [], []
    for n, c in enumerate(cases):
        # Control: the UNPATCHED level, rewritten by the same write_level, must still show
        # the bug, so "gone after the patch" cannot come from the rewrite itself.
        bug = write_level(c.buggy.textmap(), c.lump, c.cfg_text, tmp_path, f"bug{n}")
        b = Driver(bug)
        try:
            if not SCENARIOS[c.defect.kind](b, c.buggy, c.defect):
                not_shown.append(f"{c.lump} {c.defect.id} ({c.defect.kind})")
        finally:
            b.close()
        patched = apply(c.buggy.textmap(), reference_patch(c.buggy, c.fixed, c.defect))
        cfg = write_level(patched, c.lump, c.cfg_text, tmp_path, f"fix{n}")
        d = Driver(cfg)
        try:
            if SCENARIOS[c.defect.kind](d, c.buggy, c.defect):
                still_buggy.append(f"{c.lump} {c.defect.id} ({c.defect.kind})")
            m = next(m for m in c.buggy.marks if m.special.special == EXIT_NORMAL)
            room = c.buggy.spec.room(c.buggy.spec.thing(m.source).room)
            x, y, angle = approach(room, m.axis, m.at, (m.lo + m.hi) / 2, 40)
            d.reset()
            d.warp(x, y)
            d.face(angle)
            d.use()
            d.idle(2)
            if not d.game.is_episode_finished():
                broken_exit.append(f"{c.lump} {c.defect.id}")
        finally:
            d.close()
    assert len(cases) == 16
    assert not not_shown, "control failed, bug absent before patching: " + "; ".join(not_shown)
    assert not still_buggy, "bug still shows after its patch: " + "; ".join(still_buggy)
    assert not broken_exit, "patch broke the exit: " + "; ".join(broken_exit)
