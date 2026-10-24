"""M2 gate: the seven correctness invariants (PLAN: Critical correctness invariants)
hold for every shipped map, and each check catches a deliberately broken map."""

import dataclasses
from pathlib import Path

import pytest

from qatester.mapgen import udmf, validate
from qatester.mapgen.geometry import CompiledMap, compile_map
from qatester.mapgen.spec import MapSpec, load_spec

REPO = Path(__file__).resolve().parents[1]
SPECS = sorted((REPO / "maps").glob("*.yaml"))


@pytest.fixture(scope="module", params=SPECS, ids=lambda p: p.stem)
def compiled(request: pytest.FixtureRequest) -> tuple[CompiledMap, udmf.TextMap]:
    cm = compile_map(load_spec(request.param))
    return cm, cm.textmap()


def test_1_front_sidedefs_valid(compiled: tuple[CompiledMap, udmf.TextMap]) -> None:
    assert validate.check_sidedefs(compiled[1]) == []


def test_2_two_sided_flags(compiled: tuple[CompiledMap, udmf.TextMap]) -> None:
    assert validate.check_two_sided_flags(compiled[1]) == []


def test_3_vertices_and_lengths(compiled: tuple[CompiledMap, udmf.TextMap]) -> None:
    assert validate.check_vertices(compiled[1]) == []


def test_4_winding(compiled: tuple[CompiledMap, udmf.TextMap]) -> None:
    assert validate.check_winding(*compiled) == []


def test_5_sectors_closed(compiled: tuple[CompiledMap, udmf.TextMap]) -> None:
    assert validate.check_closed(compiled[1]) == []


def test_6_headroom(compiled: tuple[CompiledMap, udmf.TextMap]) -> None:
    assert validate.check_heights(compiled[0]) == []


def test_7_things_clear_of_walls(compiled: tuple[CompiledMap, udmf.TextMap]) -> None:
    assert validate.check_things(*compiled) == []


def test_every_portal_and_switch_is_emitted(
    compiled: tuple[CompiledMap, udmf.TextMap],
) -> None:
    cm, _ = compiled
    sources = {line.source for line in cm.lines if line.special is not None}
    for p in cm.spec.portals:
        is_remote = any(t.opens == p.id for t in cm.spec.things)
        if p.kind != "open" and not is_remote:
            assert p.id in sources, f"portal {p.id} has no trigger linedef"
    for t in cm.spec.things:
        if t.type in ("exit_switch", "switch"):
            assert t.id in sources, f"{t.id} has no switch linedef"


# --- each check catches a broken map -------------------------------------------------

@pytest.fixture
def clean() -> tuple[CompiledMap, udmf.TextMap]:
    cm = compile_map(load_spec(REPO / "maps" / "m01_clean.yaml"))
    return cm, cm.textmap()


def test_catches_bad_sidedef_reference(clean: tuple[CompiledMap, udmf.TextMap]) -> None:
    _, tm = clean
    tm.linedefs[0] = dataclasses.replace(tm.linedefs[0], sidefront=10_000)
    assert validate.check_sidedefs(tm)


def test_catches_blocking_two_sided_line(clean: tuple[CompiledMap, udmf.TextMap]) -> None:
    _, tm = clean
    i = next(i for i, ld in enumerate(tm.linedefs) if ld.sideback is not None)
    tm.linedefs[i] = dataclasses.replace(tm.linedefs[i], flags=tm.linedefs[i].flags | {"blocking"})
    assert validate.check_two_sided_flags(tm)
    assert validate.check_two_sided_flags(tm, allow_blocking={i}) == []


def test_catches_near_duplicate_vertex(clean: tuple[CompiledMap, udmf.TextMap]) -> None:
    _, tm = clean
    v = tm.vertices[0]
    tm.vertices.append(udmf.Vertex(v.x + 0.5, v.y))
    assert validate.check_vertices(tm)


def test_catches_reversed_winding(clean: tuple[CompiledMap, udmf.TextMap]) -> None:
    cm, tm = clean
    i = next(i for i, ld in enumerate(tm.linedefs) if ld.sideback is None)
    ld = tm.linedefs[i]
    tm.linedefs[i] = dataclasses.replace(ld, v1=ld.v2, v2=ld.v1)
    assert validate.check_winding(cm, tm)


def test_catches_open_sector(clean: tuple[CompiledMap, udmf.TextMap]) -> None:
    _, tm = clean
    del tm.linedefs[0]
    assert validate.check_closed(tm)


def test_catches_low_ceiling() -> None:
    spec = load_spec(REPO / "maps" / "m01_clean.yaml")
    rooms = [r.model_copy(update={"ceiling": 48}) if r.id == "hall" else r for r in spec.rooms]
    cm = compile_map(spec.model_copy(update={"rooms": rooms}))
    assert validate.check_heights(cm)


def test_catches_thing_against_wall() -> None:
    spec = load_spec(REPO / "maps" / "m01_clean.yaml")
    things = [
        t.model_copy(update={"pos": (1190, 224)}) if t.id == "medkit" else t for t in spec.things
    ]
    cm = compile_map(MapSpec.model_validate(spec.model_copy(update={"things": things}).model_dump()))
    assert any("medkit" in p for p in validate.check_things(cm, cm.textmap()))
