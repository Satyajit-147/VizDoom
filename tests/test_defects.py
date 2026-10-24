"""M3 gate: each defect changes exactly the intended element and appears in the ground
truth; the fixed variant of every map is fully reachable by BFS (PLAN: build order M3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qatester.mapgen import udmf
from qatester.mapgen.cli import BuildResult, build
from qatester.mapgen.defects import REGISTRY
from qatester.mapgen.geometry import PLAT_DOWN_WAIT_UP_STAY, CompiledMap, compile_map
from qatester.mapgen.groundtruth import GroundTruth
from qatester.mapgen.spec import Defect, MapSpec, load_spec

REPO = Path(__file__).resolve().parents[1]
SPECS = {p.stem: load_spec(p) for p in sorted((REPO / "maps").glob("*.yaml"))}
ALL_DEFECTS = [(name, d) for name, spec in SPECS.items() for d in spec.defects]
FAMILIES = {"crash", "stuck", "out_of_bounds", "unreachable", "invariant", "softlock"}

LineKey = tuple[tuple[float, float], tuple[float, float]]


def _lines(tm: udmf.TextMap) -> dict[LineKey, tuple[Any, ...]]:
    out = {}
    for ld in tm.linedefs:
        a, b = tm.vertices[ld.v1], tm.vertices[ld.v2]
        front = tm.sidedefs[ld.sidefront]
        back = tm.sidedefs[ld.sideback] if ld.sideback is not None else None
        out[((a.x, a.y), (b.x, b.y))] = (
            front.sector, None if back is None else back.sector, ld.flags, ld.special,
            ld.args, front.texturemiddle,
        )  # fmt: skip
    return out


def _isolated(spec: MapSpec, d: Defect) -> tuple[CompiledMap, CompiledMap]:
    intended = compile_map(spec.without_defects())
    built = compile_map(spec.model_copy(update={"defects": [d]}))
    return intended, built


def _diff(intended: CompiledMap, built: CompiledMap) -> dict[str, Any]:
    ti, tb = intended.textmap(), built.textmap()
    li, lb = _lines(ti), _lines(tb)
    return {
        "sectors": [i for i, (a, b) in enumerate(zip(ti.sectors, tb.sectors)) if a != b],
        "extra_sectors": tb.sectors[len(ti.sectors):],
        "things": [i for i, (a, b) in enumerate(zip(ti.things, tb.things)) if a != b],
        "removed": {k: v for k, v in li.items() if lb.get(k) != v},
        "added": {k: v for k, v in lb.items() if li.get(k) != v},
        "ti": ti, "tb": tb,
    }  # fmt: skip


def test_registry_covers_every_defect_kind() -> None:
    kinds = set(Defect.model_fields["kind"].annotation.__args__)  # type: ignore[union-attr]
    assert set(REGISTRY) == kinds and len(kinds) == 7


def test_corpus_matches_the_plan() -> None:
    counts = {name: len(spec.defects) for name, spec in SPECS.items()}
    assert counts["m01_clean"] == 0, "CLEAN must never carry a defect"
    assert sum(n for name, n in counts.items() if name not in ("m01_clean", "m08_holdout")) == 12
    used = {d.kind for _, spec in SPECS.items() for d in spec.defects}
    assert used == set(REGISTRY), "every defect kind appears in the corpus"


@pytest.mark.parametrize(("name", "defect"), ALL_DEFECTS, ids=[d.id for _, d in ALL_DEFECTS])
def test_defect_changes_only_its_element(name: str, defect: Defect) -> None:
    spec = SPECS[name]
    intended, built = _isolated(spec, defect)
    diff = _diff(intended, built)
    rec = built.defects[0]
    keys = list(intended.sectors)

    if defect.kind == "geometry_trap":
        sector = keys.index(intended.room_sector(defect.target))
        assert diff["sectors"] == [sector]
        drop = int(str(defect.params["drop"]))
        assert diff["tb"].sectors[sector].heightfloor == diff["ti"].sectors[sector].heightfloor - drop
        assert not diff["removed"] and not diff["added"] and not diff["things"]

    elif defect.kind == "wrong_tag":
        assert not diff["sectors"] and not diff["things"] and not diff["extra_sectors"]
        assert len(diff["removed"]) == len(diff["added"]) == 1
        (k, old), (k2, new) = next(iter(diff["removed"].items())), next(iter(diff["added"].items()))
        assert k == k2 and old[:4] == new[:4] and old[4][1:] == new[4][1:]
        assert new[4][0] not in {s.id for s in diff["tb"].sectors}, "tag must match no sector"

    elif defect.kind == "wrong_special":
        assert not diff["sectors"] and not diff["things"]
        door = keys.index(f"door:{defect.target}")
        assert len(diff["removed"]) >= 2 and diff["removed"].keys() == diff["added"].keys()
        for k, new in diff["added"].items():
            old = diff["removed"][k]
            assert door in (old[0], old[1]), "only the door's own faces change"
            assert old[4][3] != 0 and new[4] == (*old[4][:3], 0)

    elif defect.kind == "one_way_lift":
        assert not diff["sectors"] and not diff["things"]
        lift = keys.index(f"lift:{defect.target}")
        lower = keys.index(intended.room_sector(rec.affected_rooms[0]))
        assert diff["removed"] and diff["removed"].keys() == diff["added"].keys()
        for k, new in diff["added"].items():
            old = diff["removed"][k]
            assert {old[0], old[1]} == {lift, lower}
            assert old[3] == PLAT_DOWN_WAIT_UP_STAY and new[3] == 0
        upper_faces = [v for v in _lines(diff["tb"]).values()
                       if lift in (v[0], v[1]) and v[3] == PLAT_DOWN_WAIT_UP_STAY]  # fmt: skip
        assert upper_faces, "the upper side keeps its trigger"

    elif defect.kind == "teleport_loop":
        assert not diff["sectors"] and not diff["removed"] and not diff["added"]
        assert len(diff["things"]) == 2
        dest = udmf.THING_EDNUMS["teleport_dest"]
        assert all(diff["tb"].things[i].type == dest for i in diff["things"])

    elif defect.kind == "bad_pickup":
        assert not diff["sectors"] and not diff["removed"] and not diff["added"]
        assert len(diff["things"]) == 1
        i = diff["things"][0]
        assert (diff["ti"].things[i].type, diff["tb"].things[i].type) == (2012, 2013)

    elif defect.kind == "missing_blocking":
        assert not diff["sectors"] and not diff["things"] and len(diff["extra_sectors"]) == 1
        (pocket,) = rec.region
        lo, hi = (int(v) for v in defect.params["segment"])  # type: ignore[attr-defined]

        def span(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float, float]:
            """(fixed coordinate, low, high) of an axis-aligned segment."""
            if a[1] == b[1]:
                return a[1], min(a[0], b[0]), max(a[0], b[0])
            return a[0], min(a[1], b[1]), max(a[1], b[1])

        # Only the wall containing the segment is removed: one-sided, covering [lo, hi].
        assert len(diff["removed"]) == 1
        ((ra, rb), rv), = diff["removed"].items()
        wall_at, wall_lo, wall_hi = span(ra, rb)
        assert rv[1] is None and wall_lo <= lo and hi <= wall_hi
        pocket_index = len(diff["ti"].sectors)
        for (a, b), v in diff["added"].items():
            if pocket_index in (v[0], v[1]):  # new geometry borders the pocket
                for x, y in (a, b):
                    assert pocket.x0 <= x <= pocket.x1 and pocket.y0 <= y <= pocket.y1
            else:  # a leftover piece of the split wall, otherwise unchanged
                at, plo, phi = span(a, b)
                assert v == rv and at == wall_at and wall_lo <= plo and phi <= wall_hi
                assert phi <= lo or plo >= hi
        mouth = [v for v in diff["added"].values() if v[1] is not None]
        assert mouth and all("blocking" not in v[2] for v in mouth)
        assert all(v[5] is not None for v in mouth), "the gap is drawn as a wall"
    else:  # pragma: no cover
        pytest.fail(f"no expectation for {defect.kind}")


@pytest.mark.parametrize("name", SPECS)
def test_fixed_variant_fully_reachable(name: str) -> None:
    spec = SPECS[name].without_defects()
    assert spec.check_reachable() == {r.id for r in spec.rooms}


# --- ground truth JSON ---------------------------------------------------------------

@pytest.fixture(scope="module")
def builds(tmp_path_factory: pytest.TempPathFactory) -> dict[str, BuildResult]:
    out = tmp_path_factory.mktemp("build")
    return {v: build(REPO / "maps", out, v) for v in ("buggy", "fixed")}  # type: ignore[arg-type]


def test_groundtruth_describes_intent_not_bugs(builds: dict[str, BuildResult]) -> None:
    """Buggy and fixed ground truth differ only in `variant` and `defects`."""
    for lump, path in builds["buggy"].groundtruth_paths.items():
        buggy = json.loads(path.read_text())
        fixed = json.loads(builds["fixed"].groundtruth_paths[lump].read_text())
        assert fixed["defects"] == []
        for key in set(buggy) | set(fixed):
            if key not in ("variant", "defects"):
                assert buggy[key] == fixed[key], f"{lump}: {key} differs between variants"


def test_groundtruth_lists_every_defect(builds: dict[str, BuildResult]) -> None:
    for lump, path in builds["buggy"].groundtruth_paths.items():
        gt = GroundTruth.load(path)
        spec = builds["buggy"].compiled[lump].spec
        assert [d["id"] for d in gt.defects] == [d.id for d in spec.defects]
        rooms = {r["id"] for r in gt.data["rooms"]}
        for d in gt.defects:
            assert d["expected_oracle"] in FAMILIES
            assert len(d["position"]) == 3 and d["change"]
            assert d["affected_rooms"] and set(d["affected_rooms"]) <= rooms
            if d["kind"] != "bad_pickup":
                assert d["region"], f"{d['id']} has no region"


def test_groundtruth_lookups(builds: dict[str, BuildResult]) -> None:
    for path in builds["fixed"].groundtruth_paths.values():
        gt = GroundTruth.load(path)
        data = gt.data
        assert data["schema_version"] == 1
        sx, sy = data["spawn"]["pos"]
        spawn_room = next(r for r in data["rooms"] if r["id"] == data["spawn"]["room"])
        assert gt.sector_at(sx, sy) == spawn_room["sector"]
        assert gt.room_at(sx, sy) == spawn_room["id"]
        assert gt.walkable(*gt.cell_of(sx, sy))
        assert gt.sector_at(-10_000, -10_000) == -1
        w = data["walkable"]
        assert len(w["rows"]) == w["height"] and all(len(r) == w["width"] for r in w["rows"])
        walkable = sum(r.count("1") for r in w["rows"])
        room_cells = sum((r["rect"][2] - r["rect"][0]) * (r["rect"][3] - r["rect"][1])
                         for r in data["rooms"]) / w["cell_size"] ** 2  # fmt: skip
        assert 0.5 * room_cells < walkable <= 1.15 * room_cells  # centre-sampled cells
        assert data["walls"] and data["portal_graph"]
