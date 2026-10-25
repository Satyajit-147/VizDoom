"""The procedural map generator (docs/decisions.md D18).

Unit level: determinism, disjoint splits, every seed valid, every class generated.
Engine level, on a pack of generated maps from both splits:
- fixed variant: plays as intended (the M2 portal routines) and the oracles stay silent
  on that play and on random play: no false alarms on maps nobody has seen;
- buggy variant: every generated defect shows in the engine, and the oracles find it
  with its expected family and nothing else.
"""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import pytest
import test_defects_engine as defect_scenarios
import test_maps_engine as portal_routines
from recorder import RecordingDriver
from test_oracles import describe, matches
from test_oracles_engine import ROUTINES

from qatester.mapgen.cli import BuildResult, build
from qatester.mapgen.groundtruth import GroundTruth
from qatester.mapgen.procgen import (
    DEFECT_KINDS,
    EXPECTED_ORACLE,
    generate,
    seeds,
    spec_yaml,
    write_pack,
)
from qatester.mapgen.spec import load_spec
from qatester.oracles.registry import run_oracles

N_UNIT = 300
PACK = seeds("train", 12) + seeds("test", 12)


def test_same_seed_same_map() -> None:
    for s in (0, 7, 1_000_003):
        assert generate(s) == generate(s)
    assert generate(1) != generate(2)


def test_splits_are_disjoint() -> None:
    train, test = set(seeds("train", 5000)), set(seeds("test", 5000))
    assert not train & test
    with pytest.raises(ValueError):
        seeds("train", 10, start=1_000_000)


def test_every_seed_gives_a_valid_map_and_every_class_appears() -> None:
    kinds: collections.Counter[str] = collections.Counter()
    clean = 0
    for s in seeds("train", N_UNIT // 2) + seeds("test", N_UNIT // 2):
        spec = generate(s)  # validated and compiled, both variants, inside generate
        assert 5 <= len(spec.rooms) <= 9
        assert len(spec.defects) <= 3
        assert len({d.kind for d in spec.defects}) == len(spec.defects)
        for d in spec.defects:
            assert d.expected_oracle == EXPECTED_ORACLE[d.kind]
        kinds.update(d.kind for d in spec.defects)
        clean += not spec.defects
    assert set(kinds) == set(DEFECT_KINDS)
    # Uniform draw: each class is about 1.5/7 of maps. Allow sampling noise.
    assert min(kinds.values()) >= 0.6 * max(kinds.values())
    assert 0.1 * N_UNIT <= clean <= 0.4 * N_UNIT


def test_yaml_round_trip(tmp_path: Path) -> None:
    paths = write_pack(tmp_path, [3, 4])
    for slot, (p, s) in enumerate(zip(paths, [3, 4], strict=True), start=1):
        assert load_spec(p) == generate(s, slot=slot)
    assert spec_yaml(generate(3)).startswith("# Generated")
    with pytest.raises(ValueError):
        write_pack(tmp_path, range(33))


# --- engine ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gen_built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, BuildResult]:
    specs = tmp_path_factory.mktemp("gen_specs")
    write_pack(specs, PACK)
    out = tmp_path_factory.mktemp("gen_build")
    return {v: build(specs, out, v) for v in ("buggy", "fixed")}  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def gen_truth(gen_built: dict[str, BuildResult]) -> dict[tuple[str, str], GroundTruth]:
    return {(lump, v): GroundTruth.load(p) for v, r in gen_built.items()
            for lump, p in r.groundtruth_paths.items()}  # fmt: skip


LUMPS = [f"MAP{i:02d}" for i in range(1, len(PACK) + 1)]


def _random_play(d: RecordingDriver, tics: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    d.reset()
    for _ in range(tics):
        if d.game.is_episode_finished():
            break
        d._act([float(rng.random() < 0.7), float(rng.random() < 0.05),
                float(rng.uniform(-20, 20)), float(rng.random() < 0.05)], 1)  # fmt: skip


@pytest.mark.parametrize("lump", LUMPS)
def test_generated_maps_play_as_intended_and_stay_silent(
    gen_built: dict[str, BuildResult], gen_truth: dict[tuple[str, str], GroundTruth], lump: str,
) -> None:  # fmt: skip
    result = gen_built["fixed"]
    cm = result.compiled[lump]
    d = RecordingDriver(result.cfg_paths[lump])
    try:
        portal_routines.test_engine_geometry_matches_compiler((d, cm))
        for routine in ROUTINES:
            routine((d, cm))  # each asserts its portals work
        _random_play(d, 2000, seed=int(lump[3:]))
        runs = d.runs(gen_truth[lump, "fixed"])
    finally:
        d.close()
    findings = run_oracles(runs)
    assert findings == [], f"{cm.spec.name}: " + describe(findings)


def _defect_cases() -> list[tuple[str, str]]:
    return [(f"MAP{slot:02d}", d.id) for slot, s in enumerate(PACK, start=1)
            for d in generate(s, slot=slot).defects]  # fmt: skip


@pytest.mark.parametrize(("lump", "defect_id"), _defect_cases())
def test_generated_defects_are_found_by_their_family(
    gen_built: dict[str, BuildResult], gen_truth: dict[tuple[str, str], GroundTruth],
    lump: str, defect_id: str,
) -> None:  # fmt: skip
    result = gen_built["buggy"]
    cm = result.compiled[lump]
    defect = next(x for x in cm.spec.defects if x.id == defect_id)
    scenario = defect_scenarios.SCENARIOS[defect.kind]
    repeats = 6 if defect.kind == "wrong_tag" else 1
    d = RecordingDriver(result.cfg_paths[lump])
    try:
        for _ in range(repeats):
            assert scenario(d, cm, defect), f"{cm.spec.name} {defect_id}: did not show"
            d.close_episode(truncated=defect.kind == "one_way_lift")
        runs = d.runs(gen_truth[lump, "buggy"])
    finally:
        d.close()
    gt_defect = next(x for x in gen_truth[lump, "buggy"].defects if x["id"] == defect_id)
    findings = run_oracles(runs)
    assert any(matches(f, gt_defect) for f in findings), (
        f"{cm.spec.name} {defect_id}: " + (describe(findings) or "no findings"))
    # Another real defect of the same map showing up is not a false alarm.
    others = [x for x in gen_truth[lump, "buggy"].defects if x["id"] != defect_id]
    stray = [f for f in findings if f.oracle != defect.expected_oracle
             and not any(matches(f, o) for o in others)]  # fmt: skip
    assert not stray, f"{cm.spec.name} {defect_id}: " + describe(stray)
