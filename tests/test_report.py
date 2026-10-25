"""M6 report pipeline without the engine: clustering, severity, calibration, emission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qatester.mapgen.cli import BuildResult
from qatester.mapgen.groundtruth import GroundTruth
from qatester.mapgen.spec import load_spec
from qatester.report.calibrate import apply_temperature, ece, fit_temperature
from qatester.report.cluster import cluster_findings
from qatester.report.emit import build_report
from qatester.report.severity import critical_rooms, room_of, severity

REPO = Path(__file__).resolve().parents[1]


def raw(oracle: str, kind: str, x: float, y: float, conf: float = 1.0, tic: int = 100,
        step: int = 0, demo: str | None = None) -> dict[str, Any]:  # fmt: skip
    return {"oracle": oracle, "kind": kind, "position": [x, y, 0.0], "tic_range": [tic - 10, tic],
            "confidence": conf, "evidence": {"episode": 0}, "demo_path": demo, "map": "MAP03",
            "env_steps": step, "env_seed": 0}  # fmt: skip


def test_clusters_by_family_place_and_invariant_name() -> None:
    fs = [raw("out_of_bounds", "outside_level", 100 + i, 100) for i in range(5)]
    fs += [raw("out_of_bounds", "outside_level", 900, 900)]  # a lone one: kept, not noise
    fs += [raw("stuck", "wedge", 100, 100) for _ in range(3)]  # same place, other family
    fs += [raw("invariant", "health_cap", 10 * i, 0) for i in range(4)]
    fs += [raw("invariant", "ammo_nonneg", 0, 0)]
    cs = cluster_findings(fs)
    shape = sorted((c.oracle, c.key, c.size) for c in cs)
    assert shape == [
        ("invariant", "ammo_nonneg", 1), ("invariant", "health_cap", 4),
        ("out_of_bounds", "spatial", 1), ("out_of_bounds", "spatial", 5),
        ("stuck", "spatial", 3),
    ]  # fmt: skip


def test_representative_is_most_confident_then_shortest() -> None:
    fs = [raw("stuck", "wedge", 0, 0, conf=0.5, tic=50), raw("stuck", "wedge", 5, 0, conf=0.9,
          tic=400), raw("stuck", "wedge", 9, 0, conf=0.9, tic=200)]  # fmt: skip
    (c,) = cluster_findings(fs)
    assert c.representative["tic_range"][1] == 200


def test_severity_uses_the_critical_path(truth: dict[tuple[str, str], GroundTruth]) -> None:
    gt = truth["MAP01", "fixed"]
    # MAP01: start -> hall -> vault (exit); the loft only holds the key.
    assert critical_rooms(gt) == {"start", "hall", "vault"}
    assert room_of(gt, 256, 700) == "loft" and room_of(gt, -500, 224) == "start"
    (c,) = cluster_findings([raw("crash", "crash", 0, 0) for _ in range(10)])
    assert severity(c, "hall", critical_rooms(gt)) == pytest.approx(0.5 + 0.3 + 0.2)
    assert severity(c, "loft", critical_rooms(gt)) == pytest.approx(0.5 + 0.09 + 0.2)


def test_temperature_calibration() -> None:
    rng = np.random.default_rng(0)
    # Overconfident oracle: says 0.95, right 70% of the time.
    conf = np.full(400, 0.95)
    labels = (rng.random(400) < 0.7).astype(int)
    t = fit_temperature(conf, labels)
    assert t > 1.0
    after = apply_temperature(conf, t)
    assert after[0] == pytest.approx(labels.mean(), abs=0.01)  # one temperature: the hit rate
    assert ece(after, labels) < ece(conf, labels)
    assert list(apply_temperature([0.3, 1.0], 1.0)) == [0.3, 1.0]


def test_report_json_and_html(built: dict[str, BuildResult],
                              truth: dict[tuple[str, str], GroundTruth], tmp_path: Path) -> None:  # fmt: skip
    demo = tmp_path / "ep.lmp"
    demo.write_bytes(b"demo")
    fs = [raw("out_of_bounds", "outside_level", 256, -80, step=500 + i, demo=str(demo))
          for i in range(12)]  # fmt: skip
    fs += [raw("stuck", "wedge", 300, -150, conf=0.4)]
    result = built["buggy"]
    rep = build_report(
        run_id="r1", map_name="MAP03", findings=fs,
        room_cells={"start": (10, 25), "hall": (5, 20), "vault": (0, 25), "balcony": (3, 20)},
        reach_gaps=[{"rooms": ["vault"], "portals_attempted": 4}],
        coverage={"cell": 0.5, "entity": 0.0, "connection": 0.5, "abstract_states": 12},
        truth=truth["MAP03", "buggy"], spec=load_spec(REPO / "maps" / "m03_leak.yaml"),
        meta={"wad_sha256": {}, "git_sha": "abc", "env_steps": 1000, "wall_clock_s": 3.0},
        out_dir=tmp_path / "out", wad_path=result.wad_path,
    )  # fmt: skip
    ids = [f["id"] for f in rep["findings"]]
    assert ids == ["MAP03-oob-01", "MAP03-stuck-01"]  # ranked by severity
    oob = rep["findings"][0]
    assert oob["cluster_size"] == 12 and oob["first_seen_step"] == 500 and oob["room"] == "hall"
    assert oob["repro"]["demo"] == "demos/MAP03-oob-01.lmp"
    assert (tmp_path / "out" / "demos" / "MAP03-oob-01.lmp").read_bytes() == b"demo"
    assert (tmp_path / "out" / result.wad_path.name).exists()
    assert rep["coverage_gaps"] == [{"room": "vault", "cells_unvisited": 25, "portals_attempted": 4}]
    assert "defects" not in json.dumps(rep), "the answer key must not reach a report"
    on_disk = json.loads((tmp_path / "out" / "report.json").read_text())
    assert on_disk == json.loads(json.dumps(rep))
    html = (tmp_path / "out" / "report.html").read_text()
    assert "MAP03-oob-01" in html and "<svg" in html and "Coverage gaps" in html
    assert "d_leak_hall_north" not in html, "the preview must be the intended map"
