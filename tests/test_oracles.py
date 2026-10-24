"""M4 gate: oracles on synthetic traces (PLAN: Testing the oracles).

- zero findings on the entire MAP01 CLEAN trace corpus, trace by trace and as one run;
- each family fires on its positive traces with the expected kinds and nothing else;
- each family's near-misses produce no findings at all;
- every injected defect is detected, by its expected family, in its positive trace;
- oracles only ever see the intent-only ground truth;
- thresholds equal the block frozen in docs/decisions.md.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml
from fixtures.traces import clean_map01, defect_positives, families
from tracebuilder import Trace, episodes

from qatester.mapgen.groundtruth import GroundTruth
from qatester.oracles.base import Finding, OracleConfig
from qatester.oracles.level import Level
from qatester.oracles.registry import FAMILIES, run_oracles

REPO = Path(__file__).resolve().parents[1]
Truths = dict[tuple[str, str], GroundTruth]
MATCH_RADIUS_MU = 128.0


def describe(findings: list[Finding]) -> str:
    return "; ".join(f"{f.oracle}/{f.kind} at ({f.position[0]:.0f},{f.position[1]:.0f}) "
                     f"conf {f.confidence}" for f in findings)  # fmt: skip


def matches(f: Finding, defect: dict[str, Any]) -> bool:
    """Decision 7 (docs/decisions.md): same family, and within 128 mu of the recorded
    position or inside one of the defect's region rectangles."""
    if f.oracle != defect["expected_oracle"]:
        return False
    x, y = f.position[0], f.position[1]
    near = math.hypot(x - defect["position"][0], y - defect["position"][1]) <= MATCH_RADIUS_MU
    inside = any(r[0] <= x <= r[2] and r[1] <= y <= r[3] for r in defect["region"])
    return near or inside


# --- CLEAN corpus --------------------------------------------------------------------

@pytest.mark.parametrize("make", clean_map01.CORPUS, ids=lambda f: f.__name__)
def test_clean_trace_has_no_findings(truth: Truths, make: Any) -> None:
    t = truth["MAP01", "buggy"]  # CLEAN has no defects: buggy == fixed
    findings = run_oracles(episodes(t, make(t)))
    assert findings == [], describe(findings)


def test_clean_corpus_as_one_run_has_no_findings(truth: Truths) -> None:
    t = truth["MAP01", "buggy"]
    findings = run_oracles(episodes(t, *[make(t) for make in clean_map01.CORPUS]))
    assert findings == [], describe(findings)


MAPS = [f"MAP{i:02d}" for i in range(1, 9)]


@pytest.mark.parametrize("lump", MAPS)
def test_idling_at_spawn_is_silent_on_every_fixed_map(truth: Truths, lump: str) -> None:
    """Covers map shapes MAP01 lacks: lifts, teleports, union rooms, remote doors."""
    t = truth[lump, "fixed"]
    findings = run_oracles(episodes(t, Trace(t).until(10_500).end(), Trace(t).wait(400).end()))
    assert findings == [], describe(findings)


# --- families: positives and near-misses ---------------------------------------------

POSITIVE_CASES = [(fam, fx, kinds) for fam, cases in families.POSITIVES.items()
                  for fx, kinds in cases]  # fmt: skip
NEAR_CASES = [(fam, fx) for fam, fxs in families.NEAR_MISSES.items() for fx in fxs]


def test_every_family_has_a_positive_and_two_near_misses() -> None:
    assert set(families.POSITIVES) == set(families.NEAR_MISSES) == set(FAMILIES)
    assert all(len(v) >= 2 for v in families.NEAR_MISSES.values())


@pytest.mark.parametrize(("family", "fixture", "kinds"), POSITIVE_CASES,
                         ids=[fx.__name__ for _, fx, _ in POSITIVE_CASES])  # fmt: skip
def test_positive_fires_expected_family(truth: Truths, family: str, fixture: Any,
                                        kinds: set[str]) -> None:  # fmt: skip
    findings = run_oracles(fixture(truth))
    assert findings, f"{fixture.__name__}: nothing fired"
    assert {f.oracle for f in findings} == {family}, describe(findings)
    assert {f.kind for f in findings} == kinds, describe(findings)
    assert all(0.0 < f.confidence <= 1.0 for f in findings)


@pytest.mark.parametrize(("family", "fixture"), NEAR_CASES,
                         ids=[fx.__name__ for _, fx in NEAR_CASES])  # fmt: skip
def test_near_miss_is_silent(truth: Truths, family: str, fixture: Any) -> None:
    findings = run_oracles(fixture(truth))
    assert findings == [], describe(findings)


def test_confidences_follow_the_plan(truth: Truths) -> None:
    wedge = run_oracles(families.stuck_wedge_positive(truth))
    assert [f.confidence for f in wedge] == [1.0]  # 500 tics >= 350: saturated
    softlock = run_oracles(families.softlock_positive(truth))
    assert {f.confidence for f in softlock} == {0.95}  # two episodes, same room
    reach = run_oracles(families.reach_positive(truth))
    assert [f.confidence for f in reach] == [0.85]


# --- every injected defect -----------------------------------------------------------

def _all_defects(truth: Truths) -> list[tuple[str, dict[str, Any]]]:
    return [(m, d) for (m, v), t in sorted(truth.items()) if v == "buggy" for d in t.defects]


def test_every_injected_defect_is_detected(truth: Truths) -> None:
    missed = []
    count = 0
    for lump, d in _all_defects(truth):
        count += 1
        t = truth[lump, "buggy"]
        runs = defect_positives.GENERATORS[d["kind"]](t, d)
        findings = run_oracles(episodes(t, *runs))
        stray = [f for f in findings if f.oracle != d["expected_oracle"]]
        if not any(matches(f, d) for f in findings) or stray:
            missed.append(f"{lump} {d['id']} ({d['kind']} -> {d['expected_oracle']}): "
                          f"{describe(findings) or 'no findings'}")  # fmt: skip
    assert count == 15
    assert not missed, "\n".join(missed)


# --- isolation from the answer key ---------------------------------------------------

def test_oracles_refuse_full_ground_truth(truth: Truths) -> None:
    with pytest.raises(PermissionError):
        Level(truth["MAP04", "buggy"])
    view = truth["MAP04", "buggy"].intent_only()
    with pytest.raises(PermissionError):
        _ = view.defects
    Level(view)


def test_oracle_code_never_mentions_defects() -> None:
    for path in sorted((REPO / "qatester" / "oracles").glob("*.py")):
        code = re.sub(r'""".*?"""', "", path.read_text(), flags=re.DOTALL)  # ignore docstrings
        code = re.sub(r"#.*", "", code)
        code = code.replace('if "defects" in d:', "")  # level.py's guard that refuses them
        assert "defect" not in code.lower(), f"{path.name} references defects"


# --- frozen thresholds ---------------------------------------------------------------

def _frozen_block() -> dict[str, Any]:
    text = (REPO / "docs" / "decisions.md").read_text()
    m = re.search(r"<!-- frozen-oracle-thresholds -->\s*```yaml\n(.*?)```", text, re.DOTALL)
    assert m, "docs/decisions.md has no frozen threshold block"
    block: dict[str, Any] = yaml.safe_load(m.group(1))["oracles"]
    return block


def test_thresholds_match_the_frozen_block() -> None:
    assert asdict(OracleConfig()) == _frozen_block()
    assert OracleConfig.from_dict(_frozen_block()) == OracleConfig()


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError):
        OracleConfig.from_dict({"stuck": {"window": 70}})
    with pytest.raises(ValueError):
        OracleConfig.from_dict({"telepathy": {}})
