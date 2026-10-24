"""M4: the oracles on real gameplay recorded from ViZDoom, not just hand-written traces.

- Fixed maps: every portal routine from tests/test_maps_engine.py (open portals, doors,
  locked doors, switch doors, lifts both ways, teleports, exit) plus random play, recorded
  tic by tic. Every oracle must stay silent.
- Buggy maps: each defect's scenario from tests/test_defects_engine.py, recorded. Each
  must be found by its expected family, matched to the defect, with no other family firing.
"""

from __future__ import annotations

import numpy as np
import pytest
import test_defects_engine as defect_scenarios
import test_maps_engine as portal_routines
from recorder import RecordingDriver
from test_oracles import describe, matches

from qatester.mapgen.cli import BuildResult
from qatester.mapgen.groundtruth import GroundTruth
from qatester.oracles.registry import run_oracles

MAPS = [f"MAP{i:02d}" for i in range(1, 9)]
ROUTINES = [
    portal_routines.test_open_portals_both_ways,
    portal_routines.test_doors_closed_until_used_then_open_both_ways,
    portal_routines.test_locked_doors_need_their_key,
    portal_routines.test_remote_doors_open_only_from_their_switch,
    portal_routines.test_lifts_carry_up_and_walk_down,
    portal_routines.test_teleports_arrive,
    portal_routines.test_exit_switch_ends_episode,
]


def _random_play(d: RecordingDriver, tics: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    d.reset()
    for _ in range(tics):
        if d.game.is_episode_finished():
            break
        d._act([float(rng.random() < 0.7), float(rng.random() < 0.05),
                float(rng.uniform(-20, 20)), float(rng.random() < 0.05)], 1)  # fmt: skip


@pytest.mark.parametrize("lump", MAPS)
def test_oracles_silent_on_real_play_of_fixed_maps(
    built: dict[str, BuildResult], truth: dict[tuple[str, str], GroundTruth], lump: str
) -> None:
    result = built["fixed"]
    d = RecordingDriver(result.cfg_paths[lump])
    try:
        for routine in ROUTINES:
            routine((d, result.compiled[lump]))  # also asserts the routine itself works
        _random_play(d, 2000, seed=int(lump[3:]))
        runs = d.runs(truth[lump, "fixed"])
    finally:
        d.close()
    assert sum(len(ep) for _, ep in runs) > 1000
    findings = run_oracles(runs)
    assert findings == [], describe(findings)


def _defect_cases() -> list[tuple[str, str]]:
    from qatester.mapgen.spec import load_spec

    specs = sorted(portal_routines.REPO.joinpath("maps").glob("*.yaml"))
    return [(f"MAP{load_spec(p).slot:02d}", d.id) for p in specs for d in load_spec(p).defects]


@pytest.mark.parametrize(("lump", "defect_id"), _defect_cases())
def test_oracles_find_each_defect_in_real_play(
    built: dict[str, BuildResult], truth: dict[tuple[str, str], GroundTruth], lump: str,
    defect_id: str,
) -> None:  # fmt: skip
    result = built["buggy"]
    cm = result.compiled[lump]
    defect = next(x for x in cm.spec.defects if x.id == defect_id)
    scenario = defect_scenarios.SCENARIOS[defect.kind]
    repeats = 6 if defect.kind == "wrong_tag" else 1  # Family 4 needs >= 5 failed tries
    d = RecordingDriver(result.cfg_paths[lump])
    try:
        for _ in range(repeats):
            assert scenario(d, cm, defect), "the defect did not even happen"
            # A softlock is judged on episodes that time out stranded.
            d.close_episode(truncated=defect.kind == "one_way_lift")
        runs = d.runs(truth[lump, "buggy"])
    finally:
        d.close()
    gt_defect = next(x for x in truth[lump, "buggy"].defects if x["id"] == defect_id)
    findings = run_oracles(runs)
    assert any(matches(f, gt_defect) for f in findings), describe(findings) or "no findings"
    stray = [f for f in findings if f.oracle != defect.expected_oracle]
    assert not stray, describe(stray)
