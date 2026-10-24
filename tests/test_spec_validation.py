"""The six loader validation rules (PLAN: Validation rules the loader MUST enforce), each
with one passing and one failing fixture, plus the shipped specs."""

import copy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from qatester.mapgen.spec import MapSpec, load_spec

REPO = Path(__file__).resolve().parents[1]

BASE: dict[str, Any] = {
    "name": "T",
    "slot": 1,
    "seed": 0,
    "rooms": [
        {"id": "a", "x0": 0, "y0": 0, "x1": 256, "y1": 256, "floor": 0, "ceiling": 128},
        {"id": "b", "x0": 256, "y0": 0, "x1": 512, "y1": 256, "floor": 0, "ceiling": 128},
    ],
    "portals": [{"id": "p", "a": "a", "b": "b", "kind": "open", "width": 128}],
    "things": [
        {"id": "spawn", "type": "player_start", "room": "a", "pos": [128, 128]},
        {"id": "exit", "type": "exit_switch", "room": "b", "pos": [500, 128]},
    ],
    "goal": {"kind": "reach_exit", "target": "exit", "requires": [], "time_limit_tics": 2100},
    "defects": [],
}


def spec(**patch: Any) -> dict[str, Any]:
    d = copy.deepcopy(BASE)
    d.update(patch)
    return d


def rooms_plus(*extra: dict[str, Any]) -> list[dict[str, Any]]:
    return [*copy.deepcopy(BASE["rooms"]), *extra]


def things_plus(*extra: dict[str, Any]) -> list[dict[str, Any]]:
    return [*copy.deepcopy(BASE["things"]), *extra]


def test_base_fixture_is_valid() -> None:
    MapSpec.model_validate(BASE)


# Rule 1: exactly one player_start.
def test_rule1_one_player_start_passes() -> None:
    MapSpec.model_validate(spec())


def test_rule1_two_player_starts_fails() -> None:
    extra = {"id": "spawn2", "type": "player_start", "room": "b", "pos": [384, 128]}
    with pytest.raises(ValidationError, match="exactly one player_start"):
        MapSpec.model_validate(spec(things=things_plus(extra)))


# Rule 2: portal rooms exist and share a wall at least `width` long (unless teleport).
def test_rule2_teleport_between_non_adjacent_rooms_passes() -> None:
    far = {"id": "c", "x0": 0, "y0": 512, "x1": 256, "y1": 768, "floor": 0, "ceiling": 128}
    portals = [
        *BASE["portals"],
        {"id": "t", "a": "b", "b": "c", "kind": "teleport", "at": [384, 128]},
        {"id": "t2", "a": "c", "b": "a", "kind": "teleport", "at": [128, 640]},
    ]
    MapSpec.model_validate(spec(rooms=rooms_plus(far), portals=portals))


def test_rule2_portal_wider_than_shared_wall_fails() -> None:
    portals = [{"id": "p", "a": "a", "b": "b", "kind": "open", "width": 300}]
    with pytest.raises(ValidationError, match="share 256 mu of wall"):
        MapSpec.model_validate(spec(portals=portals))


def test_rule2_portal_unknown_room_fails() -> None:
    portals = [{"id": "p", "a": "a", "b": "nowhere", "kind": "open", "width": 96}]
    with pytest.raises(ValidationError, match="unknown room"):
        MapSpec.model_validate(spec(portals=portals))


def test_rule2_non_adjacent_rooms_fail() -> None:
    rooms = copy.deepcopy(BASE["rooms"])
    rooms[1].update(x0=320, x1=576)  # 64 mu gap, like the plan's MAP03 balcony
    with pytest.raises(ValidationError, match="share 0 mu of wall"):
        MapSpec.model_validate(spec(rooms=rooms))


# Rule 3: every `requires` names a key_* thing that exists.
def test_rule3_required_key_present_passes() -> None:
    portals = [{"id": "p", "a": "a", "b": "b", "kind": "locked_door", "requires": "key_red"}]
    key = {"id": "k", "type": "key_red", "room": "a", "pos": [64, 64]}
    MapSpec.model_validate(spec(portals=portals, things=things_plus(key)))


def test_rule3_required_key_missing_fails() -> None:
    portals = [{"id": "p", "a": "a", "b": "b", "kind": "locked_door", "requires": "key_red"}]
    with pytest.raises(ValidationError, match="no key_\\* thing provides"):
        MapSpec.model_validate(spec(portals=portals))


# Rule 4: the portal graph is connected and the goal reachable (check_reachable BFS).
def test_rule4_reachable_passes() -> None:
    assert MapSpec.model_validate(spec()).check_reachable() == {"a", "b"}


def test_rule4_disconnected_room_fails() -> None:
    with pytest.raises(ValidationError, match="unreachable from spawn"):
        MapSpec.model_validate(spec(portals=[]))


def test_rule4_key_behind_its_own_door_fails() -> None:
    portals = [{"id": "p", "a": "a", "b": "b", "kind": "locked_door", "requires": "key_red"}]
    key = {"id": "k", "type": "key_red", "room": "b", "pos": [384, 64]}
    with pytest.raises(ValidationError, match="unreachable from spawn"):
        MapSpec.model_validate(spec(portals=portals, things=things_plus(key)))


# Rule 5: rooms overlap only as declared union_with pairs.
def test_rule5_declared_union_passes() -> None:
    ell = {"id": "c", "x0": 128, "y0": 128, "x1": 256, "y1": 384, "floor": 0, "ceiling": 128,
           "union_with": "a"}  # fmt: skip
    MapSpec.model_validate(spec(rooms=rooms_plus(ell)))


def test_rule5_undeclared_overlap_fails() -> None:
    ell = {"id": "c", "x0": 128, "y0": 128, "x1": 256, "y1": 384, "floor": 0, "ceiling": 128}
    with pytest.raises(ValidationError, match="overlap without union_with"):
        MapSpec.model_validate(spec(rooms=rooms_plus(ell)))


# Rule 6: every defect's target exists and its kind is in the registry.
DEFECT = {"id": "d", "kind": "missing_blocking", "target": "b",
          "expected_oracle": "out_of_bounds", "severity": "high"}  # fmt: skip


def test_rule6_valid_defect_passes() -> None:
    MapSpec.model_validate(spec(defects=[DEFECT]))


def test_rule6_unknown_target_fails() -> None:
    with pytest.raises(ValidationError, match="unknown target"):
        MapSpec.model_validate(spec(defects=[{**DEFECT, "target": "ghost"}]))


def test_rule6_unknown_kind_fails() -> None:
    with pytest.raises(ValidationError):
        MapSpec.model_validate(spec(defects=[{**DEFECT, "kind": "cosmic_ray"}]))


@pytest.mark.parametrize("path", sorted((REPO / "maps").glob("*.yaml")), ids=lambda p: p.stem)
def test_shipped_specs_are_valid(path: Path) -> None:
    load_spec(path)
