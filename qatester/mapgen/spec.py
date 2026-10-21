"""MapSpec Pydantic models (PLAN: Environment, part A). Field names are normative.

M1 provides the models and YAML loading only. The six loader validation rules and
check_reachable() arrive in M2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PortalKind = Literal["open", "door", "locked_door", "lift", "teleport"]
ThingType = Literal[
    "player_start", "key_red", "key_blue", "key_yellow", "health", "armour", "ammo",
    "monster_imp", "exit_switch", "marker",
]  # fmt: skip
GoalKind = Literal["reach_exit", "collect_all", "survive"]
DefectKind = Literal[
    "missing_blocking", "wrong_tag", "wrong_special", "geometry_trap", "one_way_lift",
    "teleport_loop", "bad_pickup",
]  # fmt: skip
Severity = Literal["low", "medium", "high"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Room(_Model):
    id: str
    x0: int
    y0: int
    x1: int
    y1: int
    floor: int
    ceiling: int
    tag: int | None = None
    damage: int = 0
    light: int = 160
    union_with: str | None = None

    @model_validator(mode="after")
    def _positive_extent(self) -> Room:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"room {self.id}: need x1 > x0 and y1 > y0")
        return self


class Portal(_Model):
    id: str
    a: str
    b: str
    kind: PortalKind
    width: int = 96
    requires: str | None = None
    at: tuple[int, int] | None = None


class Thing(_Model):
    id: str
    type: ThingType
    room: str
    pos: tuple[int, int]
    interactive: bool = False


class Goal(_Model):
    kind: GoalKind
    target: str | None = None
    requires: list[str] = Field(default_factory=list)
    time_limit_tics: int


class Defect(_Model):
    id: str
    kind: DefectKind
    target: str
    params: dict[str, object] = Field(default_factory=dict)
    expected_oracle: str
    severity: Severity
    notes: str = ""


class MapSpec(_Model):
    name: str
    slot: int = Field(ge=1, le=32)
    cell_size: int = 64
    rooms: list[Room] = Field(min_length=1)
    portals: list[Portal] = Field(default_factory=list)
    things: list[Thing]
    goal: Goal
    defects: list[Defect] = Field(default_factory=list)
    seed: int

    @property
    def map_lump(self) -> str:
        return f"MAP{self.slot:02d}"

    def without_defects(self) -> MapSpec:
        """The `fixed` build variant (PLAN: Build matrix)."""
        return self.model_copy(update={"defects": []})


def load_spec(path: Path) -> MapSpec:
    with path.open(encoding="utf-8") as f:
        return MapSpec.model_validate(yaml.safe_load(f))
