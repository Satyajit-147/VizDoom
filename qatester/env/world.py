"""What the env derives from the engine state plus the *intended* level: keys held,
things picked up or used, which doors are open, what is in USE reach, and which
preconditions of the goal are met.

ViZDoom has no game variable for keys. A key (or any pickup) counts as taken when no
pickup object stands at its spawn point any more; this survives `load_state`, unlike
parsing the "passcard secured" message (docs/decisions.md D29).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from qatester.mapgen.groundtruth import GroundTruth
from qatester.oracles.level import (
    Level,
    ray_hits_rect,
    ray_hits_segment,
    rect_distance,
    segment_distance,
)

PICKUP_CLASSES = frozenset({
    "RedCard", "BlueCard", "YellowCard", "RedSkull", "BlueSkull", "YellowSkull",
    "Medikit", "Stimpack", "Soulsphere", "Megasphere", "HealthBonus",
    "GreenArmor", "BlueArmor", "ArmorBonus",
    "Clip", "ClipBox", "Shell", "ShellBox", "RocketAmmo", "RocketBox", "Cell", "CellPack",
})  # fmt: skip
PICKUP_TYPES = frozenset({"key_red", "key_blue", "key_yellow", "health", "armour", "ammo"})
SWITCH_TYPES = frozenset({"switch", "exit_switch"})
KEY_BITS = {"key_red": 1, "key_blue": 2, "key_yellow": 4}
SPAWN_MATCH_MU = 24.0  # a pickup object this close to a thing's spawn point is that thing
USE_RANGE_MU = 64.0
DOOR_OPEN_MU = 8.0


@dataclass(frozen=True)
class PickupThing:
    id: str
    type: str
    x: float
    y: float


class World:
    def __init__(self, truth: GroundTruth) -> None:
        self.level = Level(truth)  # refuses a view with defects in it
        d = truth.data
        self.pickups = [PickupThing(t["id"], t["type"], *t["pos"][:2]) for t in d["things"]
                        if t["interactive"] and t["type"] in PICKUP_TYPES]  # fmt: skip
        lines = self.level.switch_lines
        self.switches = {t["id"]: lines[t["id"]] for t in d["things"]
                         if t["interactive"] and t["type"] in SWITCH_TYPES and t["id"] in lines}  # fmt: skip
        self.usable_areas = [p.area for p in self.level.portals.values()
                             if p.kind in ("door", "locked_door", "lift")]  # fmt: skip
        self.door_sectors = sorted(s["index"] for s in d["sectors"] if s["kind"] == "door")
        goal = d["goal"]
        self.goal_kind: str = goal["kind"]
        self.required_keys: tuple[str, ...] = tuple(goal.get("requires") or ())
        self.critical_portals = self._critical_portals(goal.get("room"))

    # --- the goal's preconditions ---------------------------------------------------
    def _critical_portals(self, goal_room: str | None) -> frozenset[str]:
        """Non-open portals every route from spawn to the goal room must pass through."""
        lv = self.level
        if goal_room is None:
            return frozenset()
        start, target = lv.room_sector[lv.spawn_room], lv.room_sector[goal_room]

        def reachable(without: str | None) -> bool:
            seen, todo = {start}, deque([start])
            while todo:
                s = todo.popleft()
                if s == target:
                    return True
                for p in lv.portals.values():
                    if p.id == without:
                        continue
                    a, b = lv.room_sector[p.a], lv.room_sector[p.b]
                    nxt = b if a == s else (a if b == s and p.kind != "teleport" else None)
                    if nxt is not None and nxt not in seen:
                        seen.add(nxt)
                        todo.append(nxt)
            return False

        if not reachable(None):
            return frozenset()
        return frozenset(p.id for p in lv.portals.values()
                         if p.kind != "open" and not reachable(p.id))  # fmt: skip

    def progress(self, keys: Iterable[str], traversed: Iterable[str]) -> set[str]:
        """Goal preconditions met by holding `keys` and crossing `traversed` portals."""
        held = set(keys)
        out = {f"key:{k}" for k in self.required_keys if k in held}
        out |= {f"gate:{p}" for p in traversed if p in self.critical_portals}
        return out

    # --- engine state -----------------------------------------------------------------
    def taken(self, objects: Iterable[Any]) -> set[str]:
        """Pickup things whose object is gone from their spawn point."""
        present = [(o.position_x, o.position_y) for o in objects if o.name in PICKUP_CLASSES]
        return {
            t.id for t in self.pickups
            if not any(math.hypot(px - t.x, py - t.y) <= SPAWN_MATCH_MU for px, py in present)
        }  # fmt: skip

    def keys_held(self, taken: Iterable[str]) -> frozenset[str]:
        ids = set(taken)
        return frozenset(t.type for t in self.pickups if t.id in ids and t.type in KEY_BITS)

    def doors_mask(self, floors: tuple[float, ...], ceilings: tuple[float, ...]) -> int:
        mask = 0
        for bit, s in enumerate(self.door_sectors):
            if s < len(floors) and ceilings[s] - floors[s] > DOOR_OPEN_MU:
                mask |= 1 << bit
        return mask

    def use_target(self, x: float, y: float, angle: float) -> bool:
        r = USE_RANGE_MU
        return any(
            rect_distance(x, y, a) <= r and ray_hits_rect(x, y, angle, r, a)
            for a in self.usable_areas
        ) or any(
            segment_distance(x, y, s) <= r and ray_hits_segment(x, y, angle, r, s)
            for s in self.switches.values()
        )

    def switches_hit(self, x: float, y: float, angle: float) -> list[str]:
        r = USE_RANGE_MU
        return [i for i, s in self.switches.items()
                if segment_distance(x, y, s) <= r and ray_hits_segment(x, y, angle, r, s)]  # fmt: skip


def keys_mask(keys: Iterable[str]) -> int:
    return sum(KEY_BITS[k] for k in set(keys) if k in KEY_BITS)


def count_pickups_in_view(labels: Iterable[Any]) -> int:
    return sum(1 for lab in labels if lab.object_name in PICKUP_CLASSES)
