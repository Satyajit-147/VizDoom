"""Family 5: invariants, written once per property and applied to every map (PLAN: the
declarative DSL). A violated invariant is reported once per episode, confidence 1.0.

Deviations from the plan's list (docs/decisions.md D25):
- `z_in_sector` is dropped: it is Family 3's z_clip test; keeping both reports one event
  under two families.
- `lift_returns` is dropped: linedef triggers are not observable from the engine, and a
  lift with no way back is Family 6's softlock.
- `had_powerup` is true only after picking up a thing the spec declares a powerup
  (POWERUP_TYPES; the corpus declares none), so a Soulsphere the spec calls a medikit
  still breaks `health_cap`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from qatester.oracles.base import EpisodeContext, Finding, StepRecord
from qatester.oracles.level import Level, TraversalTracker

PICKUP_MU = 96.0  # a pickup within this distance of the player explains a stat change
HEALTH_TYPES = frozenset({"health"})
POWERUP_TYPES: frozenset[str] = frozenset()  # spec thing types that may exceed caps
HEALTH_WORDS = ("medikit", "stimpack", "health", "supercharge")


@dataclass(frozen=True)
class InvState:
    step: StepRecord
    picked_up_health: bool = False
    had_powerup: bool = False
    used_key_door: bool = False
    portals_traversed: tuple[str, ...] = field(default_factory=tuple)

    @property
    def health(self) -> float:
        return self.step.health

    @property
    def ammo(self) -> float:
        return self.step.ammo

    @property
    def keys(self) -> frozenset[str]:
        return self.step.keys

    @property
    def killcount(self) -> int:
        return self.step.killcount


Check = Callable[[InvState, InvState | None, Level], bool]


@dataclass(frozen=True)
class Inv:
    name: str
    holds: Check


INVARIANTS = [
    Inv("health_cap", lambda s, p, lv: s.health <= 100 or s.had_powerup),
    Inv("health_monotone",
        lambda s, p, lv: p is None or s.health <= p.health or s.picked_up_health),
    Inv("ammo_nonneg", lambda s, p, lv: s.ammo >= 0),
    Inv("no_phantom_keys", lambda s, p, lv: p is None or s.keys >= p.keys or s.used_key_door),
    Inv("locked_needs_key", lambda s, p, lv: all(
        lv.portals[t].requires is None or lv.portals[t].requires in s.keys
        for t in s.portals_traversed)),
    Inv("kills_le_spawned", lambda s, p, lv: s.killcount <= lv.monster_count),
]  # fmt: skip


class InvariantOracle:
    name = "invariant"

    def __init__(self, invariants: list[Inv] | None = None) -> None:
        self.invariants = invariants if invariants is not None else INVARIANTS
        self.ctx: EpisodeContext | None = None
        self.level: Level | None = None
        self.tracker: TraversalTracker | None = None
        self.prev: InvState | None = None
        self.had_powerup = False
        self.reported: set[str] = set()

    def reset(self, episode: EpisodeContext) -> None:
        self.ctx = episode
        self.level = Level(episode.truth)
        self.tracker = TraversalTracker(self.level)
        self.prev = None
        self.had_powerup = False
        self.reported = set()

    def _near(self, step: StepRecord, types: frozenset[str]) -> bool:
        assert self.level is not None
        points = [(step.x, step.y)]
        if self.prev is not None:
            points.append((self.prev.step.x, self.prev.step.y))
        return any(
            math.hypot(px - t["pos"][0], py - t["pos"][1]) <= PICKUP_MU
            for t in self.level.things if t["type"] in types for px, py in points
        )  # fmt: skip

    def observe(self, step: StepRecord) -> list[Finding]:
        assert self.level is not None and self.tracker is not None and self.ctx is not None
        prev = self.prev
        gained = prev is not None and step.health > prev.health
        picked = gained and (
            self._near(step, HEALTH_TYPES)
            or any(w in n.lower() for n in step.notifications for w in HEALTH_WORDS)
        )
        if POWERUP_TYPES and self._near(step, POWERUP_TYPES):
            self.had_powerup = True
        state = InvState(step, picked_up_health=picked, had_powerup=self.had_powerup,
                         portals_traversed=tuple(self.tracker.update(step.x, step.y)))  # fmt: skip
        out = []
        for inv in self.invariants:
            if inv.name in self.reported or inv.holds(state, prev, self.level):
                continue
            self.reported.add(inv.name)
            out.append(Finding(
                self.name, inv.name, (step.x, step.y, step.z), (step.tic, step.tic), 1.0,
                {"episode": self.ctx.episode, "health": step.health, "ammo": step.ammo,
                 "keys": sorted(step.keys), "killcount": step.killcount,
                 "portals_traversed": list(state.portals_traversed),
                 "previous_health": None if prev is None else prev.health},
            ))  # fmt: skip
        self.prev = state
        return out

    def finalize(self) -> list[Finding]:
        return []
