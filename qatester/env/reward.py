"""Composite reward R = alpha * R_valid + (1 - alpha) * R_adv (PLAN: Reward), with the
three reward-hacking countermeasures built in (PLAN: Three predictable reward-hacking
failures):

1. Oracle farming: an oracle finding pays once per (family, cluster) per episode. A
   cluster is the Bug Book matching rule: same family, within `cluster_mu`, and for
   invariants the same invariant name.
2. Novelty churn: a cell entry counts only if the cell is "armed". Entering disarms it;
   it re-arms once the agent is at least `rearm_cells` cells away (Chebyshev), i.e. has
   left it by more than one cell width. The bonus itself decays with the shared count.
3. Suicide for reset: -5 on death; time-outs truncate rather than terminate (the env).

Per-episode "first time" (entities used, portals crossed, preconditions met) is this
module's reading of the plan's "the first time each ..." (docs/decisions.md D29).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from qatester.oracles.base import Finding

Cell = tuple[int, int]


@dataclass(frozen=True)
class RewardConfig:
    alpha: float = 0.5
    # validity
    progress: float = 1.0
    goal: float = 5.0
    death: float = -5.0
    time: float = -0.0005
    # adversarial
    cell_novelty: float = 0.3
    state_novelty: float = 0.5
    entity: float = 1.0
    portal: float = 1.0
    oracle: float = 3.0
    bugbook: float = 1.5
    cluster_mu: float = 96.0
    rearm_cells: int = 2

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")


VALID_PARTS = ("progress", "goal", "death", "time")
ADV_PARTS = ("cell_novelty", "state_novelty", "entity", "portal", "oracle", "bugbook")


def cluster_key(f: Finding) -> tuple[str, str | None]:
    return (f.oracle, f.kind if f.oracle == "invariant" else None)


class EpisodeReward:
    """Per-episode reward state: which cells are armed, what has been paid already."""

    def __init__(self, cfg: RewardConfig | None = None) -> None:
        self.cfg = cfg or RewardConfig()
        self.reset()

    def reset(self) -> None:
        self.disarmed: set[Cell] = set()
        self.paid_clusters: list[tuple[tuple[str, str | None], float, float]] = []
        self.entities: set[str] = set()
        self.portals: set[str] = set()
        self.progress: set[str] = set()

    def enter_cell(self, cell: Cell) -> bool:
        """Call when the agent's cell changes; True if this entry counts for novelty."""
        r = self.cfg.rearm_cells
        self.disarmed = {c for c in self.disarmed
                         if max(abs(c[0] - cell[0]), abs(c[1] - cell[1])) < r}  # fmt: skip
        if cell in self.disarmed:
            return False
        self.disarmed.add(cell)
        return True

    def new_findings(self, findings: Iterable[Finding]) -> list[Finding]:
        """The findings that open a new (family, cluster) this episode."""
        out = []
        for f in findings:
            key = cluster_key(f)
            x, y = f.position[0], f.position[1]
            if any(k == key and math.hypot(x - px, y - py) <= self.cfg.cluster_mu
                   for k, px, py in self.paid_clusters):  # fmt: skip
                continue
            self.paid_clusters.append((key, x, y))
            out.append(f)
        return out

    def step(
        self,
        *,
        progress: Iterable[str] = (),
        goal: bool = False,
        died: bool = False,
        cell_count: int | None = None,
        state_count: int | None = None,
        entities: Iterable[str] = (),
        portals: Iterable[str] = (),
        findings: Sequence[Finding] = (),
        bugbook_hits: int = 0,
    ) -> tuple[float, dict[str, float]]:
        """Reward for one step. `cell_count` / `state_count` are the shared visit counts
        before this visit, given only for counted entries."""
        c = self.cfg
        new_progress = set(progress) - self.progress
        self.progress |= new_progress
        new_entities = set(entities) - self.entities
        self.entities |= new_entities
        new_portals = set(portals) - self.portals
        self.portals |= new_portals
        parts = {
            "progress": c.progress * len(new_progress),
            "goal": c.goal if goal else 0.0,
            "death": c.death if died else 0.0,
            "time": c.time,
            "cell_novelty": 0.0 if cell_count is None else c.cell_novelty / math.sqrt(1 + cell_count),
            "state_novelty": (0.0 if state_count is None
                              else c.state_novelty / math.sqrt(1 + state_count)),
            "entity": c.entity * len(new_entities),
            "portal": c.portal * len(new_portals),
            "oracle": c.oracle * len(self.new_findings(findings)),
            "bugbook": c.bugbook * bugbook_hits,
        }  # fmt: skip
        valid = sum(parts[k] for k in VALID_PARTS)
        adv = sum(parts[k] for k in ADV_PARTS)
        return c.alpha * valid + (1.0 - c.alpha) * adv, parts
