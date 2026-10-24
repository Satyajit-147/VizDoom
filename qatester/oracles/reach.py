"""Family 4: unreachable region (PLAN: I \\ V partitioned by why). Run-level: evaluated at
finalize() over every episode of the run on one map.

Rooms nobody visited are grouped into connected components. For each component:
- some room is not in the intended portal graph from spawn -> `broken_portal_graph`, 0.95;
- every portal into it had >= 5 proper attempts that failed and was never traversed ->
  `broken_trigger`, 0.85 (MAP04);
- otherwise the tester just did not get there: a coverage gap, never a finding.

"Proper attempt that failed" is defined once, in level.AttemptTracker: hammering a locked
door without its key, or USE that does not reach the door, is not an attempt.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from qatester.oracles.base import AttemptConfig, EpisodeContext, Finding, ReachConfig, StepRecord
from qatester.oracles.level import AttemptTracker, Level, TraversalTracker


class ReachOracle:
    name = "unreachable"

    def __init__(self, cfg: ReachConfig | None = None,
                 attempts: AttemptConfig | None = None) -> None:  # fmt: skip
        self.cfg = cfg or ReachConfig()
        self.attempt_cfg = attempts or AttemptConfig()
        self.ctx: EpisodeContext | None = None
        self.level: Level | None = None
        self.tracker: TraversalTracker | None = None
        self.attempts: AttemptTracker | None = None
        self.map: str | None = None
        self.visited_sectors: set[int] = set()
        self.traversed: set[str] = set()
        self.failed: Counter[str] = Counter()
        self.last_tic = 0
        self.coverage_gaps: list[dict[str, Any]] = []

    def reset(self, episode: EpisodeContext) -> None:
        if self.map is not None and episode.map != self.map:
            raise ValueError("ReachOracle accumulates one map per run; make one per map")
        if self.level is None:
            self.level = Level(episode.truth)
            self.tracker = TraversalTracker(self.level)
            self.attempts = AttemptTracker(self.level, self.attempt_cfg)
        self.map = episode.map
        self.ctx = episode
        assert self.tracker is not None and self.attempts is not None
        self.tracker.reset()
        self.attempts.reset()

    def observe(self, step: StepRecord) -> list[Finding]:
        lv, tracker, attempts = self.level, self.tracker, self.attempts
        assert lv is not None and tracker is not None and attempts is not None
        s = lv.sector_at(step.x, step.y)
        if lv.is_room_sector(s):
            self.visited_sectors.add(s)
        crossed = tracker.update(step.x, step.y)
        self.traversed.update(crossed)
        self.failed.update(attempts.update(step, crossed))
        self.last_tic = step.tic
        return []

    def _reachable_in_graph(self) -> set[int]:
        """Room sectors reachable from spawn in the intended portal graph, ignoring keys
        and switches. Works on sectors, so rooms joined by `union_with` (one sector) are
        one space."""
        lv = self.level
        assert lv is not None
        reached = {lv.room_sector[lv.spawn_room]}
        changed = True
        while changed:
            changed = False
            for p in lv.portals.values():
                a, b = lv.room_sector[p.a], lv.room_sector[p.b]
                ends = [(a, b)] if p.kind == "teleport" else [(a, b), (b, a)]
                for src, dst in ends:
                    if src in reached and dst not in reached:
                        reached.add(dst)
                        changed = True
        return reached

    def finalize(self) -> list[Finding]:
        lv, ctx = self.level, self.ctx
        if lv is None or ctx is None or not self.visited_sectors:
            return []
        unvisited = set(lv.room_sector.values()) - self.visited_sectors
        in_graph = self._reachable_in_graph()

        parent = {s: s for s in unvisited}

        def root(s: int) -> int:
            while parent[s] != s:
                s = parent[s]
            return s

        for p in lv.portals.values():
            a, b = lv.room_sector[p.a], lv.room_sector[p.b]
            if a in unvisited and b in unvisited:
                parent[root(a)] = root(b)
        components: dict[int, set[int]] = {}
        for s in unvisited:
            components.setdefault(root(s), set()).add(s)

        out: list[Finding] = []
        self.coverage_gaps = []
        for comp in components.values():
            rooms = sorted(r for s in comp for r in lv.rooms_of(s))
            cells = self._cells_in(comp)
            if comp - in_graph:
                rects = [lv.rooms[r] for r in rooms]
                cx = sum((r[0] + r[2]) / 2 for r in rects) / len(rects)
                cy = sum((r[1] + r[3]) / 2 for r in rects) / len(rects)
                out.append(Finding(self.name, "broken_portal_graph", (cx, cy, 0.0),
                                   (0, self.last_tic), self.cfg.confidence_graph,
                                   {"episode": ctx.episode, "rooms": rooms,
                                    "cells_unvisited": cells}))  # fmt: skip
                continue
            entries = [
                p for p in lv.portals.values()
                if (lv.room_sector[p.a] in self.visited_sectors and lv.room_sector[p.b] in comp)
                or (p.kind != "teleport" and lv.room_sector[p.b] in self.visited_sectors
                    and lv.room_sector[p.a] in comp)
            ]  # fmt: skip
            broken = [
                p for p in entries
                if self.failed[p.id] >= self.cfg.use_attempts_for_broken_trigger
                and p.id not in self.traversed
            ]  # fmt: skip
            if entries and len(broken) == len(entries):
                worst = max(broken, key=lambda p: self.failed[p.id])
                x = (worst.area[0] + worst.area[2]) / 2
                y = (worst.area[1] + worst.area[3]) / 2
                out.append(Finding(self.name, "broken_trigger", (x, y, 0.0), (0, self.last_tic),
                                   self.cfg.confidence_trigger,
                                   {"episode": ctx.episode, "rooms": rooms,
                                    "cells_unvisited": cells,
                                    "failed_attempts": {p.id: self.failed[p.id]
                                                        for p in broken}}))  # fmt: skip
            else:
                self.coverage_gaps.append({
                    "rooms": rooms, "cells_unvisited": cells,
                    "portals_attempted": sum(self.failed[p.id] for p in entries),
                })  # fmt: skip
        return out

    def _cells_in(self, comp: set[int]) -> int:
        lv = self.level
        assert lv is not None
        w = lv.truth.data["walkable"]
        ox, oy, size = w["origin"][0], w["origin"][1], w["cell_size"]
        return sum(
            1
            for j, row in enumerate(w["rows"])
            for i, c in enumerate(row)
            if c == "1" and lv.sector_at(ox + (i + 0.5) * size, oy + (j + 0.5) * size) in comp
        )
