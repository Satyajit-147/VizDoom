"""Family 6: softlock (PLAN). Regression replay arrives with the Bug Book (M7).

Confinement, judged at finalize() over the run (docs/decisions.md D25): an episode is a
candidate when it timed out, the agent spent at least the last 40% of it in one room
sector that is not the goal room, and during that stay *every exit of the room had a
proper attempt that failed* (level.AttemptTracker: a door or lift that does not respond
to USE, an opening the agent pushes through without progress, a pad that does not send
it). The plan's version flags any agent that idles or wanders in one room; this one
needs evidence that the way out is broken. Confidence 0.8, or 0.95 when two or more
episodes end confined in the same room.

Teleport loop, streamed: at least `min_loop_hops` teleport-sized jumps in a row, each
within `max_lag_tics` of the previous, bouncing between the same two rooms. Confidence
0.9. The plan's "autocorrelation of position at lag < 70 tics above 0.9" is replaced:
any smooth walk has near-1 autocorrelation at short lags, so it does not single out a
loop.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from qatester.oracles.base import (
    AttemptConfig,
    EpisodeContext,
    Finding,
    SoftlockConfig,
    StepRecord,
)
from qatester.oracles.level import AttemptTracker, Level, TraversalTracker


@dataclass(frozen=True)
class _Hop:
    tic: int
    src: tuple[float, float, float]
    from_sector: int
    to_sector: int


class SoftlockOracle:
    name = "softlock"

    def __init__(self, cfg: SoftlockConfig | None = None,
                 attempts: AttemptConfig | None = None) -> None:  # fmt: skip
        self.cfg = cfg or SoftlockConfig()
        self.attempt_cfg = attempts or AttemptConfig()
        self.ctx: EpisodeContext | None = None
        self.level: Level | None = None
        self.tracker: TraversalTracker | None = None
        self.attempts: AttemptTracker | None = None
        self.trail: list[tuple[int, float, float, float, int]] = []  # tic, x, y, z, sector
        self.failures: list[tuple[int, str]] = []  # (tic, portal) failed attempts
        self.ended = False
        self.prev: StepRecord | None = None
        self.hops: list[_Hop] = []
        self.loop_reported = False
        # room sector -> [(episode, centroid, tic0, tic1)]
        self.candidates: dict[int, list[tuple[int, tuple[float, float, float], int, int]]] = (
            defaultdict(list)
        )

    def reset(self, episode: EpisodeContext) -> None:
        if self.trail and not self.ended:
            self._end_episode(timed_out=False)
        self.ctx = episode
        if self.level is None:
            self.level = Level(episode.truth)
            self.tracker = TraversalTracker(self.level)
            self.attempts = AttemptTracker(self.level, self.attempt_cfg)
        assert self.tracker is not None and self.attempts is not None
        self.tracker.reset()
        self.attempts.reset()
        self.trail = []
        self.failures = []
        self.ended = False
        self.prev = None
        self.hops = []
        self.loop_reported = False

    def _loop(self, step: StepRecord) -> list[Finding]:
        assert self.level is not None and self.ctx is not None
        p = self.prev
        if p is None or math.hypot(step.x - p.x, step.y - p.y) < self.cfg.teleport_jump_mu:
            return []
        hop = _Hop(step.tic, (p.x, p.y, p.z), self.level.sector_at(p.x, p.y),
                   self.level.sector_at(step.x, step.y))  # fmt: skip
        if self.hops and (hop.tic - self.hops[-1].tic > self.cfg.max_lag_tics or (
            (hop.from_sector, hop.to_sector) != (self.hops[-1].to_sector, self.hops[-1].from_sector)
        )):  # fmt: skip
            self.hops = []
        self.hops.append(hop)
        if self.loop_reported or len(self.hops) < self.cfg.min_loop_hops:
            return []
        self.loop_reported = True
        first = self.hops[0]
        return [Finding(self.name, "teleport_loop", first.src, (first.tic, hop.tic),
                        self.cfg.loop_confidence,
                        {"episode": self.ctx.episode, "hops": len(self.hops),
                         "sectors": [first.from_sector, first.to_sector]})]  # fmt: skip

    def observe(self, step: StepRecord) -> list[Finding]:
        assert self.level is not None and self.tracker is not None and self.attempts
        out = self._loop(step)
        crossed = self.tracker.update(step.x, step.y)
        self.failures += [(step.tic, pid) for pid in self.attempts.update(step, crossed)]
        self.trail.append((step.tic, step.x, step.y, step.z, self.level.sector_at(step.x, step.y)))
        self.prev = step
        if step.done:
            self._end_episode(timed_out=step.truncated)
        return out

    def _end_episode(self, timed_out: bool) -> None:
        self.ended = True
        lv, ctx = self.level, self.ctx
        if not timed_out or lv is None or ctx is None or len(self.trail) < 2:
            return
        rooms = [t for t in self.trail if lv.is_room_sector(t[4])]
        if not rooms:
            return
        room = rooms[-1][4]
        # The final stay: since the agent was last in a different room sector.
        stay_start = self.trail[0][0]
        for t in reversed(rooms):
            if t[4] != room:
                stay_start = t[0]
                break
        t0, t1 = self.trail[0][0], self.trail[-1][0]
        if t1 - stay_start < self.cfg.tail_fraction * (t1 - t0) or room == lv.goal_sector:
            return
        exits = {p.id for p in lv.portals.values()
                 if lv.room_sector[p.a] == room
                 or (p.kind != "teleport" and lv.room_sector[p.b] == room)}  # fmt: skip
        tried = {pid for tic, pid in self.failures if tic >= stay_start}
        if not exits or not exits <= tried:
            return  # some exit was never properly tried, or it worked
        stay = [t for t in self.trail if t[0] >= stay_start]
        n = len(stay)
        centroid = (sum(t[1] for t in stay) / n, sum(t[2] for t in stay) / n,
                    sum(t[3] for t in stay) / n)  # fmt: skip
        self.candidates[room].append((ctx.episode, centroid, stay_start, t1))

    def finalize(self) -> list[Finding]:
        if self.trail and not self.ended:
            self._end_episode(timed_out=False)
        out = []
        for room, eps in self.candidates.items():
            conf = self.cfg.repeat_confidence if len(eps) >= 2 else self.cfg.confidence
            for episode, centroid, a, b in eps:
                out.append(Finding(self.name, "confined", centroid, (a, b), conf,
                                   {"episode": episode, "sector": room,
                                    "rooms": self.level.rooms_of(room) if self.level else [],
                                    "episodes_confined_here": len(eps)}))  # fmt: skip
        return out
