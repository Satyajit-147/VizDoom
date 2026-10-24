"""Family 2: stuck (PLAN: intent versus effect).

Wedge: over a window of W tics the agent commanded movement on more than 80% of steps,
moved less than 24 mu (diameter of the xy positions), made failed escape attempts in at
least 3 of the 4 world directions, and is alive on the ground. Confidence
min(1, duration / 350 tics).

An escape attempt (docs/decisions.md D25) is one world direction (commanded move +
facing, bucketed to N/E/S/W) held for at least `push_tics` while the player moves less
than `push_progress_mu`. Counting directions in the world frame catches an agent that
turns while pushing forward; requiring a *sustained* push stops an agent jittering in a
corner, which faces many directions for a tic or two each, from looking wedged.

Vertical trap: the agent stands in a sector whose live floor is more than 24 mu below
every neighbouring sector's live floor, and the intended level gives that sector no
lift. Neighbours and lifts come from the intended level; floors come from the engine,
because the intended level has no traps by construction. Confidence 0.9, once per
sector per episode.
"""

from __future__ import annotations

import math
from collections import deque

from qatester.oracles.base import EpisodeContext, Finding, StepRecord, StuckConfig
from qatester.oracles.level import Level

DIR_OFFSET = {"forward": 0.0, "left": 90.0, "back": 180.0, "right": -90.0}


def world_quadrants(step: StepRecord) -> set[int]:
    return {
        round(((step.angle + DIR_OFFSET[d]) % 360.0) / 90.0) % 4 for d in step.move_dirs
    }


def diameter(points: list[tuple[float, float]]) -> float:
    return max(
        (math.hypot(a[0] - b[0], a[1] - b[1]) for i, a in enumerate(points)
         for b in points[i + 1 :]),
        default=0.0,
    )  # fmt: skip


class StuckOracle:
    name = "stuck"

    def __init__(self, cfg: StuckConfig | None = None) -> None:
        self.cfg = cfg or StuckConfig()
        self.ctx: EpisodeContext | None = None
        self.level: Level | None = None
        self.window: deque[StepRecord] = deque()
        self.active_start: int | None = None
        self.active_last: StepRecord | None = None
        self.trapped_sectors: set[int] = set()

    def reset(self, episode: EpisodeContext) -> None:
        self.ctx = episode
        self.level = Level(episode.truth)
        self.window.clear()
        self.active_start = None
        self.active_last = None
        self.trapped_sectors = set()

    def _close_wedge(self) -> list[Finding]:
        if self.active_start is None or self.active_last is None:
            return []
        assert self.ctx is not None
        s = self.active_last
        duration = s.tic - self.active_start
        conf = min(1.0, duration / self.cfg.conf_saturation_tics)
        self.active_start, self.active_last = None, None
        return [Finding(self.name, "wedge", (s.x, s.y, s.z), (s.tic - duration, s.tic),
                        round(conf, 4), {"episode": self.ctx.episode,
                                         "duration_tics": duration})]  # fmt: skip

    def _failed_pushes(self, steps: list[StepRecord]) -> set[int]:
        """World directions held for >= push_tics with < push_progress_mu of movement."""
        assert self.ctx is not None
        frame = self.ctx.frame_skip
        failed: set[int] = set()
        for q in range(4):
            run: list[StepRecord] = []
            for s in [*steps, None]:
                if s is not None and q in world_quadrants(s):
                    run.append(s)
                    continue
                if run and run[-1].tic - run[0].tic + frame >= self.cfg.push_tics and diameter([(r.x, r.y) for r in run]) < self.cfg.push_progress_mu:
                    failed.add(q)
                run = []
        return failed

    def _wedged(self, step: StepRecord) -> bool:
        assert self.ctx is not None
        w = self.cfg.window_tics
        if step.tic - self.window[0].tic < w - self.ctx.frame_skip:
            return False  # window not yet full
        steps = list(self.window)
        intent = sum(bool(s.move_dirs) for s in steps) / len(steps)
        if intent <= self.cfg.intent_threshold or not step.on_ground or step.dead:
            return False
        if diameter([(s.x, s.y) for s in steps]) >= self.cfg.displacement_mu:
            return False
        return len(self._failed_pushes(steps)) >= self.cfg.min_escape_directions

    def _vertical_trap(self, step: StepRecord) -> list[Finding]:
        assert self.level is not None and self.ctx is not None
        if not step.sector_floors or not step.on_ground or step.dead:
            return []
        s = self.level.sector_at(step.x, step.y)
        if s < 0 or s in self.trapped_sectors or s >= len(step.sector_floors):
            return []
        info = self.level.sectors[s]
        neighbours = [n for n in info["neighbours"] if n < len(step.sector_floors)]
        if info["lift_trigger"] or not neighbours:
            return []
        floor = step.sector_floors[s]
        lowest_around = min(step.sector_floors[n] for n in neighbours)
        if floor >= lowest_around - self.cfg.vertical_drop_mu:
            return []
        self.trapped_sectors.add(s)
        return [Finding(self.name, "vertical_trap", (step.x, step.y, step.z),
                        (step.tic, step.tic), self.cfg.vertical_confidence,
                        {"episode": self.ctx.episode, "sector": s, "floor": floor,
                         "lowest_neighbour_floor": lowest_around,
                         "rooms": info["rooms"]})]  # fmt: skip

    def observe(self, step: StepRecord) -> list[Finding]:
        out = self._vertical_trap(step)
        self.window.append(step)
        while self.window and self.window[0].tic < step.tic - self.cfg.window_tics:
            self.window.popleft()
        if self._wedged(step):
            if self.active_start is None:
                self.active_start = self.window[0].tic
            self.active_last = step
        else:
            out += self._close_wedge()
        if step.done:
            out += self._close_wedge()
        return out

    def finalize(self) -> list[Finding]:
        return self._close_wedge()
