"""Family 3: out of bounds and geometry violation (PLAN: Lu et al.'s nav-mesh test with
exact rooms).

1. outside_level: (x, y) is not inside any room rectangle expanded by the player radius.
   Confidence 1.0. One finding per excursion, reported when it ends, positioned at the
   exit point, with the deepest distance reached.
2. z_clip: on the ground but below the live floor or above the live ceiling of the
   intended sector under the player, beyond a tolerance. Confidence 1.0.
3. wall_traversal: between consecutive records the straight path crosses an intended
   wall *and* the move is faster than running allows. Moves starting on a teleport pad
   are excused (teleport problems belong to Family 6). Confidence 1.0.
"""

from __future__ import annotations

import math

from qatester.oracles.base import BoundsConfig, EpisodeContext, Finding, StepRecord
from qatester.oracles.level import Level, segments_cross


class BoundsOracle:
    name = "out_of_bounds"

    def __init__(self, cfg: BoundsConfig | None = None) -> None:
        self.cfg = cfg or BoundsConfig()
        self.ctx: EpisodeContext | None = None
        self.level: Level | None = None
        self.prev: StepRecord | None = None
        self.excursion: tuple[StepRecord, float, StepRecord] | None = None  # exit, depth, last
        self.clipping = False

    def reset(self, episode: EpisodeContext) -> None:
        self.ctx = episode
        self.level = Level(episode.truth)
        self.prev = None
        self.excursion = None
        self.clipping = False

    def _close_excursion(self) -> list[Finding]:
        if self.excursion is None:
            return []
        assert self.ctx is not None
        exit_step, depth, last = self.excursion
        self.excursion = None
        return [Finding(self.name, "outside_level", (exit_step.x, exit_step.y, exit_step.z),
                        (exit_step.tic, last.tic), 1.0,
                        {"episode": self.ctx.episode, "exit_point": [exit_step.x, exit_step.y],
                         "depth_beyond_mu": round(depth, 1)})]  # fmt: skip

    def observe(self, step: StepRecord) -> list[Finding]:
        assert self.level is not None and self.ctx is not None
        lv, cfg = self.level, self.cfg
        out: list[Finding] = []

        # 1. outside every room
        if not lv.inside_level(step.x, step.y, cfg.player_radius_mu):
            depth = lv.depth_outside(step.x, step.y)
            if self.excursion is None:
                self.excursion = (step, depth, step)
            else:
                exit_step, deepest, _ = self.excursion
                self.excursion = (exit_step, max(deepest, depth), step)
        else:
            out += self._close_excursion()

        # 2. clipped into the floor or ceiling of the sector under the player
        s = lv.sector_at(step.x, step.y)
        clipped = False
        if step.on_ground and not step.dead and 0 <= s < len(step.sector_floors):
            floor, ceiling = step.sector_floors[s], step.sector_ceilings[s]
            clipped = step.z < floor - cfg.z_tolerance_mu or (
                step.z + cfg.player_height_mu > ceiling + cfg.z_tolerance_mu
            )
            if clipped and not self.clipping:
                out.append(Finding(self.name, "z_clip", (step.x, step.y, step.z),
                                   (step.tic, step.tic), 1.0,
                                   {"episode": self.ctx.episode, "sector": s,
                                    "floor": floor, "ceiling": ceiling}))  # fmt: skip
        self.clipping = clipped

        # 3. through a wall faster than running allows
        p = self.prev
        if p is not None and step.tic > p.tic:
            moved = math.hypot(step.x - p.x, step.y - p.y)
            limit = cfg.max_speed_mu_per_tic * (step.tic - p.tic) + cfg.player_radius_mu
            if moved > limit and not lv.on_teleport_pad(p.x, p.y, 2 * cfg.player_radius_mu):
                path = (p.x, p.y, step.x, step.y)
                crossed = [w for w in lv.walls if segments_cross(path, w)]
                if crossed:
                    out.append(Finding(self.name, "wall_traversal", (step.x, step.y, step.z),
                                       (p.tic, step.tic), 1.0,
                                       {"episode": self.ctx.episode, "from": [p.x, p.y],
                                        "moved_mu": round(moved, 1),
                                        "walls_crossed": len(crossed)}))  # fmt: skip
        self.prev = step
        if step.done:
            out += self._close_excursion()
        return out

    def finalize(self) -> list[Finding]:
        return self._close_excursion()
