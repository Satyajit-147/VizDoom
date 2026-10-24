"""Family 1: crash and hang (PLAN: Wuji's crash oracle, unchanged). Confidence 1.0.

- crash: the engine raised at this step (the env supervisor records it on the trace)
- hang: a step took longer than `step_timeout_s` wall-clock seconds
- tic_stall: game time did not advance for `stall_steps` consecutive steps while running
"""

from __future__ import annotations

from qatester.oracles.base import CrashConfig, EpisodeContext, Finding, StepRecord


class CrashOracle:
    name = "crash"

    def __init__(self, cfg: CrashConfig | None = None) -> None:
        self.cfg = cfg or CrashConfig()
        self.ctx: EpisodeContext | None = None
        self.last_tic: int | None = None
        self.stalled = 0
        self.reported: set[str] = set()

    def reset(self, episode: EpisodeContext) -> None:
        self.ctx = episode
        self.last_tic = None
        self.stalled = 0
        self.reported = set()

    def _finding(self, kind: str, s: StepRecord, evidence: dict[str, object]) -> list[Finding]:
        if kind in self.reported:
            return []
        self.reported.add(kind)
        assert self.ctx is not None
        return [Finding(self.name, kind, (s.x, s.y, s.z), (s.tic, s.tic), 1.0,
                        {"episode": self.ctx.episode, **evidence})]  # fmt: skip

    def observe(self, step: StepRecord) -> list[Finding]:
        out: list[Finding] = []
        if step.error is not None:
            out += self._finding("crash", step, {"error": step.error})
        if step.step_wall_s > self.cfg.step_timeout_s:
            out += self._finding("hang", step, {"step_wall_s": step.step_wall_s})
        if self.last_tic is not None and step.tic <= self.last_tic and not step.done:
            self.stalled += 1
            if self.stalled >= self.cfg.stall_steps:
                out += self._finding("tic_stall", step, {"tic": step.tic,
                                                         "steps": self.stalled})  # fmt: skip
        else:
            self.stalled = 0
        self.last_tic = step.tic
        return out

    def finalize(self) -> list[Finding]:
        return []
