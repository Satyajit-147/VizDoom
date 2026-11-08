"""The detect -> fix -> re-test loop with a non-learned Fixer (docs/FIXER_PLAN.md).

One level at a time:

1. QA plays the level and reports bugs (`qa(cfg_path, seed)`: any tester; the script uses
   Go-Explore + the 6 detectors). Only the reports reach this module, never the answer key.
2. For each report: build the edit menu (menu.py) and try candidate fixes nearest-first.
   Edits that belong together form one attempt: every `blocking` flag on the mouth
   between the same two sectors, every copy of the same trigger line, every teleport
   destination moved to its room centre facing the same way.
3. Each attempt is rewritten as a level and re-tested by QA with the same seed. Reward
   (FIXER_PLAN section 4): +1 if the target report is gone, -1 if the level is damaged
   (cell coverage below 80% of before), -0.5 per new report, -0.05 per edit. An attempt is
   kept only when the target is gone, nothing is damaged and nothing new is reported;
   otherwise it is undone and the next one is tried.
4. Stop when no report is left, or every remaining one has used its attempts.

`fix_until_quiet` repeats this on the SAME level (same planted bugs, nothing new
planted): each round QA plays the level as fixed so far with a new seed (and, from the
script, more steps), so it can spot bugs the earlier rounds missed, and a report that
was already tried continues further down its candidate list instead of repeating the
same attempts. It stops after `quiet` rounds in a row in which QA reported nothing the
Fixer could still try, or after `max_rounds`. "All bugs found" is never known inside the
loop (that would need the answer key); QA staying quiet is the stopping rule.

This Fixer is the plan's nearest-first baseline, plus QA's verdict to pick among
candidates; the learned Fixer (F4-F5) replaces the candidate order with a policy.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from qatester.fixer.edits import (
    CopySpecial,
    Edit,
    EditError,
    MoveThing,
    SetFlag,
    apply,
    write_level,
)
from qatester.fixer.menu import Report, build_menu
from qatester.mapgen import udmf

MATCH_MU = 128.0  # two reports are the same bug: same detector family, this close (D25)
COVERAGE_KEEP = 0.8  # below this share of the old cell coverage, a fix damaged the level
MAX_ATTEMPTS = 6  # candidate fixes tried per report


@dataclass(frozen=True)
class QAResult:
    reports: tuple[Report, ...]
    coverage: float  # cell coverage, 0..1


QA = Callable[[Path, int], QAResult]


@dataclass(frozen=True)
class Attempt:
    target: Report
    edits: tuple[Edit, ...]
    reward: float
    gone: bool
    damaged: bool
    new_reports: int
    accepted: bool


@dataclass
class LevelResult:
    initial: QAResult
    final: QAResult
    final_level: udmf.TextMap
    final_cfg: Path
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def fixes(self) -> list[Attempt]:
        return [a for a in self.attempts if a.accepted]


def same_bug(a: Report, b: Report) -> bool:
    return a.oracle == b.oracle and math.hypot(a.position[0] - b.position[0],
                                               a.position[1] - b.position[1]) <= MATCH_MU  # fmt: skip


def reward(target: Report, before: QAResult, after: QAResult,
           n_edits: int) -> tuple[float, bool, bool, int]:  # fmt: skip
    """(reward, target gone, level damaged, new reports) for one re-test."""
    gone = not any(same_bug(target, r) for r in after.reports)
    damaged = after.coverage < COVERAGE_KEEP * before.coverage
    new = sum(not any(same_bug(r, b) for b in before.reports) for r in after.reports)
    r = (1.0 if gone else 0.0) - (1.0 if damaged else 0.0) - 0.5 * new - 0.05 * n_edits
    return r, gone, damaged, new


def candidate_fixes(tm: udmf.TextMap, target: Report) -> list[tuple[Edit, ...]]:
    """The menu's edits grouped into attempts, in menu (nearest-first) order."""

    def pair(i: int) -> frozenset[int | None]:
        ld = tm.linedefs[i]
        return frozenset({tm.sidedefs[ld.sidefront].sector,
                          tm.sidedefs[ld.sideback].sector if ld.sideback is not None else None})  # fmt: skip

    groups: dict[object, list[Edit]] = {}
    for ent in build_menu(tm, target).entities:
        for e in ent.edits:
            key: object
            if isinstance(e, SetFlag):
                key = ("flag", e.flag, e.on, pair(e.linedef))
            elif isinstance(e, CopySpecial):
                key = ("copy", e.source)
            elif isinstance(e, MoveThing):
                key = ("move", e.angle)
            else:
                key = ("one", e)
            if e not in groups.setdefault(key, []):
                groups[key].append(e)
    out = []
    for edits in groups.values():
        try:
            apply(tm, edits)
        except EditError:
            continue  # edits that clash with each other
        out.append(tuple(edits))
    return out


@dataclass
class Effort:
    """How far down its candidate list a report has been tried (kept across rounds)."""

    report: Report
    used: int = 0


def fix_level(tm: udmf.TextMap, map_lump: str, cfg_text: str, qa: QA, work: Path,
              seed: int = 0, max_attempts: int = MAX_ATTEMPTS,
              log: Callable[[str], None] = print,
              effort: list[Effort] | None = None) -> LevelResult:  # fmt: skip
    """Run the loop on one level until no report is left or all have been tried.

    `effort` (from earlier rounds on this level) makes a report seen before continue
    with its next untried candidates."""
    effort = [] if effort is None else effort
    cfg = write_level(tm, map_lump, cfg_text, work, "level0")
    before = qa(cfg, seed)
    result = LevelResult(before, before, tm, cfg)
    log(f"QA: {len(before.reports)} reports, coverage {before.coverage:.0%}")
    tried: list[Report] = []
    queue = list(before.reports)
    n = 0
    while queue:
        target = queue.pop(0)
        tried.append(target)
        eff = next((e for e in effort if same_bug(e.report, target)), None)
        if eff is None:
            eff = Effort(target)
            effort.append(eff)
        cands = candidate_fixes(result.final_level, target)[eff.used:eff.used + max_attempts]
        log(f"report {target.oracle}/{target.kind} at ({target.position[0]:.0f}, "
            f"{target.position[1]:.0f})"
            + (f", candidates {eff.used + 1}-{eff.used + len(cands)}" if eff.used else "")
            + ("" if cands else ": no untried candidates left"))  # fmt: skip
        for edits in cands:
            eff.used += 1
            n += 1
            level = apply(result.final_level, list(edits))
            cand = write_level(level, map_lump, cfg_text, work, f"try{n}")
            after = qa(cand, seed)
            r, gone, damaged, new = reward(target, before, after, len(edits))
            ok = gone and not damaged and new == 0
            result.attempts.append(Attempt(target, edits, r, gone, damaged, new, ok))
            log(f"  try {n}: {len(edits)} edit(s) -> reward {r:+.2f} "
                f"(gone={gone}, damaged={damaged}, new={new}) {'KEPT' if ok else 'undone'}")  # fmt: skip
            if ok:
                result.final_level, result.final_cfg, before = level, cand, after
                result.final = after
                queue = [q for q in after.reports if not any(same_bug(q, t) for t in tried)]
                break
    return result


def fix_until_quiet(tm: udmf.TextMap, map_lump: str, cfg_text: str,
                    qa_for_round: Callable[[int], QA], work: Path, seed: int = 0,
                    max_attempts: int = MAX_ATTEMPTS, quiet: int = 2, max_rounds: int = 5,
                    log: Callable[[str], None] = print) -> tuple[LevelResult, list[LevelResult]]:  # fmt: skip
    """Rounds of `fix_level` on the same level until QA stays quiet (module docstring).

    Round r uses QA `qa_for_round(r)` with seed `seed + r`. A round is quiet when it made
    no attempt: QA reported nothing, or only reports with no untried candidate left.
    Returns the whole run as one LevelResult (first QA run, last level) and each round."""
    effort: list[Effort] = []
    rounds: list[LevelResult] = []
    level, calm = tm, 0
    for r in range(max_rounds):
        log(f"--- round {r + 1}")
        res = fix_level(level, map_lump, cfg_text, qa_for_round(r), work / f"round{r + 1}",
                        seed + r, max_attempts, log, effort)  # fmt: skip
        rounds.append(res)
        level = res.final_level
        calm = 0 if res.attempts else calm + 1
        if calm >= quiet:
            log(f"quiet for {quiet} round(s): stop")
            break
    else:
        log(f"stopped after {max_rounds} rounds")
    total = LevelResult(rounds[0].initial, rounds[-1].final, rounds[-1].final_level,
                        rounds[-1].final_cfg, [a for x in rounds for a in x.attempts])  # fmt: skip
    return total, rounds
