"""The six oracle families and a runner that feeds them episodes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from qatester.oracles.base import EpisodeContext, Finding, Oracle, OracleConfig, StepRecord
from qatester.oracles.bounds import BoundsOracle
from qatester.oracles.crash import CrashOracle
from qatester.oracles.invariants import InvariantOracle
from qatester.oracles.reach import ReachOracle
from qatester.oracles.softlock import SoftlockOracle
from qatester.oracles.stuck import StuckOracle

FAMILIES = ("crash", "stuck", "out_of_bounds", "unreachable", "invariant", "softlock")


def make_oracles(cfg: OracleConfig | None = None) -> list[Oracle]:
    """One fresh set per map per run: Families 4 and 6 accumulate across episodes."""
    c = cfg or OracleConfig()
    oracles: list[Oracle] = [
        CrashOracle(c.crash), StuckOracle(c.stuck), BoundsOracle(c.bounds),
        ReachOracle(c.reach, c.attempts), InvariantOracle(),
        SoftlockOracle(c.softlock, c.attempts),
    ]  # fmt: skip
    assert tuple(o.name for o in oracles) == FAMILIES
    return oracles


def run_oracles(
    episodes: Iterable[tuple[EpisodeContext, Sequence[StepRecord]]],
    oracles: list[Oracle] | None = None,
) -> list[Finding]:
    """Feed whole episodes through every oracle, then finalize. For recorded traces and
    tests; training streams records through `observe` instead."""
    oracles = oracles if oracles is not None else make_oracles()
    findings: list[Finding] = []
    for ctx, records in episodes:
        for o in oracles:
            o.reset(ctx)
        for r in records:
            for o in oracles:
                findings += o.observe(r)
    for o in oracles:
        findings += o.finalize()
    return findings
