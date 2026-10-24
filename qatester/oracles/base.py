"""Oracle interface, per-step record, findings and frozen thresholds (PLAN: The oracle
layer, Common interface; Appendix B `oracles:`).

Oracles are pure functions of the trace plus the intended level from groundtruth.json.
They never call into ViZDoom (enforced by tests/test_no_groundtruth_leak.py) and never
see the defect list (they get `GroundTruth.intent_only()`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol

from qatester.mapgen.groundtruth import GroundTruth

MOVE_DIRS = ("forward", "back", "left", "right")


@dataclass(frozen=True)
class StepRecord:
    """The single source of per-step truth. One record per env step (every `frame_skip`
    tics in training, every tic in replay). Positions in map units, angle in degrees
    (0 = east, counter-clockwise), `sector_*` are the engine's live heights in WAD
    sector order."""

    tic: int
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    angle: float = 0.0
    on_ground: bool = True
    dead: bool = False
    health: float = 100.0
    armour: float = 0.0
    ammo: float = 50.0
    keys: frozenset[str] = frozenset()  # key types held: key_red / key_blue / key_yellow
    killcount: int = 0
    action: int = 0
    move_dirs: frozenset[str] = frozenset()  # commanded movement, subset of MOVE_DIRS
    use: bool = False
    jump: bool = False
    sector_floors: tuple[float, ...] = ()
    sector_ceilings: tuple[float, ...] = ()
    notifications: tuple[str, ...] = ()
    done: bool = False  # last record of the episode
    truncated: bool = False  # the episode ended on its time limit
    error: str | None = None  # engine exception raised at this step
    step_wall_s: float = 0.0  # wall-clock seconds the step took


@dataclass(frozen=True)
class Finding:
    """PLAN normative fields. `evidence` always carries the episode index."""

    oracle: str
    kind: str
    position: tuple[float, float, float]
    tic_range: tuple[int, int]
    confidence: float
    evidence: dict[str, Any]
    demo_path: Path | None = None


@dataclass(frozen=True)
class EpisodeContext:
    map: str
    episode: int
    seed: int
    truth: GroundTruth  # intent-only view
    frame_skip: int = 4


class Oracle(Protocol):
    name: str

    def reset(self, episode: EpisodeContext) -> None: ...

    def observe(self, step: StepRecord) -> list[Finding]: ...

    def finalize(self) -> list[Finding]: ...


# --- thresholds: frozen at the end of M4 (docs/decisions.md) -------------------------

@dataclass(frozen=True)
class CrashConfig:
    step_timeout_s: float = 30.0
    stall_steps: int = 3


@dataclass(frozen=True)
class StuckConfig:
    window_tics: int = 70
    intent_threshold: float = 0.8
    displacement_mu: float = 24.0
    min_escape_directions: int = 3
    push_tics: int = 16  # an escape attempt: one direction held this long...
    push_progress_mu: float = 8.0  # ...while moving less than this
    conf_saturation_tics: int = 350
    vertical_drop_mu: float = 24.0
    vertical_confidence: float = 0.9


@dataclass(frozen=True)
class BoundsConfig:
    player_radius_mu: float = 16.0
    z_tolerance_mu: float = 8.0
    player_height_mu: float = 56.0
    max_speed_mu_per_tic: float = 32.0


@dataclass(frozen=True)
class AttemptConfig:
    """What counts as a proper, failed attempt to use an exit (Families 4 and 6)."""

    use_range_mu: float = 64.0  # Doom's USE reach
    window_tics: int = 35  # a door or lift must start moving within this
    throat_mu: float = 32.0  # open portal: how close to the opening a push counts
    throat_push_tics: int = 16  # ...and for how long without progress
    stall_mu_per_tic: float = 1.0
    face_cone_deg: float = 45.0
    player_radius_mu: float = 16.0


@dataclass(frozen=True)
class ReachConfig:
    use_attempts_for_broken_trigger: int = 5
    confidence_graph: float = 0.95
    confidence_trigger: float = 0.85


@dataclass(frozen=True)
class SoftlockConfig:
    tail_fraction: float = 0.4
    max_lag_tics: int = 70
    min_loop_hops: int = 4
    teleport_jump_mu: float = 200.0
    confidence: float = 0.8
    repeat_confidence: float = 0.95
    loop_confidence: float = 0.9


@dataclass(frozen=True)
class OracleConfig:
    crash: CrashConfig = field(default_factory=CrashConfig)
    stuck: StuckConfig = field(default_factory=StuckConfig)
    bounds: BoundsConfig = field(default_factory=BoundsConfig)
    reach: ReachConfig = field(default_factory=ReachConfig)
    softlock: SoftlockConfig = field(default_factory=SoftlockConfig)
    attempts: AttemptConfig = field(default_factory=AttemptConfig)

    @classmethod
    def from_dict(cls, d: Mapping[str, Mapping[str, Any]]) -> OracleConfig:
        """Build from the `oracles:` block of a run config; unknown keys are an error."""
        parts: dict[str, Any] = {}
        for f in fields(cls):
            sub = d.get(f.name, {})
            sub_cls = type(getattr(cls(), f.name))
            known = {g.name for g in fields(sub_cls)}
            unknown = set(sub) - known
            if unknown:
                raise ValueError(f"oracles.{f.name}: unknown keys {sorted(unknown)}")
            parts[f.name] = sub_cls(**sub)
        unknown_top = set(d) - {f.name for f in fields(cls)}
        if unknown_top:
            raise ValueError(f"oracles: unknown families {sorted(unknown_top)}")
        return cls(**parts)
