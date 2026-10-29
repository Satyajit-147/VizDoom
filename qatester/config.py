"""Run configuration (PLAN Appendix B), loaded from YAML and checked strictly: an unknown
key is an error, so a typo cannot silently fall back to a default.

Paths are relative to the working directory (the repo root locally and on Kaggle);
`scripts/train.py` can override the output folders from the command line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from qatester.env.reward import RewardConfig
from qatester.oracles.base import OracleConfig


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StageConfig(_Model):
    until: int
    maps: list[str] = Field(default_factory=list)
    weights: list[float] = Field(default_factory=list)
    generated_weight: float = 0.0  # total weight spread evenly over the generated maps

    @model_validator(mode="after")
    def _check(self) -> StageConfig:
        if len(self.maps) != len(self.weights):
            raise ValueError("a stage needs one weight per map")
        if sum(self.weights) + self.generated_weight <= 0:
            raise ValueError("a stage needs some positive weight")
        return self


class GeneratedConfig(_Model):
    split: Literal["train", "test"] = "train"
    count: int = 0
    start: int = 0


class MapsConfig(_Model):
    specs: Path = Path("maps")
    build: Path = Path("build")
    variant: Literal["buggy", "fixed"] = "buggy"
    generated: GeneratedConfig = Field(default_factory=GeneratedConfig)
    curriculum: list[StageConfig]

    @model_validator(mode="after")
    def _check(self) -> MapsConfig:
        untils = [s.until for s in self.curriculum]
        if not untils or untils != sorted(untils):
            raise ValueError("curriculum stages must be listed in increasing `until`")
        if any(s.generated_weight > 0 for s in self.curriculum) and self.generated.count == 0:
            raise ValueError("generated_weight > 0 needs maps.generated.count > 0")
        return self


class ValidityConfig(_Model):
    precondition: float = 1.0
    goal: float = 5.0
    death: float = -5.0
    time: float = -0.0005
    shaping: float = 1.0  # potential-based, per 1000 mu of route to the goal (D35)


class AdversarialConfig(_Model):
    cell_novelty: float = 0.3
    state_novelty: float = 0.5
    entity_first_use: float = 1.0
    portal_first_cross: float = 1.0
    oracle_cluster: float = 3.0
    bugbook_hit: float = 1.5


class RewardSection(_Model):
    alpha: float = 0.5
    validity: ValidityConfig = Field(default_factory=ValidityConfig)
    adversarial: AdversarialConfig = Field(default_factory=AdversarialConfig)

    def to_reward_config(self) -> RewardConfig:
        v, a = self.validity, self.adversarial
        return RewardConfig(
            alpha=self.alpha, progress=v.precondition, goal=v.goal, death=v.death, time=v.time,
            shaping=v.shaping,
            cell_novelty=a.cell_novelty, state_novelty=a.state_novelty,
            entity=a.entity_first_use, portal=a.portal_first_cross, oracle=a.oracle_cluster,
            bugbook=a.bugbook_hit,
        )  # fmt: skip


class PPOConfig(_Model):
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 4
    learning_rate: float = 2.5e-4
    lr_schedule: Literal["linear", "constant"] = "linear"
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_end: float = 0.1
    ent_coef: float = 0.01
    ent_coef_end: float = 0.003
    ent_decay_after: int = 500_000
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lstm_hidden_size: int = 256


class EvalConfig(_Model):
    every: int = 100_000
    episodes: int = 20
    maps: list[str] = Field(default_factory=lambda: ["MAP01"])
    variant: Literal["buggy", "fixed"] = "fixed"
    deterministic: bool = False


class ReportConfig(_Model):
    dbscan_eps_mu: float = 96.0
    dbscan_min_samples: int = 3
    severity_weights: dict[str, float] = Field(
        default_factory=lambda: {"oracle": 0.5, "blocking": 0.3, "frequency": 0.2})
    calibration: Path | None = None  # JSON with a fitted temperature (M9)


class RunConfig(_Model):
    run_name: str
    agent: Literal["random", "scripted", "ppo"]
    total_steps: int
    n_envs: int = 2
    seed: int = 0
    device: str = "auto"
    runs_dir: Path = Path("runs")
    results_dir: Path = Path("results")
    ckpt_every: int = 100_000
    max_session_hours: float = 11.5
    metrics_every: int = 10_000
    trace_policy: Literal["all", "findings", "none"] = "findings"
    maps: MapsConfig
    reward: RewardSection = Field(default_factory=RewardSection)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    oracles: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> RunConfig:
        self.oracle_config()  # unknown oracle keys are an error
        return self

    def oracle_config(self) -> OracleConfig:
        return OracleConfig.from_dict(self.oracles)


def load_config(path: Path, overrides: dict[str, Any] | None = None) -> RunConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data.update(overrides or {})
    return RunConfig.model_validate(data)
