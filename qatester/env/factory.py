"""Everything a run needs to make environments: build the maps, set up the curriculum and
the shared coverage counts, and construct one curriculum env per worker.

It lives in the env package so agent code never touches ground truth: runners pass
paths and configs, and the env loads the intended level itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from qatester.config import MapsConfig, RunConfig
from qatester.coverage import CoverageTracker
from qatester.env.curriculum import CurriculumEnv, MapEntry, Stage
from qatester.env.vizdoom_env import QADoomEnv
from qatester.mapgen.cli import artefact_name, build
from qatester.mapgen.groundtruth import GroundTruth
from qatester.mapgen.procgen import seeds, write_pack
from qatester.mapgen.spec import load_spec

PACK_SIZE = 32  # maps per generated WAD


def _entries(spec_dir: Path, out: Path, variant: str, prefix: str | None) -> dict[str, MapEntry]:
    result = build(spec_dir, out, variant)  # type: ignore[arg-type]
    entries = {}
    for path in sorted(spec_dir.glob("*.yaml")):
        spec = load_spec(path)
        lump = spec.map_lump
        name = spec.name if prefix else lump
        entries[name] = MapEntry(name, result.cfg_paths[lump], result.groundtruth_paths[lump],
                                 path)  # fmt: skip
    return entries


def prepare_maps(cfg: MapsConfig, out: Path, variant: str | None = None) -> dict[str, MapEntry]:
    """Build the corpus, and the generated maps if any, into `out`. Corpus maps are named
    by lump (MAP01..); generated maps by spec name (GEN<seed>)."""
    variant = variant or cfg.variant
    entries = _entries(cfg.specs, out / "corpus", variant, prefix=None)
    g = cfg.generated
    all_seeds = seeds(g.split, g.count, g.start) if g.count else []
    for k in range(0, len(all_seeds), PACK_SIZE):
        pack = out / f"gen_{g.split}_{k // PACK_SIZE:03d}"
        write_pack(pack / "specs", all_seeds[k:k + PACK_SIZE])
        entries.update(_entries(pack / "specs", pack, variant, prefix="GEN"))
    return entries


def stages(cfg: MapsConfig, entries: Mapping[str, MapEntry]) -> list[Stage]:
    gen = sorted(n for n in entries if n.startswith("GEN"))
    out = []
    for st in cfg.curriculum:
        weights = dict(zip(st.maps, st.weights, strict=True))
        if st.generated_weight > 0:
            for n in gen:
                weights[n] = weights.get(n, 0.0) + st.generated_weight / len(gen)
        out.append(Stage(st.until, weights))
    return out


def make_coverage(entries: Mapping[str, MapEntry], names: set[str]) -> dict[str, CoverageTracker]:
    """One shared-memory tracker per map; the creating process owns and frees them."""
    return {n: CoverageTracker(GroundTruth.load(entries[n].groundtruth_path))
            for n in sorted(names)}  # fmt: skip


def map_names(st: list[Stage]) -> set[str]:
    return {n for s in st for n in s.weights}


def make_env(cfg: RunConfig, entries: Mapping[str, MapEntry], st: list[Stage],
             coverage: Mapping[str, CoverageTracker] | None, queue: Any,
             rank: int, record_dir: Path | None, initial_steps: int = 0) -> CurriculumEnv:  # fmt: skip
    """Worker `rank`'s env. Seeds derive from the run seed, so workers differ."""
    seed = cfg.seed * 1000 + rank
    rng = np.random.default_rng([cfg.seed, rank])
    first = min(map_names(st))
    e = entries[first]
    env = QADoomEnv(
        e.cfg_path, GroundTruth.load(e.groundtruth_path),
        coverage.get(first) if coverage else None, queue, seed, record_dir,
        reward_cfg=cfg.reward.to_reward_config(), oracle_cfg=cfg.oracle_config(),
        trace_policy=cfg.trace_policy, map_name=first,
    )  # fmt: skip
    return CurriculumEnv(env, entries, st, coverage, cfg.n_envs, rng, initial_steps)


def snapshot_coverage(entries: Mapping[str, MapEntry], name: str,
                      counts: CoverageTracker | Path | None) -> CoverageTracker | None:
    """A private copy of a map's visit counts for evaluation episodes: the policy reads
    novelty from them (observation slots 14-15), so evaluating without them shows it
    inputs it never saw in training (D39). The copy keeps evaluation from changing the
    training counts. `counts` is a live tracker or a checkpoint's saved array."""
    if counts is None:
        return None
    snap = CoverageTracker(GroundTruth.load(entries[name].groundtruth_path))
    if isinstance(counts, CoverageTracker):
        snap._flat[:] = counts._flat
    else:
        snap.restore(counts)
    return snap


def eval_env(cfg: RunConfig, entries: Mapping[str, MapEntry], name: str, seed: int,
             record_dir: Path | None = None,
             coverage: CoverageTracker | None = None) -> QADoomEnv:  # fmt: skip
    """A plain env on one map for evaluation episodes. Pass the training counts (see
    `snapshot_coverage`): without them the novelty inputs differ from training."""
    e = entries[name]
    return QADoomEnv(e.cfg_path, GroundTruth.load(e.groundtruth_path), coverage, None, seed,
                     record_dir, reward_cfg=cfg.reward.to_reward_config(),
                     oracle_cfg=cfg.oracle_config(), trace_policy="none", map_name=name)  # fmt: skip


def corpus_artefact(build_dir: Path, lump: str, variant: str, suffix: str) -> Path:
    return build_dir / artefact_name(lump, variant, suffix)  # type: ignore[arg-type]
