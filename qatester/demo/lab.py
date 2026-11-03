"""QA Lab: the logic behind the demo app (`scripts/qa_lab.py`), separate from its UI.

1. `make_level` builds a level to test: a generated map with the bug classes a person
   picks (or one of the eight hand-made maps), compiled into its buggy WAD with its
   answer key.
2. `Session` lets a tester play it step by step (random, scripted, Go-Explore, or a
   trained PPO checkpoint), keeping the colour view, the trail and every detection as it happens.
3. `Session.finish` closes the run, writes the usual report, and scores it against
   the answer key: which planted bugs were found, which were missed, any false alarms.

The tester itself never sees the answer key; only the scoring step reads it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from qatester.agents.goexplore import GoExplore
from qatester.agents.random_agent import RandomAgent
from qatester.agents.scripted_agent import WallFollower
from qatester.coverage import CoverageTracker
from qatester.env.vizdoom_env import QADoomEnv, wad_of
from qatester.mapgen.cli import build
from qatester.mapgen.geometry import compile_map
from qatester.mapgen.groundtruth import GroundTruth
from qatester.mapgen.preview import render_svg
from qatester.mapgen.procgen import DEFECT_KINDS, ProcgenConfig, generate, spec_yaml
from qatester.mapgen.spec import MapSpec, load_spec
from qatester.report.emit import COLOUR, build_report, raw_finding
from qatester.report.scoring import score

BUG_LABELS = {
    "missing_blocking": "Leaky wall (you can walk out of the level)",
    "geometry_trap": "Trap pit (you can fall in, never climb out)",
    "wrong_tag": "Dead switch (opens nothing)",
    "wrong_special": "Broken lock (door opens without its key)",
    "one_way_lift": "One-way lift (no way back up)",
    "teleport_loop": "Teleport loop (bounces you back and forth)",
    "bad_pickup": "Bad pickup (health above the maximum)",
}
assert set(BUG_LABELS) == set(DEFECT_KINDS)
TESTERS = ("random", "scripted", "goexplore", "ppo")


@dataclass
class LabLevel:
    name: str
    spec: MapSpec
    cfg_path: Path
    truth: GroundTruth  # full answer key; only `Session.finish` reads its defects

    @property
    def defects(self) -> list[dict[str, Any]]:
        return self.truth.defects


def make_level(out: Path, seed: int | None = None, kinds: Sequence[str] = (),
               corpus_map: str | None = None, maps_dir: Path = Path("maps"),
               tries: int = 40) -> LabLevel:  # fmt: skip
    """A generated map from `seed` with exactly the bug classes `kinds`, or a corpus map
    (`corpus_map`, e.g. "MAP03") with its own planted bugs. If a seed cannot host the
    chosen bugs, the next seeds are tried."""
    out.mkdir(parents=True, exist_ok=True)
    if corpus_map is not None:
        result = build(maps_dir, out / "corpus", "buggy")
        path = next(p for p in sorted(maps_dir.glob("*.yaml"))
                    if load_spec(p).map_lump == corpus_map)  # fmt: skip
        return LabLevel(corpus_map, load_spec(path), result.cfg_paths[corpus_map],
                        GroundTruth.load(result.groundtruth_paths[corpus_map]))  # fmt: skip
    if seed is None:
        raise ValueError("give a seed or a corpus map")
    cfg = ProcgenConfig() if len(kinds) <= 4 else ProcgenConfig(grid=5, min_rooms=10,
                                                                 max_rooms=14)  # fmt: skip
    last: Exception | None = None
    for s in range(seed, seed + tries):
        try:
            spec = generate(s, cfg, slot=1, kinds=list(kinds))
            break
        except Exception as e:  # noqa: BLE001 - try the next seed
            last = e
    else:
        raise RuntimeError(f"no map from seeds {seed}..{seed + tries - 1} can host "
                           f"{list(kinds)}: {last}")  # fmt: skip
    specs = out / f"gen_{spec.seed}" / "specs"
    specs.mkdir(parents=True, exist_ok=True)
    for old in specs.glob("*.yaml"):
        old.unlink()
    (specs / "gen.yaml").write_text(spec_yaml(spec), encoding="utf-8")
    result = build(specs, specs.parent / "build", "buggy")
    return LabLevel(spec.name, spec, result.cfg_paths["MAP01"],
                    GroundTruth.load(result.groundtruth_paths["MAP01"]))  # fmt: skip


def preview_svg(level: LabLevel, trail: Sequence[tuple[float, float]] = (),
                findings: Sequence[dict[str, Any]] = (), show_bugs: bool = False) -> str:
    """Top-down map: the intended level, the tester's trail, its detections, and (after
    the run, if asked) the planted bugs."""
    spec = level.spec if show_bugs else level.spec.without_defects()
    markers = [(f["position"][0], f["position"][1], f.get("label", f["oracle"]),
                COLOUR[f["oracle"]]) for f in findings]  # fmt: skip
    return render_svg(compile_map(spec), markers, trail=trail)


@dataclass
class Session:
    level: LabLevel
    tester: str = "random"
    checkpoint: Path | None = None  # folder with model.zip (+ coverage_<map>.npy)
    record_dir: Path = Path("/tmp/qa_lab")
    seed: int = 0
    env: QADoomEnv = field(init=False)
    trail: list[tuple[float, float]] = field(default_factory=list, init=False)
    findings: list[dict[str, Any]] = field(default_factory=list, init=False)
    steps: int = field(default=0, init=False)
    episodes: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.tester not in TESTERS:
            raise ValueError(f"tester must be one of {TESTERS}")
        self.coverage = CoverageTracker(self.level.truth)
        # Go-Explore records no demos: ViZDoom freezes the clock after a load otherwise (D46)
        self.env = QADoomEnv(self.level.cfg_path, self.level.truth, self.coverage, None,
                             self.seed, None if self.tester == "goexplore" else self.record_dir,
                             display=True, map_name=self.level.name)  # fmt: skip
        rng = np.random.default_rng(self.seed)
        self.model: Any = None
        self.agent: Any = None
        self.goexplore: GoExplore | None = None
        if self.tester == "goexplore":
            self.goexplore = GoExplore()
            self._walk = self.goexplore.walk(self.env, rng)
            self._iteration = 0
            return
        if self.tester == "random":
            self.agent = RandomAgent(rng)
        elif self.tester == "scripted":
            self.agent = WallFollower(rng)
        else:
            if self.checkpoint is None:
                raise ValueError("the trained tester needs a checkpoint folder")
            from sb3_contrib import RecurrentPPO  # needs torch: only for this tester

            self.model = RecurrentPPO.load(self.checkpoint / "model.zip", device="cpu")
        self._new_episode()

    def _new_episode(self) -> None:
        self.obs, _ = self.env.reset()
        self.lstm: Any = None
        self.start = np.ones((1,), dtype=bool)
        if self.agent is not None:
            self.agent.reset()
        self.trail.append((float("nan"), float("nan")))  # a gap between episodes

    def _act(self) -> int:
        if self.agent is not None:
            return int(self.agent.act(self.obs))
        action, self.lstm = self.model.predict(self.obs, state=self.lstm,
                                               episode_start=self.start)  # fmt: skip
        self.start[:] = False
        return int(action)

    def step(self, n: int = 1) -> list[dict[str, Any]]:
        """Play `n` steps; returns the detections made during them."""
        if self.goexplore is not None:
            return self._step_goexplore(n)
        new: list[dict[str, Any]] = []
        for _ in range(n):
            self.obs, _, term, trunc, info = self.env.step(np.int64(self._act()))
            self.steps += 1
            self.trail.append((info["pos"][0], info["pos"][1]))
            for f in info["findings"]:
                new.append(raw_finding(f, self.level.name, self.steps, self.seed))
            if term or trunc:
                self.episodes += 1
                self._new_episode()
        self.findings += new
        return new

    def _step_goexplore(self, n: int) -> list[dict[str, Any]]:
        assert self.goexplore is not None
        new: list[dict[str, Any]] = []
        for _ in range(n):
            info, over = next(self._walk)
            if self.goexplore.iterations != self._iteration:  # it jumped to a saved state
                self._iteration = self.goexplore.iterations
                self.trail.append((float("nan"), float("nan")))
            self.steps += 1
            self.trail.append((info["pos"][0], info["pos"][1]))
            new += [raw_finding(f, self.level.name, self.steps, self.seed)
                    for f in info["findings"]]  # fmt: skip
            self.episodes += over
        self.findings += new
        return new

    @property
    def saved_states(self) -> int:
        """Go-Explore's archive size (0 for the other testers)."""
        return sum(len(a) for a in self.goexplore.archives.values()) if self.goexplore else 0

    @property
    def screen(self) -> np.ndarray | None:
        return self.env.screen

    def coverage_summary(self) -> dict[str, Any]:
        return self.coverage.summary()

    def finish(self, out: Path) -> dict[str, Any]:
        """End the run: judge the cross-episode detectors, write the report, and score
        it against the answer key."""
        tail = [raw_finding(f, self.level.name, self.steps, self.seed)
                for f in self.env.finalize_oracles()]  # fmt: skip
        self.findings += tail
        s = self.coverage.summary()
        rep = build_report(
            run_id=f"qa_lab_{self.tester}", map_name=self.level.name, findings=self.findings,
            room_cells=self.coverage.room_cells(), reach_gaps=[],
            coverage={"cell": s["cell_coverage"], "entity": s["entity_coverage"],
                      "connection": s["connection_coverage"],
                      "abstract_states": s["abstract_states"]},
            truth=self.level.truth, spec=self.level.spec,
            meta={"wad_sha256": {}, "git_sha": "demo", "env_steps": self.steps,
                  "wall_clock_s": 0.0},
            out_dir=out, wad_path=wad_of(self.level.cfg_path),
        )  # fmt: skip
        result = score(rep["findings"], self.level.defects)
        (out / "score.json").write_text(json.dumps(result, indent=1) + "\n")
        return {"report": rep, "score": result}

    def close(self) -> None:
        self.env.close()
        self.coverage.close()
        if self.goexplore is not None:
            self.goexplore.close()
