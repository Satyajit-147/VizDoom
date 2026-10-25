"""One run end to end: build the maps, run the tester, write the logs, emit the reports.

    runs/<run_id>/build/         the WADs, configs and ground truth this run used
    runs/<run_id>/demos/         one .lmp per episode
    runs/<run_id>/traces/        Parquet traces of episodes with a finding
    runs/<run_id>/findings.jsonl every raw finding, with map, env seed and step
    runs/<run_id>/episodes.jsonl one line per finished episode
    runs/<run_id>/manifest.json  git SHA, WAD SHA-256s, config, seed, timings
    results/<run_id>/<map>/      report.json, report.html, demos, the WAD

The run id is `<run_name>_seed<seed>` unless given, so a resumed Kaggle session finds
its own checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from qatester.config import RunConfig
from qatester.coverage import CoverageTracker
from qatester.env.curriculum import MapEntry
from qatester.env.factory import make_coverage, map_names, prepare_maps, stages
from qatester.env.vizdoom_env import wad_of
from qatester.mapgen.groundtruth import GroundTruth
from qatester.mapgen.spec import load_spec
from qatester.report.calibrate import load_temperature
from qatester.report.emit import build_report, raw_finding


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent, check=True)  # fmt: skip
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coverage_summary(cov: CoverageTracker) -> dict[str, Any]:
    s = cov.summary()
    return {"cell": s["cell_coverage"], "entity": s["entity_coverage"],
            "connection": s["connection_coverage"], "abstract_states": s["abstract_states"]}  # fmt: skip


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def queue_rows(items: Iterable[Mapping[str, Any]], n_envs: int,
               initial_steps: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Queue items -> (raw findings, coverage-gap rows), JSON-safe. A worker counts its
    own steps; workers run in lockstep, so run-wide step = worker step x n_envs."""
    findings, gaps = [], []
    for it in items:
        if it["type"] == "finding":
            step = initial_steps + int(it["env_steps"]) * n_envs
            findings.append(raw_finding(it["finding"], it["map"], step, it["env_seed"]))
        elif it["type"] == "coverage_gap":
            gaps.append({"map": it["map"], **it["gap"]})
    return findings, gaps


def emit_reports(cfg: RunConfig, run_id: str, run_dir: Path, entries: Mapping[str, MapEntry],
                 names: Iterable[str], coverage: Mapping[str, CoverageTracker],
                 meta: Mapping[str, Any]) -> dict[str, Path]:  # fmt: skip
    findings = read_jsonl(run_dir / "findings.jsonl")
    gaps = read_jsonl(run_dir / "coverage_gaps.jsonl")
    temperature = load_temperature(cfg.report.calibration)
    out: dict[str, Path] = {}
    for name in sorted(names):
        e = entries[name]
        dest = cfg.results_dir / run_id / name
        build_report(
            run_id=run_id, map_name=name,
            findings=[f for f in findings if f["map"] == name],
            room_cells=coverage[name].room_cells(),
            reach_gaps=[g for g in gaps if g["map"] == name],
            coverage=coverage_summary(coverage[name]),
            truth=GroundTruth.load(e.groundtruth_path), spec=load_spec(e.spec_path),
            meta=meta, out_dir=dest, wad_path=wad_of(e.cfg_path),
            eps=cfg.report.dbscan_eps_mu, min_samples=cfg.report.dbscan_min_samples,
            weights=cfg.report.severity_weights, temperature=temperature,
        )  # fmt: skip
        out[name] = dest / "report.html"
    return out


def run(cfg: RunConfig, run_id: str | None = None) -> dict[str, Path]:
    """Run `cfg` and return the report path of every map it played."""
    run_id = run_id or f"{cfg.run_name}_seed{cfg.seed}"
    run_dir = cfg.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    entries = prepare_maps(cfg.maps, run_dir / "build")
    st = stages(cfg.maps, entries)
    names = map_names(st)
    coverage = make_coverage(entries, names)
    started = time.time()
    try:
        if cfg.agent == "ppo":
            from qatester.agents.ppo_runner import train  # torch only where needed

            steps = train(cfg, run_id, run_dir, entries, st, coverage)
        else:
            from qatester.agents.rollout import run_rollout

            res = run_rollout(cfg, entries, st, coverage, run_dir)
            findings, gaps = queue_rows(res.queue_items, cfg.n_envs)
            append_jsonl(run_dir / "findings.jsonl", findings)
            append_jsonl(run_dir / "coverage_gaps.jsonl", gaps)
            append_jsonl(run_dir / "episodes.jsonl", res.episodes)
            steps = res.steps
        wall = time.time() - started
        wads = sorted({wad_of(entries[n].cfg_path) for n in names})
        meta = {"wad_sha256": {w.name: sha256(w) for w in wads}, "git_sha": git_sha(),
                "env_steps": steps, "wall_clock_s": round(wall, 1)}  # fmt: skip
        manifest = {"run_id": run_id, **meta, "seed": cfg.seed, "agent": cfg.agent,
                    "maps": sorted(names), "python": sys.version.split()[0],
                    "platform": platform.platform(), "finished_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "config": cfg.model_dump(mode="json")}  # fmt: skip
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        return emit_reports(cfg, run_id, run_dir, entries, names, coverage, meta)
    finally:
        for c in coverage.values():
            c.close()


def summarise(reports: Mapping[str, Path]) -> str:
    lines = []
    for name, path in sorted(reports.items()):
        rep = json.loads(path.with_name("report.json").read_text(encoding="utf-8"))
        fams: dict[str, int] = defaultdict(int)
        for f in rep["findings"]:
            fams[f["oracle"]] += 1
        lines.append(f"{name}: {len(rep['findings'])} candidate defects "
                     f"({', '.join(f'{k} {v}' for k, v in sorted(fams.items())) or 'none'}), "
                     f"{rep['raw_findings']} raw findings, cell coverage "
                     f"{100 * rep['coverage']['cell']:.0f}% -> {path}")  # fmt: skip
    return "\n".join(lines)
