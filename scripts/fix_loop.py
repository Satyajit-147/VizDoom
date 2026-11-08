"""The detect -> fix -> re-test loop on generated levels, scored afterwards (FIXER_PLAN).

    python scripts/fix_loop.py --out runs/fixloop --levels 3 --seed-start 100
    python scripts/fix_loop.py --out runs/fixloop --forever          # the endless loop
    python scripts/fix_loop.py --summarize runs/fixloop               # table of all levels

Per level: generate a map with random planted bugs; QA (Go-Explore with patience + the
6 detectors) reports bugs; the non-learned Fixer (qatester/fixer/loop.py) repairs what QA
reports, keeping only fixes QA confirms. The answer key is read only AFTER the loop, to
score it two independent ways:
- a fresh QA run on the final level with a different seed;
- each planted bug's own engine scenario (tests/test_defects_engine.py), on the original
  level (control: it must show) and on the final one: the gold standard for "repaired".
Each level is written to <out>/level_<seed>.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))  # the engine scenarios and the test driver

from doomdriver import Driver, approach
from test_defects_engine import SCENARIOS

from qatester.agents.goexplore import GoExplore
from qatester.coverage import CoverageTracker
from qatester.demo.lab import LabLevel, make_level
from qatester.env.vizdoom_env import QADoomEnv, wad_of
from qatester.fixer.edits import describe
from qatester.fixer.loop import QAResult, fix_level
from qatester.fixer.menu import Report
from qatester.mapgen.cli import compile_checked
from qatester.mapgen.geometry import EXIT_NORMAL, CompiledMap
from qatester.mapgen.procgen import DEFECT_KINDS
from qatester.report.emit import build_report, raw_finding
from qatester.report.scoring import score


def make_qa(level: LabLevel, steps: int, patience: float, work: Path):  # type: ignore[no-untyped-def]
    """QA as the Fixer sees it: Go-Explore + detectors -> reports and coverage only."""

    def qa(cfg: Path, seed: int) -> QAResult:
        cov = CoverageTracker(level.truth)
        env = QADoomEnv(cfg, level.truth, cov, None, seed, None, map_name=level.name)
        ge = GoExplore(patience=patience, root=Path(tempfile.mkdtemp(dir=work)))
        found: list[dict[str, Any]] = []
        try:
            for info, _ in ge.walk(env, np.random.default_rng(seed)):
                found += [raw_finding(f, level.name, ge.steps, seed) for f in info["findings"]]
                if ge.steps >= steps:
                    break
            found += [raw_finding(f, level.name, ge.steps, seed) for f in env.finalize_oracles()]
            s = cov.summary()
            rep = build_report(
                run_id=f"fixloop_{cfg.stem}", map_name=level.name, findings=found,
                room_cells=cov.room_cells(), reach_gaps=[],
                coverage={"cell": s["cell_coverage"], "entity": s["entity_coverage"],
                          "connection": s["connection_coverage"],
                          "abstract_states": s["abstract_states"]},
                truth=level.truth, spec=level.spec,
                meta={"wad_sha256": {}, "git_sha": "fixloop", "env_steps": ge.steps,
                      "wall_clock_s": 0.0},
                out_dir=work / f"report_{cfg.stem}_{seed}", wad_path=wad_of(cfg),
            )  # fmt: skip
        finally:
            env.close()
            cov.close()
            ge.close()
        reports = tuple(Report(f["oracle"], f["kind"], tuple(f["position"]))  # type: ignore[arg-type]
                        for f in rep["findings"])  # fmt: skip
        return QAResult(reports, float(s["cell_coverage"]))

    return qa


def shows(cfg: Path, cm: CompiledMap, defect: Any) -> bool:
    d = Driver(cfg)
    try:
        return bool(SCENARIOS[defect.kind](d, cm, defect))
    finally:
        d.close()


def exit_works(cfg: Path, cm: CompiledMap) -> bool:
    m = next(m for m in cm.marks if m.special.special == EXIT_NORMAL)
    room = cm.spec.room(cm.spec.thing(m.source).room)
    x, y, angle = approach(room, m.axis, m.at, (m.lo + m.hi) / 2, 40)
    d = Driver(cfg)
    try:
        d.reset()
        d.warp(x, y)
        d.face(angle)
        d.use()
        d.idle(2)
        return bool(d.game.is_episode_finished())
    finally:
        d.close()


def _as_findings(reports: tuple[Report, ...]) -> list[dict[str, Any]]:
    return [{"id": str(i), "oracle": r.oracle, "position": list(r.position)}
            for i, r in enumerate(reports)]  # fmt: skip


def run_level(seed: int, kinds: list[str], args: argparse.Namespace) -> dict[str, Any]:
    work = args.out / f"level_{seed}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    level = make_level(work / "gen", seed=seed, kinds=kinds)
    cm = compile_checked(level.spec)  # the buggy level the Fixer starts from
    qa = make_qa(level, args.qa_steps, args.patience, work)
    print(f"=== level seed {seed} ({level.name}), planted kinds {kinds}", flush=True)
    res = fix_level(cm.textmap(), level.spec.map_lump, level.cfg_path.read_text(), qa,
                    work / "levels", seed=seed, max_attempts=args.max_attempts,
                    log=lambda s: print("  " + s, flush=True))  # fmt: skip

    # --- scoring, after the loop: the answer key is read only from here on ---
    defects = level.defects
    before = score(_as_findings(res.initial.reports), defects)
    fresh = qa(res.final_cfg, seed + 1000)
    after = score(_as_findings(fresh.reports), defects)
    bugs = []
    for d in cm.spec.defects:
        bugs.append({
            "id": d.id, "kind": d.kind,
            "qa_found_before": bool(before["found"][d.id]),
            "engine_shows_before": shows(level.cfg_path, cm, d),
            "engine_shows_after": shows(res.final_cfg, cm, d),
            "qa_found_after_fresh_seed": bool(after["found"][d.id]),
        })  # fmt: skip
    out = {
        "seed": seed, "map": level.name, "kinds": kinds, "qa_steps": args.qa_steps,
        "patience": args.patience,
        "initial": {"reports": len(res.initial.reports), "coverage": res.initial.coverage,
                    "false_alarms": len(before["false_alarms"])},
        "final": {"reports": len(res.final.reports), "coverage": res.final.coverage,
                  "fresh_seed_reports": len(fresh.reports),
                  "fresh_seed_false_alarms": len(after["false_alarms"]),
                  "exit_works": exit_works(res.final_cfg, cm)},
        "attempts": [{"target": f"{a.target.oracle}/{a.target.kind}",
                      "edits": [describe(e) for e in a.edits], "reward": round(a.reward, 3),
                      "gone": a.gone, "damaged": a.damaged, "new_reports": a.new_reports,
                      "kept": a.accepted} for a in res.attempts],
        "bugs": bugs,
    }  # fmt: skip
    (args.out / f"level_{seed}.json").write_text(json.dumps(out, indent=1) + "\n")
    rep = [f"{b['kind']}: QA found {b['qa_found_before']}, engine shows "
           f"{b['engine_shows_before']} -> {b['engine_shows_after']}" for b in bugs]
    print("  RESULT: " + "; ".join(rep), flush=True)
    return out


def summarize(out: Path) -> str:
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("level_*.json"))]
    bugs = [b for r in rows for b in r["bugs"]]
    planted = len(bugs)
    real = [b for b in bugs if b["engine_shows_before"]]  # control passed
    found = [b for b in real if b["qa_found_before"]]
    repaired = [b for b in found if not b["engine_shows_after"]]
    repaired_unfound = [b for b in real if not b["qa_found_before"] and not b["engine_shows_after"]]
    attempts = [a for r in rows for a in r["attempts"]]
    lines = [
        f"# Fixer loop results ({len(rows)} generated levels, non-learned Fixer)", "",
        f"- planted bugs: {planted} ({len(real)} confirmed in the engine before any fix)",
        f"- found by QA: {len(found)} / {len(real)}",
        f"- **repaired (engine scenario no longer shows): {len(repaired)} / {len(found)} of those QA found**",
        f"- repaired although QA never reported them (side effects): {len(repaired_unfound)}",
        f"- fix attempts: {len(attempts)}, kept: {sum(a['kept'] for a in attempts)}",
        (f"- levels whose exit still works after the loop: "
         f"{sum(r['final']['exit_works'] for r in rows)} / {len(rows)}"),
        (f"- false alarms on the final levels (fresh QA seed): "
         f"{sum(r['final']['fresh_seed_false_alarms'] for r in rows)}"),
        "", "| Bug kind | planted (confirmed) | found by QA | repaired | still found by fresh QA |",
        "|---|---|---|---|---|",
    ]  # fmt: skip
    for k in sorted({b["kind"] for b in bugs}):
        kb = [b for b in real if b["kind"] == k]
        kf = [b for b in kb if b["qa_found_before"]]
        lines.append(f"| {k} | {len(kb)} | {len(kf)} | "
                     f"{sum(not b['engine_shows_after'] for b in kf)} | "
                     f"{sum(b['qa_found_after_fresh_seed'] for b in kb)} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, help="results folder (required unless --summarize)")
    ap.add_argument("--levels", type=int, default=1)
    ap.add_argument("--forever", action="store_true", help="the endless outer loop")
    ap.add_argument("--seed-start", type=int, default=100)
    ap.add_argument("--kinds", nargs="*", help="bug kinds to plant (default: 2 random)")
    ap.add_argument("--qa-steps", type=int, default=40_000)
    ap.add_argument("--patience", type=float, default=0.1)
    ap.add_argument("--max-attempts", type=int, default=6)
    ap.add_argument("--summarize", type=Path, help="print the table for a results folder")
    args = ap.parse_args(argv)
    if args.summarize:
        print(summarize(args.summarize))
        return
    if args.out is None:
        ap.error("--out is required")
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed_start)
    count = itertools.count() if args.forever else range(args.levels)
    for i in count:
        kinds = args.kinds or [str(k) for k in rng.choice(sorted(DEFECT_KINDS), size=2, replace=False)]
        run_level(args.seed_start + 1000 * i, kinds, args)
    (args.out / "SUMMARY.md").write_text(summarize(args.out))
    print(summarize(args.out))


if __name__ == "__main__":
    main()
