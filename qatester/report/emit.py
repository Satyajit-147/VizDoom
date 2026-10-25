"""Step 5: report.json and a self-contained report.html per map (PLAN: Report emission).

The JSON schema is the plan's, fixed because the Phase-2 builder consumes it. The HTML
is rendered from the same dict, so the two cannot disagree. `coverage_gaps` is its own
list: "the tester did not go there" is never mixed into findings.

Each finding's representative demo is copied next to the report, together with the WAD,
so a results folder replays on its own (`scripts/replay.py`).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jinja2

from qatester.mapgen.geometry import compile_map
from qatester.mapgen.groundtruth import GroundTruth
from qatester.mapgen.preview import render_svg
from qatester.mapgen.spec import MapSpec
from qatester.oracles.base import Finding
from qatester.report.calibrate import apply_temperature
from qatester.report.cluster import Cluster, RawFinding, cluster_findings
from qatester.report.severity import critical_rooms, room_of, severity

ABBR = {"crash": "crash", "stuck": "stuck", "out_of_bounds": "oob", "unreachable": "unreach",
        "invariant": "inv", "softlock": "soft"}  # fmt: skip
COLOUR = {"crash": "#000000", "stuck": "#e67e22", "out_of_bounds": "#c0392b",
          "unreachable": "#8e44ad", "invariant": "#2980b9", "softlock": "#16a085"}  # fmt: skip


def raw_finding(f: Finding, map_name: str, env_steps: int, env_seed: int) -> RawFinding:
    """A Finding as JSON-safe data, with where and when it was found."""
    d = asdict(f)
    d["position"] = [float(v) for v in f.position]
    d["tic_range"] = [int(v) for v in f.tic_range]
    d["demo_path"] = str(f.demo_path) if f.demo_path is not None else None
    d["evidence"] = json.loads(json.dumps(f.evidence, default=str))
    return {**d, "map": map_name, "env_steps": int(env_steps), "env_seed": int(env_seed)}


def summary(c: Cluster, room: str) -> str:
    f = c.representative
    ev, (x, y) = f["evidence"], f["position"][:2]
    at = f"in {room} near ({x:.0f}, {y:.0f})"
    text = {
        ("crash", "crash"): f"The engine crashed {at}: {ev.get('error', '')}.",
        ("crash", "hang"): f"The engine stopped responding {at}.",
        ("crash", "tic_stall"): f"Game time stopped advancing {at}.",
        ("stuck", "wedge"): f"The player got wedged {at} and could not move in 3+ directions.",
        ("stuck", "vertical_trap"): (f"The player fell into a pit {at}: its floor is "
                                     f"{ev.get('lowest_neighbour_floor', 0) - ev.get('floor', 0):.0f}"
                                     " mu below every way out and nothing lifts them back."),
        ("out_of_bounds", "outside_level"): (f"The player left the level {at} and travelled "
                                             f"{ev.get('depth_beyond_mu', 0):.0f} mu into "
                                             "undefined space."),
        ("out_of_bounds", "z_clip"): f"The player clipped into the floor or ceiling {at}.",
        ("out_of_bounds", "wall_traversal"): f"The player passed through a wall {at}.",
        ("unreachable", "broken_portal_graph"): (f"Rooms {', '.join(ev.get('rooms', []))} were "
                                                 "never reached although the level says they "
                                                 "connect."),
        ("unreachable", "broken_trigger"): (f"Repeated tries to open the way into "
                                            f"{', '.join(ev.get('rooms', []))} failed {at}."),
        ("softlock", "confined"): (f"The player ended episodes stranded in {room}: every way "
                                   "out was tried and failed."),
        ("softlock", "teleport_loop"): (f"The player bounced between teleports "
                                        f"{ev.get('hops', '')} times {at}."),
    }.get((c.oracle, f["kind"]))
    if text is None and c.oracle == "invariant":
        text = f"Invariant `{f['kind']}` broke {at}."
    return text or f"{c.oracle}/{f['kind']} {at}."


def room_gaps(room_cells: Mapping[str, tuple[int, int]],
              reach_gaps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:  # fmt: skip
    """Rooms no worker ever entered. `portals_attempted` sums Family 4's per-worker count
    of failed attempts on the way in."""
    out = []
    for room, (visited, total) in sorted(room_cells.items()):
        if visited == 0 and total > 0:
            tried = sum(int(g.get("portals_attempted", 0)) for g in reach_gaps
                        if room in g.get("rooms", []))  # fmt: skip
            out.append({"room": room, "cells_unvisited": total, "portals_attempted": tried})
    return out


def build_report(
    *,
    run_id: str,
    map_name: str,
    findings: Sequence[RawFinding],
    room_cells: Mapping[str, tuple[int, int]],
    reach_gaps: Sequence[Mapping[str, Any]],
    coverage: Mapping[str, float],
    truth: GroundTruth,
    spec: MapSpec,
    meta: Mapping[str, Any],
    out_dir: Path,
    wad_path: Path | None = None,
    eps: float = 96.0,
    min_samples: int = 3,
    weights: Mapping[str, float] | None = None,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Cluster, rank and write report.json / report.html for one map into `out_dir`."""
    intent = truth.intent_only() if "defects" in truth.data else truth
    critical = critical_rooms(intent)
    out_dir.mkdir(parents=True, exist_ok=True)
    clusters = cluster_findings(findings, eps, min_samples)
    rows = []
    for c in clusters:
        rep = c.representative
        room = room_of(intent, rep["position"][0], rep["position"][1])
        rows.append((severity(c, room, critical, weights), c, rep, room))
    rows.sort(key=lambda r: (-r[0], r[1].oracle, r[2]["position"]))
    counters: dict[str, int] = {}
    out_findings = []
    for sev, c, rep, room in rows:
        abbr = ABBR[c.oracle]
        counters[abbr] = counters.get(abbr, 0) + 1
        fid = f"{map_name}-{abbr}-{counters[abbr]:02d}"
        demo = None
        if rep["demo_path"] and Path(rep["demo_path"]).exists():
            (out_dir / "demos").mkdir(exist_ok=True)
            shutil.copyfile(rep["demo_path"], out_dir / "demos" / f"{fid}.lmp")
            demo = f"demos/{fid}.lmp"
        out_findings.append({
            "id": fid, "oracle": c.oracle, "kind": rep["kind"], "severity": sev,
            "confidence_raw": rep["confidence"],
            "confidence_calibrated": round(float(apply_temperature([rep["confidence"]],
                                                                   temperature)[0]), 4),
            "position": rep["position"], "room": room, "cluster_size": c.size,
            "first_seen_step": c.first_seen_step,
            "repro": {"demo": demo, "save": None, "episode": rep["evidence"].get("episode"),
                      "env_seed": rep["env_seed"], "tic_range": rep["tic_range"],
                      "steps": rep["tic_range"][1] // 4},
            "evidence": rep["evidence"], "summary": summary(c, room),
        })  # fmt: skip
    if wad_path is not None and wad_path.exists():
        shutil.copyfile(wad_path, out_dir / wad_path.name)
    report = {
        "run_id": run_id, "map": map_name, **dict(meta),
        "coverage": dict(coverage),
        "findings": out_findings,
        "coverage_gaps": room_gaps(room_cells, reach_gaps),
        "map_lump": spec.map_lump, "wad": wad_path.name if wad_path is not None else None,
        "raw_findings": len(findings), "temperature": temperature,
    }  # fmt: skip
    (out_dir / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    markers = [(f["position"][0], f["position"][1], f["id"].split("-", 1)[1], COLOUR[f["oracle"]])
               for f in out_findings]  # fmt: skip
    svg = render_svg(compile_map(spec.without_defects()), markers)
    (out_dir / "report.html").write_text(render_html(report, svg), encoding="utf-8")
    return report


def render_html(report: Mapping[str, Any], svg: str) -> str:
    env = jinja2.Environment(
        loader=jinja2.PackageLoader("qatester", "templates"),
        autoescape=jinja2.select_autoescape(["html", "j2"]),
    )
    return env.get_template("report.html.j2").render(
        r=report, svg=svg, colours=COLOUR,
        evidence_json={f["id"]: json.dumps(f["evidence"], indent=1) for f in report["findings"]},
    )
