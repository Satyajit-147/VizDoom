"""Phase 1 results: every table and figure, regenerated from results/ alone (M9).

    python scripts/make_results.py collect <downloaded kernel output dir> ...
    python scripts/make_results.py build [--results results] [--out docs/results]

`collect` copies what the tables need out of Kaggle outputs into
results/<run_id>/: each map's report.json with its answer key beside it
(groundtruth.json, from the run's build folder or rebuilt from configs/<run_name>.yaml,
since maps are deterministic), and the run's metrics.jsonl or goexplore.jsonl.

`build` scores every report against its answer key (report.scoring, the one matching
rule) and writes docs/results/RESULTS.md with the tables, and the figures as PNGs.
Nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from qatester.report.scoring import score

TIERS = {  # run_name -> (label, tier); the order is the tables' and the colours' order
    "random_map07": ("Random", "T1"), "scripted_map07": ("Scripted", "T2"),
    "t3a_map07": ("PPO, play reward", "T3a"), "t3b_map07": ("PPO + curiosity", "T3b"),
    "t4_goexplore_map07": ("Go-Explore", "T4"),
    "unseen_random": ("Random", "T1"), "unseen_scripted": ("Scripted", "T2"),
    "unseen_goexplore": ("Go-Explore", "T4"), "t3b": ("PPO + curiosity (curriculum)", "T3b"),
}  # fmt: skip
COLOUR = {"T1": "#2a78d6", "T2": "#eb6834", "T3a": "#1baf7a", "T3b": "#eda100", "T4": "#e87ba4"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
KIND_LABEL = {"missing_blocking": "leaky wall", "geometry_trap": "trap pit",
              "wrong_tag": "dead switch", "wrong_special": "broken lock",
              "one_way_lift": "one-way lift", "teleport_loop": "teleport loop",
              "bad_pickup": "bad pickup"}  # fmt: skip
M8_TARGET = 0.80
PPO_STEPS = 600_000


def run_name(run_id: str) -> str:
    return re.sub(r"_seed\d+$", "", run_id)


# --- collect ------------------------------------------------------------------------
def _rebuilt_truths(name: str, cache: dict[str, dict[str, Path]], tmp: Path) -> dict[str, Path]:
    if name not in cache:
        from qatester.config import load_config
        from qatester.env.factory import prepare_maps

        cfg = load_config(REPO / "configs" / f"{name}.yaml")
        entries = prepare_maps(cfg.maps, tmp / name)
        cache[name] = {n: e.groundtruth_path for n, e in entries.items()}
    return cache[name]


def collect(sources: Sequence[Path], dest: Path) -> None:
    cache: dict[str, dict[str, Path]] = {}
    with tempfile.TemporaryDirectory(prefix="mr") as t:
        for src in sources:
            for rep in sorted(src.glob("results/*/*/report.json")):
                run_id, map_name = rep.parent.parent.name, rep.parent.name
                out = dest / run_id / map_name
                out.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(rep, out / "report.json")
                built = [p for p in (src / "runs" / run_id / "build").rglob("*.groundtruth.json")
                         if not p.name.endswith(".fixed.groundtruth.json")
                         and _gt_name(p) == map_name]  # fmt: skip
                gt = built[0] if built else _rebuilt_truths(run_name(run_id), cache, Path(t))[
                    map_name]
                shutil.copyfile(gt, out / "groundtruth.json")
            for f in ("metrics.jsonl", "goexplore.jsonl"):
                for p in sorted(src.glob(f"runs/*/{f}")):
                    (dest / p.parent.name).mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, dest / p.parent.name / f)
    print(f"collected into {dest}: {len(list(dest.glob('*/*/report.json')))} reports")


def _gt_name(p: Path) -> str:
    d = json.loads(p.read_text())
    return str(d["name"] if str(d["name"]).startswith("GEN") else d["map"])


# --- scoring ------------------------------------------------------------------------
def load_runs(results: Path) -> dict[str, dict[str, Any]]:
    """run_id -> {"maps": {map: {"report", "truth", "score"}}, "metrics", "goexplore"}."""
    runs: dict[str, dict[str, Any]] = {}
    for d in sorted(p for p in results.iterdir() if p.is_dir()):
        if run_name(d.name) not in TIERS:
            continue
        r: dict[str, Any] = {"maps": {}, "metrics": [], "goexplore": []}
        for rep in sorted(d.glob("*/report.json")):
            report = json.loads(rep.read_text())
            truth = json.loads((rep.parent / "groundtruth.json").read_text())
            r["maps"][rep.parent.name] = {"report": report, "truth": truth,
                                          "score": score(report["findings"], truth["defects"])}  # fmt: skip
        for f in ("metrics", "goexplore"):
            p = d / f"{f}.jsonl"
            r[f] = [json.loads(line) for line in p.read_text().splitlines() if line] if p.exists() else []
        runs[d.name] = r
    return runs


def final_cell(run: dict[str, Any], map_name: str) -> float | None:
    if map_name in run["maps"]:
        return float(run["maps"][map_name]["report"]["coverage"]["cell"])
    rows = [m for m in run["metrics"] if map_name in m.get("coverage", {})]
    return float(rows[-1]["coverage"][map_name]["cell"]) if rows else None


def curve(run: dict[str, Any], map_name: str) -> list[tuple[int, float]]:
    """(run steps, cell coverage) over the run."""
    if run["metrics"]:
        return [(m["step"], m["coverage"][map_name]["cell"]) for m in run["metrics"]
                if map_name in m.get("coverage", {})]  # fmt: skip
    pts: dict[int, list[float]] = defaultdict(list)
    workers = max(1, len(run["goexplore"]))
    for item in run["goexplore"]:  # coverage is shared, so any worker's samples do
        for c in item["summary"]["curve"]:
            if map_name in c["cell"]:
                pts[c["step"] * workers].append(c["cell"][map_name])
    return [(s, max(v)) for s, v in sorted(pts.items())]


def at_step(points: Sequence[tuple[int, float]], step: int) -> float | None:
    before = [v for s, v in points if s <= step]
    return before[-1] if before else None


def group(runs: dict[str, dict[str, Any]], names: Iterable[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for n in names:
        ids = sorted(r for r in runs if run_name(r) == n)
        if ids:
            out[n] = ids
    return out


def fmt(xs: Sequence[float], pct: bool = True) -> str:
    if not xs:
        return "—"
    m = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{100 * m:.0f}% ± {100 * sd:.0f}" if pct else f"{m:.2f} ± {sd:.2f}"


def bug_totals(maps: dict[str, dict[str, Any]], names: Iterable[str] | None = None
               ) -> dict[str, Any]:  # fmt: skip
    keep = [m for m in maps if names is None or m in names]
    reports = sum(len(maps[m]["report"]["findings"]) for m in keep)
    false = sum(len(maps[m]["score"]["false_alarms"]) for m in keep)
    kinds: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for m in keep:
        for d in maps[m]["truth"]["defects"]:
            kinds[d["kind"]][1] += 1
            kinds[d["kind"]][0] += bool(maps[m]["score"]["found"][d["id"]])
    found, planted = sum(v[0] for v in kinds.values()), sum(v[1] for v in kinds.values())
    return {"maps": len(keep), "reports": reports, "false": false, "found": found,
            "planted": planted, "kinds": dict(kinds),
            "precision": (reports - false) / reports if reports else None,
            "recall": found / planted if planted else None}  # fmt: skip


def pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.0f}%"


# --- figures ------------------------------------------------------------------------
def _axes(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_map07(rows: list[tuple[str, str, list[float]]], out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=160, facecolor=SURFACE)
    _axes(ax)
    for i, (label, tier, xs) in enumerate(rows):
        m = statistics.mean(xs)
        ax.bar(i, m, width=0.62, color=COLOUR[tier], alpha=0.35, zorder=2)
        spread = [i + 0.2 * (k / max(1, len(xs) - 1) - 0.5) for k in range(len(xs))]
        ax.scatter(spread, sorted(xs), s=34, color=COLOUR[tier], edgecolor=SURFACE,
                   linewidth=1.5, zorder=3)  # fmt: skip
        ax.text(i, 0.025, f"mean {100 * m:.0f}%", ha="center", va="bottom", color=INK,
                fontsize=9)  # fmt: skip
    ax.axhline(M8_TARGET, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(len(rows) - 0.5, M8_TARGET + 0.012, "M8 target 80%", ha="right", color=MUTED,
            fontsize=8)  # fmt: skip
    ax.set_xticks(range(len(rows)), [f"{lab}\n({t})" for lab, t, _ in rows], fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.set_ylabel("MAP07 cells explored", color=INK)
    ax.set_title("MAP07 coverage per seed (dots) and mean (bar)", color=INK, fontsize=11,
                 loc="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def fig_curves(series: list[tuple[str, str, list[list[tuple[int, float]]]]], out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=160, facecolor=SURFACE)
    _axes(ax)
    for label, tier, seeds in series:
        for k, pts in enumerate(seeds):
            if pts:
                xs, ys = zip(*pts, strict=True)
                ax.plot(xs, ys, color=COLOUR[tier], linewidth=1.6, alpha=0.85,
                        label=f"{label} ({tier})" if k == 0 else None)  # fmt: skip
    ax.axhline(M8_TARGET, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1e3:.0f}k")
    ax.set_xlabel("environment steps", color=INK)
    ax.set_ylabel("MAP07 cells explored", color=INK)
    ax.set_title("MAP07 coverage during the run, one line per seed", color=INK, fontsize=11,
                 loc="left")  # fmt: skip
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def fig_kinds(rows: list[tuple[str, str, dict[str, list[int]]]], out: Path) -> None:
    import matplotlib.pyplot as plt

    kinds = list(KIND_LABEL)
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=160, facecolor=SURFACE)
    _axes(ax)
    w = 0.8 / max(1, len(rows))
    for j, (label, tier, by) in enumerate(rows):
        xs = [i - 0.4 + w * (j + 0.5) for i in range(len(kinds))]
        ys = [by[k][0] / by[k][1] if by.get(k, [0, 0])[1] else 0.0 for k in kinds]
        ax.bar(xs, ys, width=w - 0.03, color=COLOUR[tier], label=f"{label} ({tier})", zorder=2)
    ax.set_xticks(range(len(kinds)), [KIND_LABEL[k] for k in kinds], fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.set_ylabel("planted bugs found", color=INK)
    ax.set_title("Unseen maps: share of planted bugs found, by bug type", color=INK,
                 fontsize=11, loc="left")  # fmt: skip
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


# --- build --------------------------------------------------------------------------
def build(results: Path, out: Path) -> None:
    from scipy.stats import mannwhitneyu

    runs = load_runs(results)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Phase 1 results", "",
          ("Generated by `python scripts/make_results.py build` from `results/` alone. Every "
          "bug count is scored against the maps' answer keys with `qatester/report/scoring.py`: "
          "a report counts only if its detector family is the one the bug should trigger and it "
          "lies within 128 map units of the bug."), ""]  # fmt: skip

    # MAP07
    m07 = group(runs, ["random_map07", "scripted_map07", "t3a_map07", "t3b_map07",
                       "t4_goexplore_map07"])  # fmt: skip
    if m07:
        md += ["## MAP07 SPRAWL (12 rooms, 4 planted bugs)", "",
               ("| Tester | Steps | Seeds | Cells explored, final | at 600k steps | "
               "Bugs found (of 4) | False alarms |"), "|---|---|---|---|---|---|---|"]  # fmt: skip
        fig_rows, curves, finals = [], [], {}
        for name, ids in m07.items():
            label, tier = TIERS[name]
            cov = [c for i in ids if (c := final_cell(runs[i], "MAP07")) is not None]
            mid = [v for i in ids if (v := at_step(curve(runs[i], "MAP07"), PPO_STEPS)) is not None]
            scored = [runs[i]["maps"]["MAP07"]["score"] for i in ids if "MAP07" in runs[i]["maps"]]
            found = [sum(bool(v) for v in s["found"].values()) for s in scored]
            false = sum(len(s["false_alarms"]) for s in scored)
            steps = max(int(runs[i]["maps"]["MAP07"]["report"]["env_steps"])
                        for i in ids if "MAP07" in runs[i]["maps"]) if scored else PPO_STEPS  # fmt: skip
            md.append(f"| {label} ({tier}) | {steps / 1e3:.0f}k | {len(cov)} | {fmt(cov)} | "
                      f"{fmt(mid) if tier in ('T3a', 'T3b', 'T4') else '—'} | "
                      f"{fmt(found, pct=False) if found else '—'} (n={len(found)}) | "
                      f"{false} |")  # fmt: skip
            fig_rows.append((label, tier, cov))
            finals[tier] = cov
            if tier in ("T3a", "T3b", "T4"):
                curves.append((label, tier, [curve(runs[i], "MAP07") for i in ids]))
        md.append("")
        if "T4" in finals:
            ok = sum(c >= M8_TARGET for c in finals["T4"])
            md += [(f"**M8 acceptance** (Go-Explore covers at least 80% of MAP07 at 1M steps): "
                   f"{ok} of {len(finals['T4'])} seeds, "
                   f"**{'PASS' if ok == len(finals['T4']) else 'FAIL'}**."), ""]  # fmt: skip
        for a, b in (("T3a", "T3b"), ("T3a", "T4"), ("T1", "T4")):
            if a in finals and b in finals and len(finals[a]) > 1 and len(finals[b]) > 1:
                u, p = mannwhitneyu(finals[b], finals[a], alternative="greater")
                a12 = (sum((x > y) + 0.5 * (x == y) for x in finals[b] for y in finals[a])
                       / (len(finals[a]) * len(finals[b])))  # fmt: skip
                md.append(f"- {b} > {a} on final MAP07 coverage: one-sided Mann-Whitney "
                          f"U = {u:.0f}, p = {p:.3f}, A12 = {a12:.2f}")
        md.append("")
        fig_map07(fig_rows, out / "map07_coverage.png")
        fig_curves(curves, out / "map07_curves.png")
        md += ["![MAP07 coverage](map07_coverage.png)", "", "![MAP07 curves](map07_curves.png)",
               "", ("The PPO tiers trained for 600k steps. The random, scripted and Go-Explore "
               "testers ran 1M; the *at 600k* column compares them at the PPO budget where a "
               "curve exists."), ""]  # fmt: skip

    # unseen maps
    un = group(runs, ["unseen_random", "unseen_scripted", "unseen_goexplore"])
    if un:
        md += ["## Maps no tester was tuned on", "",
               ("32 generated maps from the *test* split (never used in training), MAP08 "
               "HOLDOUT, and the hand-made maps MAP01-MAP06 (MAP01 has no bugs: any report "
               "there is a false alarm). About 100k steps per map."), "",
               ("| Tester | Maps | Bug reports | False alarms | Precision | Planted bugs found "
               "| Recall | Cells explored, mean |"), "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        kind_rows = []
        for name, ids in un.items():
            label, tier = TIERS[name]
            maps = runs[ids[0]]["maps"]
            t = bug_totals(maps)
            cov = statistics.mean(m["report"]["coverage"]["cell"] for m in maps.values())
            md.append(f"| {label} ({tier}) | {t['maps']} | {t['reports']} | {t['false']} | "
                      f"{pct(t['precision'])} | {t['found']} / {t['planted']} | "
                      f"{pct(t['recall'])} | {100 * cov:.0f}% |")  # fmt: skip
            kind_rows.append((label, tier, t["kinds"]))
        md += ["", "Planted bugs found, by type:", "",
               "| Bug type | " + " | ".join(f"{TIERS[n][0]} ({TIERS[n][1]})" for n in un) + " |",
               "|---|" + "---|" * len(un)]  # fmt: skip
        for k, kind in KIND_LABEL.items():
            cells = [f"{r[2][k][0]} / {r[2][k][1]}" if k in r[2] else "—" for r in kind_rows]
            md.append(f"| {kind} | " + " | ".join(cells) + " |")
        md.append("")
        fig_kinds(kind_rows, out / "unseen_by_kind.png")
        md += ["![Bugs found by type](unseen_by_kind.png)", ""]

    # the trained curriculum agent, on its training maps
    cur = group(runs, ["t3b"])
    if cur:
        md += ["## PPO + curiosity on the curriculum (71 training maps, 2M steps)", "",
               "| Seed | Bug reports | False alarms | Precision | Planted bugs found | Recall |",
               "|---|---|---|---|---|---|"]  # fmt: skip
        for i in cur["t3b"]:
            t = bug_totals(runs[i]["maps"])
            md.append(f"| {i.rsplit('seed', 1)[1]} | {t['reports']} | {t['false']} | "
                      f"{pct(t['precision'])} | {t['found']} / {t['planted']} | "
                      f"{pct(t['recall'])} |")  # fmt: skip
        md.append("")
    (out / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out / 'RESULTS.md'} and {len(list(out.glob('*.png')))} figures")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("sources", nargs="+", type=Path)
    c.add_argument("--results", type=Path, default=REPO / "results")
    b = sub.add_parser("build")
    b.add_argument("--results", type=Path, default=REPO / "results")
    b.add_argument("--out", type=Path, default=REPO / "docs" / "results")
    args = ap.parse_args(argv)
    if args.cmd == "collect":
        collect(args.sources, args.results)
    else:
        build(args.results, args.out)


if __name__ == "__main__":
    main()
