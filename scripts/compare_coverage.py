"""M7 acceptance: is T3b's cell coverage on a map significantly higher than T3a's?

    python scripts/compare_coverage.py --runs runs --a t3a_map07 --b t3b_map07 --map MAP07

Reads each run's last metrics line (`runs/<name>_seed<k>/metrics.jsonl`), takes the
map's cell coverage per seed, and runs a one-sided Mann-Whitney U test (b > a), with the
Vargha-Delaney A12 effect size. Prints a verdict and writes JSON with --out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scipy.stats import mannwhitneyu


def coverage(runs: Path, name: str, map_name: str) -> dict[int, float]:
    out = {}
    for d in sorted(runs.glob(f"{name}_seed*")):
        rows = [json.loads(line) for line in (d / "metrics.jsonl").read_text().splitlines()
                if line.strip()] if (d / "metrics.jsonl").exists() else []  # fmt: skip
        if rows and map_name in rows[-1]["coverage"]:
            out[int(d.name.rsplit("seed", 1)[1])] = float(rows[-1]["coverage"][map_name]["cell"])
    return out


def a12(b: list[float], a: list[float]) -> float:
    """Probability that a b-run beats an a-run (ties count half)."""
    wins = sum((x > y) + 0.5 * (x == y) for x in b for y in a)
    return wins / (len(a) * len(b))


def compare(runs: Path, a: str, b: str, map_name: str, alpha: float = 0.05) -> dict[str, Any]:
    ca, cb = coverage(runs, a, map_name), coverage(runs, b, map_name)
    va, vb = list(ca.values()), list(cb.values())
    res: dict[str, Any] = {"map": map_name, a: ca, b: cb, "alpha": alpha}
    if len(va) < 2 or len(vb) < 2:
        res["verdict"] = "not enough seeds"
        return res
    test = mannwhitneyu(vb, va, alternative="greater")
    res.update(u=float(test.statistic), p=float(test.pvalue), a12=a12(vb, va),
               verdict="PASS" if test.pvalue < alpha else "FAIL")  # fmt: skip
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--a", default="t3a_map07")
    ap.add_argument("--b", default="t3b_map07")
    ap.add_argument("--map", default="MAP07")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    res = compare(args.runs, args.a, args.b, args.map, args.alpha)
    for name in (args.a, args.b):
        vals = res[name]
        print(f"{name}: cell coverage on {args.map} by seed: "
              + ", ".join(f"{k}: {v:.3f}" for k, v in sorted(vals.items())))  # fmt: skip
    if "p" in res:
        print(f"Mann-Whitney U = {res['u']:.1f}, one-sided p = {res['p']:.4f}, "
              f"A12 = {res['a12']:.2f} -> M7 {res['verdict']}")
    else:
        print(res["verdict"])
    if args.out:
        args.out.write_text(json.dumps(res, indent=1) + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
