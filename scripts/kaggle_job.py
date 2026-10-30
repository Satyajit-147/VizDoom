"""Run training jobs on Kaggle (2x T4) without touching the web UI.

    python scripts/kaggle_job.py --name m7-map07 \\
        --wave configs/t3a_map07.yaml:0,configs/t3b_map07.yaml:0 \\
        --wave configs/t3a_map07.yaml:1,configs/t3b_map07.yaml:1

1. Uploads the current git HEAD as a new version of the private dataset
   `<user>/rl-qa-tester-code` and waits until Kaggle has processed it.
2. Writes a notebook that unpacks that code, installs it, and runs the waves one after
   another. Each wave runs up to two jobs at once, one per T4, with progress every 5
   minutes in the log and the D38 watchdog. At the end it prints each run's
   evaluations, coverage and Bug Book size.
3. Pushes it as kernel `<user>/rl-qa-tester-<name>` with GPU T4 x2 and prints the URL.

Watch it with `kaggle kernels status <user>/rl-qa-tester-<name>`. Fetch results with
`kaggle kernels output ... --file-pattern`. Needs the Kaggle CLI and a token.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
USER = "idorizz69"
DATASET = "rl-qa-tester-code"


def kaggle(*args: str) -> str:
    out = subprocess.run([str(Path(sys.executable).parent / "kaggle"), *args],
                         capture_output=True, text=True, check=False)  # fmt: skip
    if out.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{out.stdout}\n{out.stderr}")
    return out.stdout


def upload_code(message: str) -> None:
    with tempfile.TemporaryDirectory(prefix="qa_ds") as d:
        subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar.gz", "-o",
                        f"{d}/code.tar.gz", "HEAD"], check=True)  # fmt: skip
        Path(d, "dataset-metadata.json").write_text(json.dumps({
            "title": DATASET, "id": f"{USER}/{DATASET}", "licenses": [{"name": "CC0-1.0"}]}))
        kaggle("datasets", "version", "-p", d, "-m", message)
    for _ in range(60):
        if "ready" in kaggle("datasets", "status", f"{USER}/{DATASET}"):
            return
        time.sleep(10)
    raise SystemExit("the code dataset did not become ready in 10 minutes")


SETUP = '''import os, glob, subprocess, pathlib, tarfile
os.environ.update(SDL_VIDEODRIVER="dummy", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
work = pathlib.Path("/kaggle/working/rl-qa-tester"); work.mkdir(parents=True, exist_ok=True)
tars = glob.glob("/kaggle/input/**/code.tar.gz", recursive=True)
tarfile.open(tars[0]).extractall(work)
os.chdir(work)
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout or "code unpacked")
'''

LAUNCH = '''import os, sys, time, signal, subprocess, pathlib, json
WAVES = __WAVES__
LOG = pathlib.Path("/kaggle/working/logs"); LOG.mkdir(parents=True, exist_ok=True)
base = {**os.environ, "SDL_VIDEODRIVER": "dummy", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
def lines(p): return p.read_text(errors="replace").splitlines() if p.exists() else []
t_all = time.time()
for w, wave in enumerate(WAVES):
    procs = {}
    for gpu, (cfg, seed) in enumerate(wave):
        name = pathlib.Path(cfg).stem + f"_seed{seed}"
        log = LOG / f"{name}.log"
        procs[name] = (subprocess.Popen([sys.executable, "scripts/train.py", "--config", cfg, "--seed", str(seed),
            "--runs-dir", "/kaggle/working/runs", "--results-dir", "/kaggle/working/results"],
            stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True,
            env={**base, "CUDA_VISIBLE_DEVICES": str(gpu)}), log)
    print(f"=== wave {w + 1}/{len(WAVES)}: {', '.join(procs)} ===", flush=True)
    start, done_at, last = time.time(), {}, 0.0
    while any(p.poll() is None for p, _ in procs.values()):
        time.sleep(30); now = time.time()
        for name, (p, log) in procs.items():
            ls = lines(log)
            if name not in done_at and any(l.startswith("RUN COMPLETE") for l in ls):
                done_at[name] = now
            stuck = name in done_at and now - done_at[name] > 180
            if p.poll() is None and (stuck or now - start > 8 * 3600):
                print(f"{name}: {'finished but not exiting' if stuck else 'past 8 h'}: dumping stacks, killing", flush=True)
                os.kill(p.pid, signal.SIGUSR1); time.sleep(5); os.killpg(p.pid, signal.SIGKILL)
        if now - last >= 300:
            last = now
            for name, (p, log) in procs.items():
                ls = lines(log); prog = [l for l in ls if l.startswith("[")]
                state = "running" if p.poll() is None else f"exited ({p.returncode})"
                print(f"[{(now - t_all) / 60:5.0f} min] {name} {state}: {(prog or ls or [''])[-1][:150]}", flush=True)
    for name, (p, log) in procs.items():
        print(f"--- {name}: last lines ---"); print("\\n".join(lines(log)[-8:]), flush=True)
'''

PREFLIGHT = '''# Pre-flight: stop here, before any long run, if the training code is broken.
import subprocess, sys
r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_ppo_smoke.py", "tests/test_bugbook_rule.py",
                    "tests/test_m7_env.py", "-q", "-x", "-p", "no:cacheprovider"], capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-2000:])
if r.returncode != 0:
    raise SystemExit("pre-flight tests failed: not starting the training waves")
'''

SUMMARY = '''# ===== SUMMARY =====
import json, pathlib
def rows(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
for run in sorted(pathlib.Path("/kaggle/working/runs").glob("*")):
    m = rows(run / "metrics.jsonl"); e = rows(run / "eval.jsonl")
    print(f"\\n== {run.name}: {m[-1]['step'] if m else 0:,} steps")
    if m:
        print("  coverage:", {k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in m[-1]["coverage"].items()})
        print("  clusters by family:", m[-1]["clusters_by_family"], "| bug book:", m[-1].get("bugbook"))
    for row in e[-2:]:
        print("  eval", row["step"], {k: (round(v["goal_rate"], 2), round(v["mean_steps"])) for k, v in row["maps"].items()})
'''


def code(src: str) -> dict[str, object]:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(True)}  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--wave", action="append", required=True,
                    help="config:seed[,config:seed] - at most two jobs, one per T4")
    ap.add_argument("--no-upload", action="store_true", help="reuse the last code upload")
    args = ap.parse_args(argv)
    waves = []
    for w in args.wave:
        jobs = [(c.rsplit(":", 1)[0], int(c.rsplit(":", 1)[1])) for c in w.split(",")]
        if len(jobs) > 2:
            raise SystemExit("a wave holds at most two jobs (one per T4)")
        waves.append(jobs)
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    if not args.no_upload:
        print(f"uploading code {sha} ...", flush=True)
        upload_code(f"code at {sha}")
    with tempfile.TemporaryDirectory(prefix="qa_k") as d:
        nb = {"cells": [code(SETUP), code('!pip install -q -e ".[train]"'), code(PREFLIGHT),
                        code(LAUNCH.replace("__WAVES__", json.dumps(waves))), code(SUMMARY)],
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                          "name": "python3"}},
              "nbformat": 4, "nbformat_minor": 5}  # fmt: skip
        for i, c in enumerate(nb["cells"]):  # type: ignore[arg-type]
            c["id"] = f"cell{i}"
        Path(d, "job.ipynb").write_text(json.dumps(nb, indent=1))
        kid = f"{USER}/rl-qa-tester-{args.name}"
        Path(d, "kernel-metadata.json").write_text(json.dumps({
            "id": kid, "title": f"rl-qa-tester {args.name}", "code_file": "job.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [f"{USER}/{DATASET}"], "kernel_sources": [],
            "competition_sources": []}))  # fmt: skip
        print(kaggle("kernels", "push", "-p", d, "--accelerator", "NvidiaTeslaT4").strip())
    print(f"watch: https://www.kaggle.com/code/{kid} (Logs tab)")


if __name__ == "__main__":
    main()
