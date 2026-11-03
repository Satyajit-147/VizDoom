# rl-qa-tester: an automatic game tester (Phase 1, ViZDoom)

Team 19. A program that plays Doom levels by itself and reports their bugs: walls you can
walk through, pits you can't climb out of, switches that open nothing, and so on.

To show it works, we **plant known bugs** in levels, keep a hidden **answer key**, let a
tester play, and score its bug reports against the key: how many planted bugs it found
(recall), and how many of its reports were real (precision).

**New to the project? Read in this order:**
1. This README: what it is, how to run it.
2. [docs/STATUS.md](docs/STATUS.md): what is done, the results, what is missing.
3. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how every part works, module by module.
4. [CONTRIBUTING.md](CONTRIBUTING.md): how to make changes without breaking anything.
5. [docs/decisions.md](docs/decisions.md): the log of every deviation from the plan
   (D1–D47), and why. [docs/PLAN.md](docs/PLAN.md) is the original build specification.

## Headline results so far

| | Result |
|---|---|
| Bug detectors | **0 false alarms** on 104 bug-free maps over 1.76M steps. All 90 recorded bug reports from training replay deterministically (89 of 90 exactly). |
| Go-Explore tester on MAP07 (12 rooms, 4 planted bugs) | **92% of the map explored and 3 of 4 bugs found, on all 5 seeds**, against 40-64% explored for the other testers (one-sided Mann-Whitney p = 0.004). Milestone M8 passed. |
| Trained PPO tester (curriculum, 71 maps) | Precision 100% (51 of 51 reports real), recall 17-24%: it judges well, but doesn't reach most bugs. |
| Curiosity bonus (M7) | Did **not** significantly improve exploration over plain PPO (p = 0.50). An honest negative result, explained in [docs/STATUS.md](docs/STATUS.md). |

## How it works (one picture)

```
 maps/*.yaml or the procedural generator
        │  qatester/mapgen: compile to a Doom level, plant bugs
        ▼
 level.wad (buggy and fixed builds) + groundtruth.json (answer key, hidden)
        │
        ▼
 Tester ── action ──► QADoomEnv (qatester/env, wraps ViZDoom)
 (random / scripted /         │ each step: a StepRecord (position, health, keys, doors...)
  PPO / Go-Explore)           ▼
        ◄── observation ── 6 bug-detector families (qatester/oracles)
                              │ findings
                              ▼
 qatester/report: cluster, rank, write report.json + report.html (with replays)
                              │
                              ▼
 qatester/report/scoring.py: compare with the answer key → precision / recall
```

Testers never see the answer key: nothing under `qatester/agents/` may import ground
truth, and a test enforces it.

## Quick start (local, CPU only)

Python 3.11+ with a C++ runtime (ViZDoom ships as a wheel with its own engine).

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,demo]"     # light: no torch
.venv/bin/python -m pytest -q              # ~450 tests, about 4 minutes
```

If your shell exports a ROS (or other) `PYTHONPATH`, prefix commands with
`env -u PYTHONPATH` so it doesn't shadow the venv.

### The demo app (QA Lab)

```bash
.venv/bin/streamlit run scripts/qa_lab.py     # then open http://localhost:8501
```

1. **Map:** *Surprise me* (a random new map with 2-3 random hidden bugs), *Generate a
   new map* (you pick seed and bug types), or one of the 8 hand-made maps.
2. **Tester:** Go-Explore, Random or Scripted (and a trained PPO checkpoint if PyTorch and
   a `checkpoints/**/model.zip` are present).
3. **Start**: watch the game view, the tester's path on the map, and detections live.
4. **Finish and reveal the answer key**: planted bugs found/missed, false alarms,
   precision/recall, and a full HTML report.

Good showcase: *Hand-made map → MAP07* with Go-Explore and about 40,000 steps.

### Run a tester from the command line

```bash
# Go-Explore on MAP07, small budget, 2 workers (configs/ holds every experiment)
.venv/bin/python scripts/train.py --config configs/t4_goexplore_map07.yaml \
    --total-steps 50000 --n-envs 2
# → runs/<run_id>/ (findings, manifest)  and  results/<run_id>/MAP07/report.html
```

`scripts/train.py` runs any tier: `random`, `scripted` and `goexplore` run on CPU, and
`ppo` needs the `[train]` extra (torch + stable-baselines3), which is installed only on
Kaggle.

## Where things run

| Work | Where |
|---|---|
| Map compiler, detectors, tests, demo, random/scripted/Go-Explore testers (short) | local CPU |
| PPO training | Kaggle, GPU T4 x2 (`scripts/kaggle_job.py`) |
| Long evaluation runs (random/scripted/Go-Explore, millions of steps) | Kaggle CPU sessions (free: `kaggle_job.py --cpu`) |

## Repository map

| Path | What it is |
|---|---|
| `qatester/mapgen/` | map spec, compiler to Doom UDMF, bug planting, procedural generator, answer keys |
| `qatester/env/` | the Gymnasium env over ViZDoom: actions, observation, reward, curriculum |
| `qatester/oracles/` | the 6 bug-detector families |
| `qatester/agents/` | testers: random, scripted, PPO runner, Go-Explore |
| `qatester/archive.py` | Go-Explore's archive of saved game states |
| `qatester/report/` | clustering, severity, calibration, HTML report, scoring vs the answer key |
| `qatester/replay.py` | replay a recorded episode through the detectors |
| `qatester/demo/lab.py` | the demo's logic (the Streamlit UI is `scripts/qa_lab.py`) |
| `maps/` | the 8 hand-made maps (YAML) |
| `configs/` | one YAML per run / experiment |
| `scripts/` | entry points: train, replay, demo, Kaggle launcher, results, benchmarks |
| `tests/` | unit, engine-level and end-to-end tests |
| `docs/` | plan, decisions, status, architecture, results |
