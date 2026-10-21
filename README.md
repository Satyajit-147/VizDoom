# rl-qa-tester

RL-based QA tester for games, Phase 1 (ViZDoom). Team 19.

The build specification is [docs/PLAN.md](docs/PLAN.md) and is normative. Deviations from it
are recorded in [docs/decisions.md](docs/decisions.md).

## Where things run

| Work | Where |
|---|---|
| Map compiler, oracle unit tests, smoke tests | local dev machine (light venv, no torch) |
| Training, throughput benches, experiments E1–E7 | Kaggle notebook, 2x T4 |

## Local setup (no torch)

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/test_smoke.py -q   # M0 gate
```

If your shell exports a ROS `PYTHONPATH`, prefix commands with `env -u PYTHONPATH` so ROS
packages do not shadow the venv.

## Kaggle setup

Repo: https://github.com/Satyajit-147/VizDoom

Open `notebooks/kaggle_m0_smoke.ipynb` on Kaggle with **Accelerator: GPU T4 x2** and
**Internet: on**. It clones this repo, runs `pip install -e ".[train,dev]"` and runs the M0
gate. For a private repo, add a `GITHUB_TOKEN` secret under Add-ons → Secrets.

The vizdoom wheel bundles `freedoom2.wad`, so no IWAD download is needed.
