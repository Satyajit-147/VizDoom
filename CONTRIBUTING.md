# Contributing

How to change this project without breaking it. Read [README.md](README.md),
[docs/STATUS.md](docs/STATUS.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first.

## Workflow

- Work on a branch and open a pull request into `main`; don't push to `main` directly.
- Keep one change per PR, with its tests. Describe what changed and how you checked it.
- Before opening a PR, everything below must pass:

```bash
env -u PYTHONPATH .venv/bin/python -m pytest -q        # ~450 tests, ~4 min
env -u PYTHONPATH .venv/bin/ruff check qatester tests scripts
env -u PYTHONPATH .venv/bin/mypy qatester
```

(`env -u PYTHONPATH` only matters if your shell exports one, e.g. ROS.)

## The rules that keep results honest

1. **Agents never see the answer key.** Nothing under `qatester/agents/` imports ground
   truth, and nothing under `qatester/oracles/` imports vizdoom. A test enforces both.
2. **Detector thresholds are frozen** ([docs/decisions.md](docs/decisions.md) D27). A
   change needs a new D-entry with *engine evidence* (a replayed demo, an in-engine
   check), never "it makes the numbers better". Tuning a detector or a reward until an
   experiment passes would be tuning to the test.
3. **Every false-alarm fix gets a regression test that fails without the fix.** Check it
   by reverting the fix and watching the test fail.
4. **Every random source takes an explicit seeded `np.random.Generator`.**
5. **When the plan is wrong, say so**: record it in decisions.md and propose the change.
6. **Validate a detector change before any long run:**
   - replay recorded findings through the new code (`qatester/replay.py`; real finds must
     stay, false ones must go);
   - run testers on the *fixed* (bug-free) builds: any report there is a false alarm.

## Where to run what

| Task | Where |
|---|---|
| Tests, map building, detectors, demo, short tester runs | locally, CPU, no torch |
| PPO training | Kaggle GPU T4 x2 only |
| Long random / scripted / Go-Explore runs | Kaggle **CPU** sessions (no GPU quota) |

### Kaggle

`scripts/kaggle_job.py` needs the Kaggle CLI (`pip install kaggle`) and an API token in
`~/.kaggle/`. It uploads the current **git HEAD** (commit first) as a private dataset
and pushes a notebook. Kaggle user names are hard-coded as `USER` at the top of the
script: change it to your own account to run under your quota.

```bash
# GPU: two jobs per wave, one per T4; stop after wave 1 unless the agent makes progress
python scripts/kaggle_job.py --name m7 \
    --wave configs/t3b_map07.yaml:0,configs/t3b_map07.yaml:1 \
    --wave configs/t3b_map07.yaml:2 --gate 1:-0.5
# CPU: one job per wave
python scripts/kaggle_job.py --cpu --name m8 --wave configs/t4_goexplore_map07.yaml:0
# results, once finished (the log is a JSON list of {stream_name, data})
kaggle kernels output <user>/rl-qa-tester-m8 -p out/ --page-size 200 \
    --file-pattern "(logs/.*|runs/[^/]*/(metrics|goexplore)\.jsonl|results/.*/report\.json)"
python scripts/make_results.py collect out/ && python scripts/make_results.py build
```

Running something on GPU? Run the same config locally for a few thousand steps first
(PPO needs torch; at least run the CPU tiers and the tests).

## Pitfalls we hit (so you don't)

- **ViZDoom hangs on file paths longer than ~113 characters.** Recordings and saves go
  through short temp paths (D34). Keep it that way.
- **ViZDoom engines outlive a killed Python process.** Kill leftover `vizdoom` processes
  by PID.
- **Engines starting at the same instant race on `./_vizdoom/`** (D43): workers start 2 s
  apart.
- **After a demo has been recorded in a process, `DoomGame.load` freezes the game clock.**
  Go-Explore therefore records no demos (D46).
- **`DoomGame.save` advances the game by 2 tics.** Anything that measures time per step
  must use the records' tics, not assume 4 (D47).
- **Doom acts on USE when it is pressed, not while it is held**, and from the pose
  *before* a frame-skipped step (D41, D44).
- `kaggle kernels output` lists 20 files per page by default; pass `--page-size 200`.
- Don't `git stash` while a background run is importing the repo's code.
- When killing processes with `pkill -f`, make sure the pattern can't match your own
  shell's command line.

## Good first tasks

From [docs/STATUS.md](docs/STATUS.md) "What is not done", CPU-only:
- **E4**: Go-Explore reset heuristics (mixed vs frontier vs uniform) on MAP07: three
  configs differing only in `goexplore.heuristic`, 3 seeds each.
- **E7**: regression check: replay a buggy run's Bug Book entries on the fixed build;
  they should stop reproducing.
- **Calibration**: fit the temperature on MAP02/04/06 reports, report ECE on MAP03/05/07
  (`qatester/report/calibrate.py`).
- **Go-Explore patience**: let some Go-Explore iterations run until the episode times
  out, so it can confirm "stuck forever" bugs (one-way lift, trap).
