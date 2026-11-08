# Project status (2026-11-08)

What is built, what the results are, what is missing. The *why* behind every change from
the plan is in [decisions.md](decisions.md) (cited as D-numbers below).

## Milestones

The plan ([PLAN.md](PLAN.md)) has ten milestones; each has an acceptance test that must
pass before the next one starts.

| Milestone | What | Status |
|---|---|---|
| M0 | Repo, environment, ViZDoom smoke test | ✅ done |
| M1 | Map spec + compiler (YAML → Doom UDMF level) | ✅ done (D12) |
| M2 | 8 hand-made maps, validated in the engine | ✅ done (D19) |
| M3 | Planting the 7 bug types; buggy and fixed builds; answer keys | ✅ done (D24) |
| M4 | The 6 bug-detector families | ✅ done (D28), hardened since (D36, D41, D44, D47) |
| M5 | Gymnasium env, reward, coverage tracking | ✅ done (D30) |
| — | Procedural map generator (train/test split for unseen-map tests) | ✅ done (D31, D32) |
| M6 | Testers T1 (random), T2 (scripted), T3a (PPO) running; PPO learns MAP01 | ✅ passed: 100% / 90% goal rate at eval (D40) |
| M7 | T3b (PPO + curiosity + Bug Book) explores MAP07 significantly better than T3a | ❌ failed twice (D44, D45): a documented negative result |
| M8 | T4 Go-Explore covers ≥ 80% of MAP07 at 1M steps | ✅ passed: 92% on all 5 seeds (D46) |
| M9 | Full evaluation E1-E7, figures, calibration | 🟡 partial (see below) |
| Demo | QA Lab app: generate maps, plant bugs, watch a tester, reveal the answer key | ✅ done |
| — | Go-Explore patience: confirm "stuck forever" bugs | ✅ 4/4 MAP07 bugs on all 5 seeds (D48) |
| Fixer F0 | Six edit ops that can repair every bug kind | ✅ all 16 planted bugs, all 7 kinds, in the engine (D49) |
| Fixer F1 | Edit menu built from QA's report and the level only | ✅ right fix in the menu for 114/114 bugs (D50) |
| Fixer loop | detect → fix → re-test on generated levels, **non-learned** Fixer | ✅ working; results below |
| Fixer F2-F6 | LLM Fixer, GRPO training, Planter co-evolution | ❌ not built ([FIXER_PLAN.md](FIXER_PLAN.md)) |

## Results

### MAP07 SPRAWL (12 rooms, 4 planted bugs): the main comparison

| Tester | Steps | Map explored (5 seeds) | Planted bugs found (of 4) |
|---|---|---|---|
| T1 Random | 1M | 40% ± 9 | 0.6 |
| T2 Scripted | 1M | 46% ± 7 | 0.8 |
| T3a PPO (play reward) | 600k | 64% ± 12 | 1.0 (2 seeds with reports) |
| T3b PPO + curiosity | 600k | 60% ± 20 | 1.3 (3 seeds with reports) |
| **T4 Go-Explore** | 1M | **92% ± 0** | **3 of 4, every seed** |
| **T4 + patience** (D48) | 1M | **92%, every seed** | **4 of 4, every seed** (0 false alarms) |

- T4 > T3a and T4 > T1 on coverage: one-sided Mann-Whitney U = 25, p = 0.004, A12 = 1.00
  (the strongest possible result with 5 vs 5 seeds).
- T3b > T3a: U = 13, p = 0.50, A12 = 0.52 → M7 fails.
- The one MAP07 bug plain Go-Explore misses is the one-way lift: see "Known limitations".
  With patience (10% of steps on runs that play until the episode times out) it is found
  on every seed, at the same coverage (D48; local Windows runs, 5 seeds).

These numbers are from the first Kaggle run of these configs. A rerun on the final
detector code (D47) is in progress; `scripts/make_results.py` turns its output into
`docs/results/RESULTS.md` and figures.

### Bug detectors (precision)

- Replaying all 90 bug reports from the curriculum training run through the current
  detectors: **0 false alarms** (48 before the D44 fixes), and **41 planted bugs found**
  (35 before; none lost).
- Random and scripted testers on the bug-free builds of 104 maps (8 hand-made, 64
  generated training maps, 32 generated test maps), 1.76M steps: **0 false alarms**.
- 24 demo runs (random maps with random bugs, three testers): 8 false alarms before the
  D47 fixes, **0 after**, same bugs found.

### The trained curriculum agent (T3b on 71 maps, 2M steps, 2 seeds)

Precision **100%** (21 and 30 reports, all real). Recall **17% and 24%** of 119 planted bugs.
By type: trap pits 7-9/22, one-way lifts 4-7/16, bad pickups 4/16, leaky walls 2-3/12,
teleport loops 1-5/23, broken locks 1/11, dead switches 0/19. It never reaches a level's
goal: reaching the bugs, not judging them, is the bottleneck.

### Why M7 failed (D44, D45)

- First run: the novelty bonus was 20-100× the task reward, so the agent ignored keys and
  doors and wandered (coverage 33-47%). Fix: `adversarial.novelty_scale = 0.05`.
- Rerun: novelty no longer drowns the task, but results are bimodal. Two T3b seeds beat
  every T3a seed (82%, 79%), and one never left the first third of MAP07 (33%). Leaving it
  needs one door opened with USE while facing it, then walked through within ~37 steps;
  a count-based bonus does not reliably teach that.
- Go-Explore (M8) fixes exactly this failure: it saves the game at every new place and
  restarts from there, so progress past a door is never lost.

### Fixer loop: detect → fix → re-test (non-learned Fixer)

8 generated levels (seeds 201, 302, 403, 504, 1201, 1302, 1403, 1504), 2 planted bugs each.
QA is T4 Go-Explore + patience, 40k steps per run. The Fixer sees only QA's report and the
level; for each report it tries the menu's candidate fixes nearest-first (at most 6), and
keeps a fix only if a fresh QA run no longer reports the bug, coverage stays ≥ 80% of the
first run and no new bug appears; anything else is undone. The answer key is used **only
afterwards**, to score: each bug's engine scenario is replayed on the final level, and a
fresh QA seed checks for false alarms. Run: `scripts/fix_loop.py`.

| | Batch 1 (first menu) | **Batch 2 (menu rules D51)** |
|---|---|---|
| planted bugs (all confirmed in the engine) | 16 | 16 |
| found by QA | 11 | 11 |
| fixes kept | 9 | 9 |
| **truly repaired (engine scenario gone)** | 7 + 2 shortcuts | **9 of 11, no shortcuts** |
| levels whose exit still works | 8 / 8 | 8 / 8 |
| false alarms on the final levels | 0 | 0 |

Batch 1 exposed two shortcuts the reward allowed: a teleport loop "fixed" by walling off
the teleport pad (which also hid a trap behind it), and a one-way lift "fixed" by raising
the lower room to the top of the lift. The menu no longer offers either (D51, regression
tests in `tests/test_fixer_menu.py`); in batch 2 the same levels get real repairs:
teleport destinations moved back to their rooms, and the hidden trap found and raised.

Batch 2 by bug kind:

| Bug kind | planted | found by QA | repaired |
|---|---|---|---|
| wrong_tag (dead switch) | 3 | 3 | 3 |
| geometry_trap (pit) | 3 | 3 | 3 |
| teleport_loop | 2 | 2 | 2 |
| bad_pickup | 2 | 1 | 1 |
| one_way_lift | 3 | 1 | 0 |
| wrong_special (broken lock) | 3 | 1 | 0 |

The two found-but-unrepaired bugs: a broken lock (level 1403), where 6 attempts did not
remove the report, and a one-way lift (level 403), where copying the lift's trigger to its
other lines removed the report but QA then reported something new, so the loop rejected
it (the rule is strict on purpose; whether that new report was real is not checked). 5 bugs were never
reported by QA at 40k steps, so the Fixer never saw them; those are QA's recall, not the
Fixer's.

## What is not done

| Item | Why not / what it needs |
|---|---|
| **T4 with the PPO policy.** The plan's T4 explores from each restart with the trained agent; ours uses a sticky random explorer that presses USE when something usable is in front (D46) | GPU training with save/load resets inside PPO rollouts (treat each reset as an episode boundary for GAE): days of work |
| **E3** alpha sweep (reward balance α ∈ {0, 0.5, 1} on T3b, 3 seeds) | ~4 GPU hours |
| **E5** component ablations (no Bug Book, no productivity guard, absolute position) | ~6 GPU hours |
| **E4** reset heuristic comparison (mixed vs frontier vs uniform on T4) | CPU only now that T4 is CPU: configs + runs, ~half a day |
| **E6** T4 on unseen maps | running now (`configs/unseen_*.yaml`) |
| **E7** regression: replay the Bug Book of a buggy build on the fixed build | CPU only; `qatester/replay.py` exists; needs a script |
| **Calibration** (fit temperature on MAP02/04/06, report ECE on MAP03/05/07) | CPU only; `qatester/report/calibrate.py` exists; needs runs + a script |
| **PPO in the demo app** | CPU PyTorch locally + a checkpoint from the Kaggle outputs |
| Higher recall from PPO | research: the agent must reach more of each level |
| **Learned Fixer** (F2-F6 in [FIXER_PLAN.md](FIXER_PLAN.md)) | the loop runs with a non-learned Fixer; an LLM Fixer (Qwen + GRPO, rewarded by QA's re-test) and the Planter that makes the loop co-evolve are designed, not built |
| **LLM co-evolution, full run** (repo `Sanjayazhagan/game-qa-coevolution`) | Phase 2 trained twice: the GRPO planner barely beats untrained Qwen (scale run: 69% vs 63% of the oracle on training levels, 63% vs 76% on unseen ones). Phase 3 co-evolution ran once as a 1-round check (Mistake Book 3, QA findings 0); the 8-round run crashed on a checkpoint bug, since fixed, and was not rerun (~3.5 GPU hours) |

Kaggle GPU quota: 30 hours a week per account, reset every Saturday 00:00 UTC.

## Known limitations (be upfront about these)

1. **Go-Explore can't confirm "stuck forever" bugs** (fixed for MAP07 by patience, D48).
   The softlock detector confirms a one-way lift or a trap only when an episode *times
   out* with the player stuck, and plain Go-Explore restarts every 500 steps. Patience
   lets 10% of its steps play to the time-out and finds the lift on every seed; it is off
   by default, so turn it on (`goexplore.patience: 0.1`) for new runs.
2. **Go-Explore findings have no replay.** A ViZDoom demo replays only from the map start,
   and in a process that has recorded a demo, ViZDoom freezes the clock after every
   load (D46). Go-Explore reports carry a position but no `.lmp`.
3. **Puzzle levels limit exploration.** On generated maps a switch can open a door in
   another room. Random-style exploration rarely presses the right switch, so demo runs
   explore 25-85% of a map and find 0-3 of its 2-3 bugs.
4. **Results are on Doom-like levels generated by our own generator**, with 7 bug types we
   defined. Real game bugs are messier.

## Infrastructure notes

- Training and long runs go through `scripts/kaggle_job.py` (see CONTRIBUTING.md). It
  uploads the code as a private Kaggle dataset and pushes a notebook that runs a
  pre-flight test cell before any long job; GPU jobs can have a progress gate that stops
  a run whose agent isn't learning; a watchdog kills a finished run that hangs on exit.
- Every finding carries a seed, map, step count and (for non-Go-Explore testers) a replay,
  so any report can be reproduced with `scripts/replay.py`.
