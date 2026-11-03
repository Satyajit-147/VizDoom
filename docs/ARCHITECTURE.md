# How it works

A module-by-module guide to the code, in the order data flows through it. For *why* any
choice differs from [PLAN.md](PLAN.md), see the D-number in [decisions.md](decisions.md).

Units: Doom map units (mu); 1 tic = 1/35 s. "Frame skip 4" = one env step is 4 tics.

---

## 1. Levels and the answer key: `qatester/mapgen/`

| File | Role |
|---|---|
| `spec.py` | `MapSpec`: rooms (rectangles with floor/ceiling heights), portals between rooms (`open`, `door`, `locked_door`, `lift`, `teleport`, switch-operated doors), things (keys, health, switches, monsters, the exit switch), the goal, and planted `defects`. Loaded from YAML (`maps/*.yaml`). |
| `validate.py` | Spec checks: rooms don't overlap, portals join adjacent rooms, the goal is reachable on the fixed build, and so on. |
| `geometry.py` | Compiles a spec to Doom geometry: a grid of regions → sectors → linedefs with specials (doors, lifts, teleports, switches). Rooms are separated by a 16 mu wall strip; a lift platform reaches 64 mu into the lower room (D14). |
| `defects.py` | Plants each bug by mutating the *compiled* map (D20). |
| `udmf.py`, `wad.py` | Write UDMF TEXTMAP and pack it into a `.wad` with a ViZDoom `.cfg`. |
| `groundtruth.py` | The answer key, `*.groundtruth.json`: rooms, sectors, walls, portals (areas, teleport destinations), switch lines, things, the goal, and `defects` (kind, position, region, expected detector family). `GroundTruth.intent_only()` strips the defects; that is all detectors may see (D22). |
| `procgen.py` | The procedural generator: a seed → a grid layout of 5-14 rooms with doors, locked doors, switch doors, lifts and teleports, plus any chosen bug types. Seeds are split into `train` and `test`, so testers can be measured on maps never trained on (D18, D31). |
| `preview.py` | SVG top-down maps (used by reports and the demo). |
| `cli.py` | `build(spec_dir, out_dir, variant)`: compile a folder of specs into a WAD with a `.cfg` and an answer key per map, for `buggy` or `fixed`. |

**The 7 bug types** and the detector family each one should trigger:

| `kind` | In the game | Expected detector |
|---|---|---|
| `missing_blocking` | a wall segment you can walk through, into a pocket outside the level | out_of_bounds |
| `geometry_trap` | a pit 64+ mu deep with no way out | stuck (softlock also accepted) |
| `wrong_tag` | a switch tagged to the wrong sector: it opens nothing | unreachable |
| `wrong_special` | a locked door that opens without its key | invariant |
| `one_way_lift` | a lift with no trigger on its lower side | softlock |
| `teleport_loop` | two teleporters that drop you in front of each other's pads | softlock |
| `bad_pickup` | a health pickup that pushes health past the maximum | invariant |

**Maps:** `maps/m01_clean.yaml` … `m08_holdout.yaml`. MAP01 has no bugs (a false-alarm
check); MAP07 SPRAWL (12 rooms, 4 bugs) is the main benchmark; MAP08 is never used in
training.

## 2. The environment: `qatester/env/`

`QADoomEnv` (`vizdoom_env.py`) is a Gymnasium env over one ViZDoom map.

- **Actions** (`actions.py`): 14 discrete combinations of 9 buttons (noop, forward,
  backward, strafes, forward+turn, use, forward+use, attack, jump, forward+jump, turn
  left/right), each held for 4 tics.
- **Observation** (`observation.py`): `depth`, a stack of 64×64 depth frames, plus 18
  `scalars`: angle, velocity, on-ground, speed ratio, health, armour, ammo, keys held,
  time used, cell novelty, neighbour novelty, pickups in view, and slot 17 =
  "a usable door/lift/switch is in reach ahead".
- **Each step** builds a `StepRecord` (position, velocity, angle, health, keys, commanded
  moves, USE, live sector floor/ceiling heights…), feeds it to every detector, computes
  the reward, updates coverage, and returns `info` with the step's findings.
- `world.py`: what the agent may know about the *intended* level (keys taken, doors open,
  usable things in reach), derived from the intended level, never from the defects.
- `reward.py` + `shaping.py`: the reward (below).
- `curriculum.py`: `CurriculumEnv` switches maps between episodes by stage weights
  (`configs/*.yaml → maps.curriculum`), including across WADs.
- `factory.py`: builds maps, curricula, shared coverage counters, and one env per worker.
- `supervisor.py`: runs engine calls with a watchdog and restarts a hung engine.
- Recording: each episode can record a `.lmp` demo; Go-Explore hooks are
  `save_state(path)` / `reset(options={"load_state": path})`.

### Reward: `R = α · R_valid + (1 − α) · R_adv`

| Part | Value | Meaning |
|---|---|---|
| progress | +1 | a key taken or a critical door passed, first time per episode |
| goal | +5 | reached the exit |
| death | −5 | |
| time | −0.0005 per step | |
| shaping | +1 per 1000 mu closer to the goal | potential-based, so it telescopes and can't be farmed (D35) |
| cell novelty | 0.3 / √(1 + visits), × `novelty_scale` | visits shared across workers and the whole run |
| state novelty | 0.5 / √(1 + visits) of (cell, keys, open doors), × `novelty_scale` | weight β decays 0.5 → 0.1 over training |
| entity / portal | +1 each | first use of a thing / first crossing of a portal, per episode |
| oracle | +3 | a new bug cluster this episode |
| Bug Book | +1.5 | reaching a known bug's place, once per entry per episode |

T3a uses α = 1 (play properly only); T3b uses α = 0.5 with `novelty_scale = 0.05` (D44).
Anti-hacking rules (cells must re-arm before paying novelty again, one oracle payout per
cluster, death penalty) are tested in `tests/test_reward_hacking.py`.

## 3. The bug detectors: `qatester/oracles/`

Every detector sees only `StepRecord`s and the *intended* level. They never see the
defects, and never import vizdoom (both enforced by `tests/test_no_groundtruth_leak.py`).
Thresholds are frozen in [decisions.md](decisions.md) D27. Changing one needs a recorded
decision with evidence.

| Family | File | Kinds | Fires when |
|---|---|---|---|
| crash | `crash.py` | crash, hang, tic_stall | the engine raised, a step took > 30 s, or game time stopped |
| stuck | `stuck.py` | wedge, vertical_trap | pushing in ≥ 3 world directions on the ground without moving for 70 tics; or standing in a sector more than 24 mu below every neighbour with no lift |
| out_of_bounds | `bounds.py` | outside_level, z_clip, wall_traversal | outside every room; below the floor / above the ceiling; crossing a wall faster than running allows (teleports excused) |
| unreachable | `reach.py` | broken_trigger, broken_portal_graph | at the end of a run: rooms never visited whose every way in had ≥ 5 proper failed attempts |
| invariant | `invariants.py` | health_cap, health_monotone, ammo_nonneg, no_phantom_keys, locked_needs_key, kills_le_spawned | a game rule is broken (e.g. health > 100 without a powerup, a locked door passed without its key) |
| softlock | `softlock.py` | confined, teleport_loop | an episode times out with the player in one room for its last 40% and every exit properly tried and failed; or ≥ 4 back-and-forth teleport hops, each landing within 80 mu of a pad |

The key idea: **a detector needs evidence that the level is broken, not just that the
tester behaved badly.** `level.py`'s `AttemptTracker` defines a *proper failed attempt*:
a USE press (not a held button) that really reaches the door/lift face or switch (before
any wall, from the pose the engine acts on), while it is not moving, holding the key if
needed, followed by no movement of the door within 35 tics. Most of the false-alarm fixes
(D36, D41, D44, D47) tightened this definition, each checked against the engine.

## 4. The testers: `qatester/agents/`

| Tier | File | Policy |
|---|---|---|
| T1 random | `random_agent.py` | uniform random action |
| T2 scripted | `scripted_agent.py` | right-hand wall follower on the depth frame; USE on anything usable or on a wall at arm's length; backs off when stuck |
| T3a / T3b PPO | `policy.py`, `ppo_runner.py` | sb3-contrib `RecurrentPPO`: a small CNN on depth + an MLP on scalars → LSTM(256) → policy and value heads (941k parameters); VecNormalize on rewards; checkpoints, evaluation, Bug Book |
| T4 Go-Explore | `goexplore.py` + `../archive.py` | see below |

`rollout.py` runs T1/T2/T4 across worker processes, and `ppo_runner.py` runs PPO. Both
send findings to the main process through a queue. `qatester/runner.py` is the entry for
any config (`scripts/train.py`): build maps → run the tester → write findings, episodes,
manifest → emit reports.

### Go-Explore (T4)

```
loop:
  with probability 0.5 (if the archive has cells): pick a cell, load its save
  otherwise: start a new episode
  explore for up to 500 steps with a sticky random explorer
  (repeat the last action with p = 0.9; press USE when something usable is ahead)
  every new cell, or a cell reached earlier than before → save the game into the archive
  credit new cells to the cell the run restarted from
```

- **Cell** = (x // 64, y // 64, z // 64, keys held, open-door mask). The door mask was
  needed for MAP07: the moment a door opens is saved, so restarts begin with it open (D46).
- **Choosing a restart** (`archive.py`): `mixed` = 1/√(1+visits) × 1/(1+d), where d is
  the distance to the nearest unreached cell, × the productivity guard
  (1 + new cells found after restarts) / (1 + restarts). The `count`, `frontier` and
  `uniform` heuristics are also available.
- Save 3.6 ms, load 8.6 ms on MAP07; a save file is 37 KB; LRU eviction past 20,000 cells.

### The Bug Book: `qatester/bugbook.py`

A memory of found bugs: add (frequency 1) / confirm (+1, capped at 20) / decrement on a
replay that no longer reproduces / remove at 0. It gives the +1.5 reward target, and it is
the basis of the regression experiment E7.

## 5. Coverage: `qatester/coverage.py`

Four measures per map: cell coverage (64 mu walkable cells visited), entity coverage
(interactive things used), connection coverage (portals crossed), and distinct abstract
states. Counts live in shared memory, so all workers see the same novelty.

## 6. Reports and scoring: `qatester/report/`

| File | Role |
|---|---|
| `cluster.py` | DBSCAN (96 mu) groups repeated detections of one bug; per family |
| `severity.py` | ranks clusters by detector family, blocking effect, frequency |
| `calibrate.py` | temperature scaling of confidences (fitting it is M9 work) |
| `emit.py` | `report.json` + self-contained `report.html` per map, with the demo and WAD copied beside it so a finding replays on its own |
| `scoring.py` | **the one matching rule**: a report matches a planted bug if its family is the expected one (or an accepted alternative) and it lies within 128 mu of the bug or inside its region |

`qatester/replay.py` replays a `.lmp` tic by tic through the detectors. We used it to
validate every detector change against all recorded findings.

## 7. The demo: `qatester/demo/lab.py` + `scripts/qa_lab.py`

`make_level()` builds a generated map with chosen (or random) bug types, or a hand-made
map. `Session` plays a tester step by step, keeps the colour frame, the path and the
detections, and `finish()` writes the report and scores it against the answer key. The
Streamlit file is only the UI.

## 8. Running on Kaggle: `scripts/kaggle_job.py`

1. `git archive HEAD` → uploaded as a new version of the private dataset
   `<user>/rl-qa-tester-code`.
2. A notebook is generated: unpack, `pip install -e .[train]`, a **pre-flight** pytest
   cell (stops before any long run if the code is broken), the launcher cell (waves of
   jobs, one per GPU, progress every 5 min, a watchdog, an optional progress gate), and a
   summary cell. The launcher cell is executed locally with no waves before pushing.
3. Pushed as kernel `<user>/rl-qa-tester-<name>`, on GPU T4 x2, or on CPU with `--cpu`.

Results come back with `kaggle kernels output <kernel> --file-pattern ...`, and
`scripts/make_results.py collect …` / `build` turns them into `results/` and
`docs/results/`.

## 9. Configs: `configs/*.yaml`

A `RunConfig` (`qatester/config.py`): `agent` (random | scripted | ppo | goexplore),
`total_steps`, `n_envs`, `seed`, `maps` (variant, generated maps, curriculum stages),
`reward`, `ppo`, `eval`, `bugbook`, `goexplore`, `report`, `oracles`. One file per
experiment: `t1_random_map03`, `t3a_map01`, `t3a_map07` / `t3b_map07` (M7), `t3b`
(curriculum with 64 generated maps), `t4_goexplore_map07` / `random_map07` /
`scripted_map07` (M8, E1), `unseen_*` (E6).

## 10. Tests: `tests/`

About 450 tests. The important groups:

- `test_oracles.py`: every detector on hand-written traces (`tracebuilder.py`): each
  family fires on its positive case and stays silent on near-misses, plus a regression
  test for every false alarm we fixed.
- `test_defects.py`, `test_defects_engine.py`: every bug type is planted correctly and
  detected when played in the real engine.
- `test_env_*.py`, `test_reward_hacking.py`, `test_shaping.py`: env contract and reward.
- `test_runner.py`, `test_goexplore.py`, `test_demo_lab.py`: end to end.
- `test_no_groundtruth_leak.py`: agents never import ground truth; oracles never import vizdoom.
- `test_kaggle_job.py`: the generated Kaggle cells actually run.
- `test_ppo_smoke.py`: PPO end to end (skipped locally without torch; runs in the Kaggle pre-flight).
