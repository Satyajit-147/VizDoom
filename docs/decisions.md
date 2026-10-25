# Decisions and deviations from PLAN.md

Every deviation from `docs/PLAN.md` is recorded here with its date, in order.

## 2026-10-21 — M0

### D1. Compute split: all training on Kaggle, light CPU work locally
All training, `bench_env.py` throughput measurements and experiments E1–E7 run in a Kaggle
notebook (2x T4). The local dev machine (12 cores, 7 GB RAM, no GPU) runs only light CPU work:
the map compiler, WAD loading checks, oracle unit tests and short smoke tests.
Code reaches Kaggle through the GitHub repo (`git clone` + `pip install -e ".[train,dev]"`).

### D2. Torch and the RL stack moved into a `[train]` extra
`torch`, `stable-baselines3` and `sb3-contrib` are in `[project.optional-dependencies].train`,
not in the core dependencies, so the local install never pulls torch. Kaggle installs
`.[train,dev]`.

### D3. Lower-bound version constraints, not exact pins
Kaggle preinstalls torch/numpy matched to its CUDA. Exact pins would force reinstalls and
risk a CUDA mismatch. Exact versions are recorded per run in `runs/<id>/manifest.json`
(already required by the plan). Locally verified with vizdoom 1.3.2 on Python 3.12.

### D4. Python 3.12 locally instead of 3.11
Only 3.12 is installed on the dev machine. `requires-python = ">=3.11"`, so the package also
works on whatever interpreter the Kaggle image ships. Nothing in the plan depends on 3.11.
Kaggle image as of 2026-10-21: Python 3.13.15, torch 2.11.0+cu128, 2 CUDA devices, 4 CPUs.
So the code must run on 3.12 (local) and 3.13 (Kaggle); target 3.11+ syntax only.

### D5. IWAD: use the freedoom2.wad bundled with the vizdoom wheel
The vizdoom 1.3.2 wheel ships `freedoom2.wad` inside the package, so the plan's
`TODO(human)` IWAD download is unnecessary on both machines. No IWAD is committed (`*.wad`
is gitignored).

### D6. Config loading: plain YAML + argparse, not Hydra
PLAN marks this as a free choice ("or plain argparse + YAML if you prefer"). Plain YAML has
fewer moving parts and no output-directory magic interfering with Kaggle paths.
`hydra-core` is not a dependency.

### D7. M0 smoke test uses the normative throughput settings
`tests/test_smoke.py` loads the stock `basic.cfg` but overrides resolution (160x120),
format (GRAY8), window/sound/console (off) and uses frame skip 4, so that the 300 steps/s
gate measures the configuration we actually train with.
Local result: 1130 steps/s. Kaggle result (GPU T4 x2 session, 4 vCPU): 2067 steps/s,
4/4 tests passed. **M0 accepted 2026-10-21.**

### D8. `test_no_groundtruth_leak.py` added at M0
Hard rules 1 and 2 are enforced from the first commit. The test checks both
`qatester.mapgen.groundtruth` (hard rules) and `qatester.groundtruth` (architecture
section), since the plan uses both names.

## 2026-10-21 — M1

### D9. Appendix A's minimal TEXTMAP has reversed linedef winding (plan error)
Appendix A lists the room corners counter-clockwise ((-320,-320) → (0,-320) → (0,0) →
(-320,0)), which puts every linedef's front (right) side **outside** the room. Verified
empirically in ViZDoom 1.3.2: built verbatim, the map loads and collision works, but the
depth buffer is all zeros, so no wall renders from inside. Clockwise order renders correctly.
The compiler orients every linedef with its front sector on the right (`geometry.py`),
`validate.check_winding` asserts it, and `tests/test_maps_engine.py` asserts the depth
buffer is non-empty so a winding regression fails loudly.

### D10. Config file names for the two build variants
PLAN names configs `build/MAPnn.cfg` but builds two variants into the same directory.
Buggy variant: `MAPnn.cfg`; fixed variant: `MAPnn.fixed.cfg`. They differ only in
`doom_scenario_path`, as the plan requires. `doom_scenario_path` is relative to the
`.cfg` location, which ViZDoom resolves correctly (tested from a temp directory).

### D11. Spec models landed in M1; validation rules stay in M2
`qatester/mapgen/spec.py` has the Pydantic models with the normative field names (plus
`union_with` from the validation rules and `key_yellow`, which the observation's key slots
and Appendix A both use). The six loader validation rules and `check_reachable()` are M2.
`maps/m01_clean.yaml` is the Appendix A single room as an M1 placeholder; M2 replaces it
with the full 4-room CLEAN map. `build()` accepts a spec directory or a single file.

### D12. M1 accepted 2026-10-21
`scripts/build_maps.py maps/m01_clean.yaml -o build/`: ViZDoom loads MAP01, the player
spawns at (-160, -160), POSITION_X stays finite through 500 random steps. 16/16 tests,
`mypy --strict qatester/mapgen/` clean.

## 2026-10-24 — M2

### D13. MapSpec extensions and extra loader checks
- New thing type `switch` with `opens: <door portal id>`: a wall switch that opens a remote
  door. MAP04's defect ("the switch intended to open the door to the east wing") cannot be
  expressed in the plan's schema without it. A door opened by a switch gets no USE trigger on
  its own faces. MAP07 uses one too.
- `union_with` rooms count as connected in `check_reachable()` (one walkable space).
- Extra checks beyond the six rules: ids unique across rooms/portals/things (defect targets
  need this); `requires` only on `locked_door`; lifts join rooms at different floors;
  teleports need `at` inside room a; things lie inside their room; a `reach_exit` goal targets
  an `exit_switch`.

### D14. How rooms, walls and portals become Doom geometry
The plan says shared walls become "two-sided linedefs ... given an impassable flag". That
gives an invisible wall, and a door with `tag=0` would act on the whole of room b. Instead:
- A wall shared by two rooms is a 16 mu (`WALL`) strip of void straddling the shared edge,
  so each room has an ordinary one-sided wall.
- `open`: the opening (width x 16) is cut out of the strip and joined to room a.
- `door` / `locked_door`: the opening is its own door sector, emitted closed, with
  `Door_Raise` / `Door_LockedRaise` on both faces (`playeruse`, `repeatspecial` so it can
  be reused). Doors and open portals refuse floor steps > 24 mu (unwalkable).
- `lift`: a platform sector over the opening plus 64 mu (`LIFT_DEPTH`) into the lower room,
  floor at the upper height, `Plat_DownWaitUpStay` on every face that borders a room.
- `teleport`: a 64x64 pad sector centred on `at` in room a whose edges carry `Teleport`
  (`playercross`); the `TeleportDest` sits at room b's centre. Things may not stand within
  32 mu of a pad.
- `exit_switch` / `switch`: a 64 mu special on the one-sided wall of the thing's room
  nearest to its `pos`.
The compiler cuts the plane along every rectangle edge, gives each grid cell its
highest-priority owner (rooms < void strips < portal sectors), emits a linedef wherever
neighbouring owners differ, and merges collinear pieces.

### D15. MAP03: the plan's balcony cannot be reached by its lift (plan error)
The worked example puts `balcony` at x 0..320, y 64..256: 64 mu from `start` and touching it
at a single corner point at most. Its own validation rule (portal rooms must share a wall at
least `width` long) rejects the `lift` portal. The balcony now spans x -320..0, y 0..256,
directly north of `start`. `exitsw` moved to (900, -160) so its nearest wall is unambiguous.

### D16. Corpus details the plan leaves open
- MAP01 (CLEAN) uses open, door and locked-door portals but no lift or teleport. Those are
  covered false-positive-wise by the `fixed` variants of MAP03/06/07/08.
- MAP04's "1 secret" is not modelled: the schema has no secret field. Add one if the
  secret count turns out to matter for an oracle.
- Monsters: one imp, in MAP07 r13 only, so monsters are present but not dominant.
- Time limits: 10500 tics (5 min) for every map except MAP07 at 21000.
- MAP08 HOLDOUT uses shapes absent from MAP01–07 (an L-shaped `union_with` corridor, a
  teleport pair guarding the key) and gets its defects in M3.

### D17. Engine-level tests replace `tests/test_m1_load.py`
`tests/test_maps_engine.py` loads all eight maps in ViZDoom and checks: the engine's
sectors and linedefs equal the compiler's; spawn position and floor; walls render; 500
random steps stay inside the level; open portals and doors work both ways; locked doors
stay shut without the key and open with it; remote doors open only from their switch;
lifts carry the player up and let them walk back down; teleports arrive; the exit switch
ends the episode. Uses GZDoom cheats (`warp`, `give`) in tests only. Verified to catch
three injected compiler bugs (lock 0, lift with no trigger, dead switch).

### D19. M2 accepted 2026-10-24
All 8 specs compile; `tests/test_geometry_invariants.py`, `tests/test_spec_validation.py`
and `tests/test_maps_engine.py` green (189 tests); every map walked once by hand in
SPECTATOR mode (`scripts/walk_map.py`) with no issues reported.

## 2026-10-24 — Requirement: find bugs on unseen maps

### D18. Generalisation is a hard requirement, tested on many unseen maps
Team requirement: the tester must find the 7 bug classes on maps it has never seen, not
only on MAP01–07. In this architecture the oracles label bugs; the policy has to learn
*where to go and what to try* (push walls, USE locked doors, ride lifts, enter dead ends).
Both halves must transfer. The plan's single holdout map (MAP08) and 7 fixed training maps
are not enough evidence. Added work:
1. **After M3:** `qatester/mapgen/procgen.py`, a seeded random MapSpec generator: random
   room layouts and portal kinds, plus 0–3 defects drawn from the 7 classes at random valid
   targets. Not the Phase-2 adversarial generator: no feedback loop.
2. **Training (M6–M8):** the curriculum mixes generated maps (train seeds) with MAP01–07.
3. **Evaluation (M9, E6 expanded):** a held-out test set of 30–50 generated maps (unseen
   seeds) plus MAP08. The headline result is per-class recall on unseen maps. Random defect
   placement also shows the oracles have no map-specific assertions (objective O2).
   Evaluation is CPU-only, so it costs no GPU quota.
4. Existing guards stay: no absolute position in the observation, agents cannot import
   ground truth, and the E5 absolute-position ablation shows what happens without them.

## 2026-10-24 — M3

### D20. Defects mutate the compiled map; how each one is built
PLAN calls each defect "a spec -> spec mutation", but five of the seven live below the
MapSpec's abstraction (linedef flags, special arguments, TIDs). `defects.py` therefore
mutates the compiled draft after rooms, portals and switches are laid out and before
linedefs are built, and returns a record for the ground truth. Constructions:
- `missing_blocking`: resolves open issue 6. The wall segment becomes a **two-sided,
  non-blocking line drawn with a wall texture**, backed by a 128 mu pocket sector outside
  every room. The player walks "through the wall" into space no room claims. Params:
  `wall`, `segment` [lo, hi] in absolute map units, optional `depth`.
- `wrong_tag`: the switch's `Door_Open` arg0 becomes a tag no sector carries.
- `wrong_special`: `Door_LockedRaise` arg3 (lock) becomes 0 on the door's faces.
- `geometry_trap`: the room's floor drops by `drop` (> 24); rejected if the room has a lift
  or teleport, since then it is no trap.
- `one_way_lift`: the lift's faces bordering the lower room lose their trigger.
- `teleport_loop`: param `partner`. Each teleport lands the player 56 mu in front of the
  other's pad, facing it, so walking on bounces between the two rooms.
- `bad_pickup`: the medikit becomes a Soulsphere (2013): health 200.

### D21. MAP02's 48 mu alcove is escapable by jumping (plan error)
The plan sizes the alcove trap at 48 mu, "above the 24 mu auto-step height". With JUMP in
the action set (on purpose, per the plan), a jump (~30 mu) plus the step-up (24 mu) clears
about 54 mu. Verified in ViZDoom: the player jumps out of a 48 mu alcove but stays trapped
at 64 and 96. The alcove drop is now 64 mu.

### D22. Ground truth describes intent, plus the defect list
`build/MAPnn.groundtruth.json` (`MAPnn.fixed.groundtruth.json` for the fixed variant) is
generated from the spec compiled **without** defects: rooms, the walkable raster at
`cell_size`, intended walls, intended sectors with neighbours and lift triggers, a
point-to-sector lookup grid, portals and the portal graph, things, monster count and goal.
Only `defects` differs between variants (a test enforces this). Each defect carries
`position` (x, y, z), `region` (rects it occupies or cuts off), `affected_rooms` and a
one-line `change`. The buggy geometry never appears; oracles must see it through the
engine and compare it against intent.

### D23. Corpus: 12 defects on MAP02–07 as planned, plus 3 on the holdout
MAP08 gets `missing_blocking`, `wrong_special` and `bad_pickup`, so the generalisation
evaluation (E6, D18) has defects to find. MAP01 stays defect-free.

### D24. M3 accepted 2026-10-24
`tests/test_defects.py`: each defect, applied alone, changes exactly its element; the
fixed variant of every map passes `check_reachable()`; ground truth checks.
`tests/test_defects_engine.py`: all 15 defects show in the buggy build and none in the
fixed build, using one generic scenario per defect kind; every exit still works.
220 tests total, `mypy --strict qatester/mapgen/` clean.

## 2026-10-24 — M4

### D25. Oracle layer: how each family deviates from or sharpens the plan
Protocol as planned: `reset(episode)`, `observe(step)`, `finalize()`. One set of oracles
per map per run (Families 4 and 6 accumulate across episodes). Oracles get
`GroundTruth.intent_only()`; `level.Level` refuses anything with a defect list, and a
test checks oracle code never mentions defects (resolves open issue 9).
- **Crash (1):** as planned, plus `tic_stall` (game time frozen for 3 steps).
- **Stuck (2):** *wedge* counts escape attempts in **world** directions (N/E/S/W from
  commanded move + facing), and only **sustained** pushes count: a direction held for
  >= 16 tics with < 8 mu progress. Found by recording real random play: a jittery agent
  turning in a corner faces 3–4 directions for a tic each and was flagged; the plan's
  agent-frame count misses a wedged agent that turns while pushing. *Vertical trap*
  compares the engine's live floors with the intended neighbours and lifts (resolves
  open issue 10).
- **Out of bounds (3):** the three planned tests. An excursion is reported once, at its
  end, with exit point and depth. Wall traversal excuses moves that start on a teleport
  pad (teleports are Family 6's) and uses a 32 mu/tic speed limit.
- **Unreachable (4):** works on connected components of unvisited room *sectors*, so
  `union_with` rooms are one space (bug found on MAP08). "Pressed USE >= 5 times" is
  replaced by >= 5 **proper attempts that failed** (`level.AttemptTracker`): the USE ray
  must actually reach the door (Doom's 64 mu use range), a locked door needs its key
  held, a switch door counts presses of its switch, and the door or lift must then not
  move within 35 tics. Only a traversal *after* an attempt clears it: a one-way lift was
  traversed on the way down. Without these rules, random play raised false alarms on
  clean MAP02 and MAP07.
- **Invariants (5):** `z_in_sector` dropped (duplicates Family 3's z_clip); `lift_returns`
  dropped (linedef triggers are not observable, and Family 6 covers it). `had_powerup` is
  true only for spec-declared powerups, none in the corpus, so the Soulsphere still breaks
  `health_cap` (resolves open issue 8).
- **Softlock (6):** *confined* needs a timed-out episode whose final stay of >= 40% is in
  one non-goal room where **every exit had a proper attempt that failed**. The plan's
  version flagged random agents idling in the spawn room (9 false alarms in the first
  soak). *Teleport loop* = >= 4 alternating teleport hops each within 70 tics; the plan's
  position autocorrelation is near 1 for any smooth walk.
- **Matching rule** for scoring (resolves open issue 7 for M4's tests): same family, and
  within 128 mu of the defect's position **or** inside one of its region rectangles. The
  final M9 evaluation uses the same rule.

### D26. Process notes for the write-up
- The oracles were written before their trace fixtures; the plan asks for fixtures
  first. Mitigation: beyond the synthetic fixtures, the oracles were validated on
  gameplay recorded from ViZDoom (`tests/test_oracles_engine.py`) and on random-play
  soaks: 2 x 5 seeds x 6000 tics on every fixed map, **0 findings**. The same busy random
  policy on the buggy maps produced 25 findings, **all on injected defects**, covering 7
  of 15 defects (none on MAP07, whose defects sit behind the key task).
- Defects and oracles were written by the same author (Claude), against the plan's
  intent that two people do this. The plan's thresholds are kept where they work; every
  added threshold is listed below. A teammate review of this block is recommended
  before E1.
- ViZDoom exposes no game variable for keys held. M5's environment must derive keys from
  pickup notifications or disappearing key actors.

### D27. Frozen oracle thresholds (2026-10-24, end of M4)
Copied from `OracleConfig()`; `tests/test_oracles.py` fails if code and block disagree.
Never edit in response to a result. A genuine change is a new, named experiment, re-run
from scratch, with both versions reported. Additions beyond PLAN Appendix B: `crash.*`,
`stuck.push_tics`, `stuck.push_progress_mu`, `stuck.vertical_*`,
`bounds.player_height_mu`, `bounds.max_speed_mu_per_tic`, `reach.confidence_*`,
`softlock.min_loop_hops`, `softlock.teleport_jump_mu`, `softlock.*confidence`, `attempts.*`.
Removed: `softlock.autocorr_threshold` (D25).

<!-- frozen-oracle-thresholds -->
```yaml
oracles:
  crash:
    step_timeout_s: 30.0
    stall_steps: 3
  stuck:
    window_tics: 70
    intent_threshold: 0.8
    displacement_mu: 24.0
    min_escape_directions: 3
    push_tics: 16
    push_progress_mu: 8.0
    conf_saturation_tics: 350
    vertical_drop_mu: 24.0
    vertical_confidence: 0.9
  bounds:
    player_radius_mu: 16.0
    z_tolerance_mu: 8.0
    player_height_mu: 56.0
    max_speed_mu_per_tic: 32.0
  reach:
    use_attempts_for_broken_trigger: 5
    confidence_graph: 0.95
    confidence_trigger: 0.85
  softlock:
    tail_fraction: 0.4
    max_lag_tics: 70
    min_loop_hops: 4
    teleport_jump_mu: 200.0
    confidence: 0.8
    repeat_confidence: 0.95
    loop_confidence: 0.9
  attempts:
    use_range_mu: 64.0
    window_tics: 35
    throat_mu: 32.0
    throat_push_tics: 16
    stall_mu_per_tic: 1.0
    face_cone_deg: 45.0
    player_radius_mu: 16.0
```

### D28. M4 accepted 2026-10-24
`tests/test_oracles.py`: zero findings on the MAP01 CLEAN corpus (7 traces, alone and as
one run) and on idling at spawn on all 8 fixed maps; every family fires on its positives
with the expected kinds only; every near-miss is silent; all 15 injected defects are
detected by their expected family in hand-authored positive traces.
`tests/test_oracles_engine.py`: silent on recorded real play of all 8 fixed maps; every
defect found in recorded real play by its expected family, with no stray family.

## 2026-10-24 — M5

### D29. QADoomEnv: what the plan leaves open, and where it deviates
- **Keys held.** ViZDoom has no game variable for keys. A key, or any pickup, counts as
  taken once no pickup object stands within 24 mu of its spawn point (`env/world.py`).
  This also holds after `load_state`, which parsing "Red passcard secured!" would not.
  Notifications still go on the trace.
- **Actions.** The plan's 12 combinations, with binary turn buttons. A held turn gives 7°
  on the first step, then about 14° per step. There is no turn in place, as the plan
  specifies: watch for this in M6. A tenth button, `TURN_LEFT_RIGHT_DELTA`, exists only
  for the test-only `warp` reset option, which needs `cheats=True`. No action sets it,
  and training never enables cheats.
- **"The first time each …".** Entity, portal and goal-precondition rewards are paid
  once per episode, not once per run: a per-run first would stop paying early in
  training. The oracle reward is paid once per (family, cluster) per episode. A cluster
  is the Bug Book matching rule: same family, within 96 mu, and the same invariant name.
  The Bug Book term exists but stays 0 until M7.
- **Goal preconditions.** The required keys, plus the "gates": non-open portals that
  every spawn-to-goal route must pass (critical portals in the intended portal graph).
  A gate counts as met when it is traversed.
- **Novelty churn.** Entering a cell disarms it. It re-arms once the agent is at least 2
  cells away (Chebyshev distance). An abstract state (cell, keys, open doors) is counted
  on a counted cell entry or on a key or door change. A door counts as open when its
  sector's live ceiling is more than 8 mu above its floor.
- **Shared counts.** One `multiprocessing.shared_memory` block per run. Increments are
  not atomic across workers; a lost count only makes novelty very slightly larger.
  Abstract states hash into a table of 2^20 buckets. `coverage=None` turns novelty off,
  which makes the env deterministic for a given seed.
- **Observation details.** Slot 6 is displacement / (7.5 mu/tic × tics), and 1.0 when
  no movement is commanded. Slot 16 counts pickup classes in the labels buffer. Slot 17
  uses the intended level's doors, lifts and switches with Doom's 64 mu USE ray. The env
  reads intent; the agent code never does.
- **Oracles in the loop.** Families 1, 2, 3 and 5 stream: they are paid on the step
  they fire and put on the queue. Families 4 and 6 judge across episodes, so their
  findings come from `finalize_oracles()`, which `close()` calls. They are not rewarded,
  because no step is left to pay them on. An episode cut short by `reset` or
  `load_state` is closed for the oracles with a final `done` copy of its last record.
  That way an excursion or a wedge still in progress is still reported.
- **Demos.** The plan re-runs an episode that found something, with recording on. We
  record every episode instead (about 65 KB per full episode). This is simpler, and it
  does not depend on re-runs being deterministic. A `.lmp` only replays from the map
  start, so episodes started by `load_state` keep no demo. Parquet traces are kept per
  episode: for episodes with a streamed finding by default (`trace_policy="findings"`),
  or for all episodes (`"all"`, for evaluation).
- **Save lands 2 tics late.** `DoomGame.save` takes effect 2 tics after the call, and
  the state is not refreshed until the next action. A save made while the player is
  moving therefore captures a point up to about 16 mu past the last record (the test
  bounds the drift). M8 must take a cell's position from the state after `load_state`,
  not from the record at save time.
- **Supervisor.** Engine exceptions become a `crash` finding. A watchdog thread kills
  the engine's child process once a call passes `crash.step_timeout_s` (30 s), which
  becomes a `hang` finding. Either way the episode is truncated and the engine restarts.
  Init is retried 5 times, with backoff 2, 4, 8 and 16 s. A crash is paid like any other
  finding.
- **Gymnasium checker.** It runs with `coverage=None`, because shared novelty depends on
  history by design. The episode counter is `env.episode`, not part of `info`: the
  checker requires the same info for the same seed.
- **Goal kinds.** Only `reach_exit` is supported. The corpus uses nothing else, and the
  env raises an error on any other kind.

### D30. M5 accepted 2026-10-25
`tests/test_env_contract.py` covers Gymnasium's `check_env`, spaces, keys, the USE flag,
the exit, live oracles with a trace and a demo, demo replay, determinism, save/load,
crash and hang. `tests/test_reward_hacking.py` covers the three scripted adversarial
policies, each held under its bound. `tests/test_env_units.py` covers the parts that need
no engine. All of them pass locally (Python 3.12, gymnasium 1.4.0) and on Kaggle (Python
3.13.15, gymnasium 1.2.0, vizdoom 1.3.2): 30 passed, including the leak tests.

`scripts/bench_env.py` on a Kaggle T4 x2 session (4 vCPU, Intel Xeon @ 2.00GHz), MAP07
buggy, random actions, all oracles live, 20 s per row:

| workers | steps/s per worker | total steps/s | share of step in the engine |
|---|---|---|---|
| bare ViZDoom, 1 | 1486.4 | — | — |
| 1 | 912.6 | 912.6 | 54.9% |
| 2 | 739.6 | 1479.1 | 55.9% |
| 4 | 614.5 | 2458.1 | 53.0% |
| 16 | 147.0 | 2351.7 | 86.9% |

The target of more than 400 steps/s with 1 worker is met (912.6). Throughput saturates
at 4 workers, one per vCPU; 16 workers only oversubscribe the CPUs. The plan's budget
assumes about 300 steps/s per job, with 2 jobs of 2 workers each. The env alone gives
about 1,200 steps/s per job in that layout. PPO updates will take some of that, and M6
measures the real training rate.

## 2026-10-25 — Procedural maps (D18, item 1)

### D31. `qatester/mapgen/procgen.py`: how generated maps are made
- **Seeds and splits.** A seed fixes a map completely. Train seeds are 0, 1, 2, …;
  test seeds start at 1,000,000, so the two splits never share a map. A pack is at most
  32 maps (one WAD, MAP01–MAP32): `python -m qatester.mapgen.procgen --split train
  --count 32 -o <dir>`, then the usual `qatester.mapgen.cli build`.
- **Defect classes come first.** 0–3 distinct kinds are drawn uniformly from the 7. The
  layout is then made to contain what each kind needs: a pit behind a lift, a teleport
  pair, a switch door, a locked door, a health pickup, a spare dead end, an outer wall.
  Every class is therefore about equally common: each appears on about 1.5/7 of maps,
  and about a quarter of maps are clean.
- **Layout.** 5–9 rooms grown on a 4×4 grid. Column widths and row heights are drawn
  from 320–512 mu, and neighbouring rooms share a full wall. A random spanning tree from
  the spawn room gives the portals; up to 2 extra edges add loops. Some leaves become
  pits (floor −64) or balconies (+64), reachable only by lift, so no open portal ever
  joins two floors more than a step apart. Tree edges are open 50%, door 30%, locked
  door 10% (at most one per key colour) and switch door 10%.
- **Placement rules that keep maps playable.**
  - Keys and switches go, shallowest door first, in a floor-0 room reachable without
    their own door and without any door not yet unlocked.
  - The exit goes in the deepest leaf.
  - Teleport pads sit 112 mu in from a corner of a room at least 384 mu across. That
    keeps them clear of lift platforms, and 80+ mu from the room centre where arrivals
    land, so a clean map never bounces the player straight back.
  - Wall switches go 40 mu inside a wall with no portal, spaced 128 mu apart when two
    share a wall.
  - Pickups stay 96 mu clear of walls and 80 mu clear of pads.
- **Defects stay findable and separate.**
  - The `wrong_tag` door is chosen so that what it cuts off holds nothing the map needs
    (spawn, exit, keys, switches). No other defect goes there.
  - `geometry_trap` uses a dead end reserved for it: no lift, no teleport, no pickups,
    no switch door in front.
  - `one_way_lift` only targets lifts down into pits. A one-way lift up to a balcony
    would make a region unreachable, which is a different class.
  - `missing_blocking` uses an outer wall that no switch uses.
  - Two defects never share a room.
- **Every map is checked as it is made.** The spec validators, both compiled variants
  through the geometry validators, and ground-truth generation all run inside
  `generate`. On failure the generator retries with seed `[seed, attempt]`, which keeps
  it deterministic. On average a map takes 1.2 attempts.

### D32. Generated maps validated in the engine (2026-10-25)
`tests/test_procgen.py` covers a pack of 24 generated maps (12 train, 12 test seeds):
- On the fixed variant, every M2 portal routine works (doors, locked doors, switch
  doors, lifts both ways, teleports, the exit), and the oracles stay silent on that play
  and on random play.
- On the buggy variant, each defect's scenario shows the defect, and the oracles find it
  with its expected family. The only other findings allowed are matches to another real
  defect of the same map.

A one-off sweep of 96 more maps (train seeds 100–147, test seeds 1,000,100–1,000,147):
all 96 play as intended with zero findings, and all 141 defects are found by their
expected family.

The sweep exposed a bug in the test helper `doomdriver.approach`. It decided which side
of a wall the room is on by checking that the wall sits on the room's edge. A switch on
a wall shared with a neighbouring room (no portal there) sits WALL/2 inside the edge, so
the test player was placed outside the room. The helper now compares the wall with the
room's centre. The corpus only had switches on outer walls, so the bug never showed
there. The env and the oracles take switch positions from the ground truth and were
never affected.

## Open issues found in PLAN.md

Each must be decided before the milestone named.

1. **[M6/M7] RecurrentPPO cannot express the specified network.** The plan specifies a GRU
   trunk and two value heads (extrinsic γ=0.995, intrinsic γ=0.99), and says this is "a
   configuration change, not an implementation" with sb3-contrib's `RecurrentPPO`. It is not:
   `RecurrentPPO` supports only LSTM and a single value head with a single γ. Options:
   (a) use `RecurrentPPO` + `MultiInputLstmPolicy` with LSTM and one value head on the
   combined reward (simplest, deviates from the architecture section); or (b) write a small
   custom PPO with GRU and dual heads (matches the spec, more code to test).
3. **[M8] Archive cap.** The M8 prompt says LRU eviction at 20,000 cells; the Kaggle rules
   and `t4_goexplore.yaml` say 5,000. Proposal: configurable `max_cells`, default 5,000.
4. **[M9] Seed count.** "Do not cut … the 10-seed protocol" and R10 ("keep 10 for E1 and
   E2") conflict with the Kaggle budget table (5 seeds for E1/E2, 3 for ablations), which
   the plan adopts elsewhere. Proposal: follow the budget table and state it in the report.
5. **[M6] Report example uses 2M steps.** The `report.json` example has `env_steps: 2000000`
   while runs are 1M. Illustrative only, so no action needed.

## Resolved issues

6. **[M3] `missing_blocking` as specified injects nothing:** fixed by D20 (passable fake wall into a pocket sector).
7. **[M4/M9] Matching rule too narrow for room-sized defects:** D25. Within 128 mu of the position **or** inside the defect's region.
8. **[M4] `health_cap` would excuse the Soulsphere:** D25. Only spec-declared powerups excuse it.
9. **[M4] Oracles must not read `defects`:** D25. Intent-only view, enforced by tests.
10. **[M4] Vertical-trap check needs live floors:** D25. Live engine floors against the intended topology.
11. **[M5] Worker counts for the bench:** `bench_env.py` runs 1, 2, 4 and 16 workers. Treat 16 as indicative only on Kaggle's 4 vCPUs; training uses 2 per job.
