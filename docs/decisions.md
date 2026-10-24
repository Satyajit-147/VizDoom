# Decisions and deviations from PLAN.md

Every deviation from `docs/PLAN.md` is recorded here with its date. Newest last.

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

## Open issues found in PLAN.md (not yet resolved)

These are internal contradictions or wrong assumptions in the plan. Each must be decided
before the milestone named.

1. **[M6/M7] RecurrentPPO cannot express the specified network.** The plan specifies a GRU
   trunk and two value heads (extrinsic γ=0.995, intrinsic γ=0.99), and says this is "a
   configuration change, not an implementation" with sb3-contrib's `RecurrentPPO`. It is not:
   `RecurrentPPO` supports only LSTM and a single value head with a single γ. Options:
   (a) use `RecurrentPPO` + `MultiInputLstmPolicy` with LSTM and one value head on the
   combined reward (simplest, deviates from the architecture section); or (b) write a small
   custom PPO with GRU and dual heads (matches the spec, more code to test).
2. **[M5/M6] Worker counts.** The `train.py` sketch uses `n_envs=16`, the config and Kaggle
   budget use `n_envs=2`, and the M5 prompt asks for benchmarks at 1, 4 and 16 workers.
   Kaggle has 4 vCPUs, so the 16-worker bench is meaningless there. Proposal: bench 1, 2
   and 4 on Kaggle.
3. **[M8] Archive cap.** The M8 prompt says LRU eviction at 20,000 cells; the Kaggle rules
   and `t4_goexplore.yaml` say 5,000. Proposal: configurable `max_cells`, default 5,000.
4. **[M9] Seed count.** "Do not cut … the 10-seed protocol" and R10 ("keep 10 for E1 and
   E2") conflict with the Kaggle budget table (5 seeds for E1/E2, 3 for ablations), which
   the plan adopts elsewhere. Proposal: follow the budget table and state it in the report.
5. **[M6] Report example uses 2M steps.** The `report.json` example has `env_steps: 2000000`
   while runs are 1M. Illustrative only, so no action needed.

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
6. **[M3] `missing_blocking` as specified injects nothing.** Appendix A's mutation clears
   `blocking` on a one-sided wall. Verified on MAP03's hall (2026-10-24): the player still
   stops at the wall, because GZDoom treats every one-sided line as solid. A real
   missing-collision defect needs the wall segment emitted as a **two-sided, non-blocking
   line with a wall texture**, backed by a hidden pocket sector outside every room
   rectangle. The player walks "through the wall" into space no room claims, which is what
   the out-of-bounds oracle tests. To decide at the start of M3.

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
