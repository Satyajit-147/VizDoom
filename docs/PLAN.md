# RL-Based QA Tester for Games — Phase 1 Implementation Plan (ViZDoom)

## How to use this document

This is a build specification, not a proposal. Every section is written so a coding agent can
act on it without asking follow-up questions: paths, file names, function signatures,
numeric thresholds and acceptance tests are all stated explicitly.

Conventions used throughout.

   Marker                      Meaning

   MUST                        A hard requirement. Changing it breaks a
                               downstream component or an evaluation claim.

   SHOULD                      A strong default. Deviate only with a recorded
                               reason in docs/decisions.md .

   MAY                         Genuinely free choice.

   TODO(human)                 A step a person must do by hand (e.g. obtaining an
                               IWAD). Claude Code must stop and ask.

Units. All in-world distances are Doom map units (mu). For calibration: player radius 16 mu,
player height 56 mu, standard door width 64–128 mu, maximum step-up height 24 mu,
maximum drop without damage 24 mu for an unforced step but unlimited for a fall. One tic
= 1/35 s of game time.

Repository root. Everything is written relative to a repository named rl-qa-tester/ .
The Python package is qatester .

Order of work. Follow the milestones in the build-order section. Each milestone has an
acceptance test that MUST pass before the next begins. Do not start the RL agent before
the oracle layer passes M4; a tester whose oracles are wrong produces results that
cannot be interpreted.

What is deliberately out of scope here. Generalisation to other engines, the adversarial
level generator, multi-agent testers, and LLM components. Those are Phase 2 and later;
the final section lists the seams left open for them.

## Phase 1 scope and objectives

Phase 1 delivers a working, single-agent RL quality-assurance tester for one game —
Doom, driven through ViZDoom — that autonomously explores purpose-built levels,
detects injected defects using automatic oracles, and emits a deduplicated, ranked bug
report. It is evaluated against a labelled ground truth so that precision, recall and cost are
real numbers rather than anecdotes.

### Why Doom via ViZDoom is the right Phase-1 game

The survey's own complexity framework (Li et al., five dimensions) places first-person
games in the middle band: partial observability, moderate state space, discrete action
space, intermediate reward sparsity, single agent. That is exactly the band where RL
testing is non-trivial but tractable on student hardware. Concretely, ViZDoom gives four
things no other easy environment gives at once:

 1. A real commercial-lineage engine with real bug classes. Collision, reachability,
    softlock, trigger and sequence-break bugs occur in Doom for the same engine-level
    reasons they occur in shipped games, so findings transfer conceptually.
 2. Privileged ground-truth access. set_sectors_info_enabled(True) exposes map
     geometry, set_objects_info_enabled(True) exposes every actor, and precise player
      POSITION_X/Y/Z and VELOCITY_X/Y/Z are game variables. This is what makes
     automatic oracles possible — it is the ViZDoom analogue of the navigation mesh Lu et
     al. used as ground truth.
 3. State save and restore. DoomGame.save(path) / DoomGame.load(path) (available
     since 1.1.9) satisfy the hard precondition for Go-Explore-style archive-and-reset, which
     Lu et al. list as an assumption their method requires.
 4. Throughput. Up to ~7000 fps single-threaded in synchronous mode with rendering
    stripped down. Wuji needed 1,260 machine-hours; we must stay inside a few hundred
    GPU-hours at most, and ViZDoom makes that possible.

### Objectives (each maps to a measurable deliverable)

   #          Objective                                       Deliverable                   Pass criterion

   O1         Reproducible,                                    qatester.mapgen compiling    7 maps build and load
              programmatically                                a MapSpec to a playable       in ViZDoom; clean map
              generated buggy levels                          UDMF WAD plus a ground-       has zero ground-truth
                                                              truth JSON                    defects

   O2         Automatic, game-agnostic                         qatester.oracles with 6      ≥ 5 of 7 injected defect
              bug oracles                                     oracle families               classes detected
                                                                                            without any map-
                                                                                            specific assertion

   O3         An exploration-first RL                          qatester.agents PPO +        at least 80% spatial
              tester                                          count-based curiosity + Go-   cell coverage on the
                                                              Explore resets                large map within 1M
                                                                                            env steps

   O4         Demonstrate winning ≠                           Task-reward-only PPO          Task-only PPO finds
              testing                                         baseline                      strictly fewer defects
                                                                                            than the curiosity
                                                                                            tester, with a
                                                                                            significance test

   O5         Bug Book experience                              qatester.bugbook archive     Ablating it measurably
              replay (Code-A1 transfer)                       with frequency-decay          reduces regression-
                                                              update                        catch rate on re-tested
                                                                                            builds

   O6         Usable output, not raw                          Deduplicated, severity-       < 1.5 reports per true
              logs                                            ranked, calibrated HTML +     defect after clustering;
                                                              JSON report                   ECE < 0.1 after
                                                                                            calibration

   O7         Honest cost accounting                           results/cost.json            Wall-clock, env steps
                                                                                            and GPU-hours
                                                                                            recorded for every run

### Explicit non-goals for Phase 1

       No level generator and no Builder–Tester loop. Maps are hand-specified; the generator
       is Phase 2.

     No multi-agent tester. cMarlTest -style passive agents are Phase 3.
     No engine generalisation layer. The oracle interface is designed to be portable, but only
     a ViZDoom backend is implemented.
     No LLM in the loop. The Code-A1 contribution we borrow is the architecture (two
     opposing objectives, experience replay of past failures), not its language models.
     No pixel-only constraint. The policy sees depth and scalars; using privileged
     coordinates for oracles and archives is allowed and is the point.

## Grounding in the literature

Every non-obvious design decision below traces to a specific finding in the surveyed
work. This table is the justification section of the final report; keep it current as the design
evolves.

   Source                           Finding we rely on                Phase-1 design decision

   Zheng et al., Wuji               A reward-maximising agent         Two mandatory baselines: task-only
                                    follows the main path and never   PPO and random. We report state
                                    reaches off-path bugs; line       and cell coverage, never line
                                    coverage was identical            coverage. The tester optimises
                                    (98/73/66%) across strategies     exploration and task progress, not
                                    while bug counts differed 16x     either alone.

   Zheng et al., Wuji               Crash and stuck oracles are       Oracle families 1-3 are automatic;
                                    cheap and general; logical bugs   family 5 is a declarative invariant
                                    need assertions                   DSL so assertions are written once
                                                                      per property, not once per map

   Zheng et al., Wuji               Pareto selection over             Tier-5 agent keeps a population of 8
                                    (exploration score, winning       policies selected by non-dominated
                                    score) beats either fitness       sorting on (coverage, task progress,
                                    alone                             oracle hits)

   Bergdahl et al.                  RL agents find exploits such as   The BUG_LEAK map has a linedef
                                    a missing collision mesh that     with its blocking flag cleared; the
                                    scripted navigation agents        exploit oracle is the out-of-bounds
                                    structurally cannot               test

   Source                           Finding we rely on                   Phase-1 design decision

   Lu et al., Go-                   Archive-and-reset with               Tier-4 agent is Go-Explore over
   Explore 3D                       heuristic reset selection beats      DoomGame.save/load. Authored
                                    RND and uniform resetting; the       room polygons replace the nav-
                                    nav-mesh is ground truth for         mesh as ground truth. Reset
                                    unexpected positions; 16             selection uses the same three
                                    parallel agents covered 1.5 km2      heuristics: visit count, distance to
                                    in about 10 h                        nearest undiscovered cell, and a
                                                                         weighted mix.

   Ferdous et al.,                  Curiosity reward doubled entity      Tier-3 reward includes a count-
   RLbT                             coverage (80% vs 40%);               based novelty term over an entity-
                                    abstracting state to entities-       state abstraction (which doors are
                                    and-properties keeps the space       open, which keys held), not only
                                    tractable; exploration logs          position
                                    revealed two design flaws

   Ferdous et al.,                  Partial observability makes a        Noted as a Phase-3 seam; Phase 1
   cMarlTest                        single agent waste time in           runs independent parallel workers
                                    visited regions                      sharing one archive, which captures
                                                                         most of the wall-clock benefit
                                                                         without MARL

   Mastain and                      Narrow BDD scenarios keep the        The task-progress reward is derived
   Petrillo                         RL problem small; defining the       from the MapSpec automatically
                                    reward function was the              (reach the exit, collect required
                                    hardest, most iterative step         keys), so there is no hand-tuned
                                                                         reward per map. Scenario-style
                                                                         assertions live in the invariant DSL.

   Gisslen et al.,                  A generator rewarded purely for      Not used in Phase 1, but reward
   ARLPCG                           solver failure converges to          shaping carries an alpha trade-off
                                    impossible levels unless the         weight so the Phase-2 generator
                                    solver adapts; an auxiliary scalar   plugs into the same interface
                                    controls difficulty

   Beukman et al.,                  Regret-based level choice            The archive reset heuristic down-
   ReMiDi                           stagnates on levels with high        weights cells reset to N or more
                                    irreducible regret                   times without producing new
                                                                         coverage, the Phase-1 analogue of
                                                                         rejecting unlearnable cases

   Source                           Finding we rely on                 Phase-1 design decision

   Earle et al.,                    Agents with small local            The policy never sees absolute
   PCGRL+                           observations generalise better     coordinates. Observation is a 64x64
                                    than global-observation agents,    depth crop plus egocentric scalars.
                                    which exploit border tiles to      Absolute position is used only by
                                    infer coordinates                  oracles, archives and metrics.

   Earle et al.,                    GPU-parallel envs made 1e9-        SubprocVecEnv ViZDoom workers
   PCGRL+                           step training feasible             at 160x120 with rendering features
                                                                       disabled. Worker count follows the
                                                                       host's CPU cores: 2 per job on
                                                                       Kaggle's 4-vCPU session, 8-16 on a
                                                                       larger machine.

   Yuan et al.,                     Combining gradient-based RL        Tier-5 combines PPO gradients with
   MOPCGRL                          with gradient-free evolutionary    population-level evolutionary
                                    search explores better than        selection
                                    either alone

   Huang et al.,                    Self-improving agents become       Every bug report carries a
   Calibration                      systematically overconfident       confidence; confidences are
                                    and ECE rises; calibration after   temperature-scaled on a held-out
                                    self-improvement is the            labelled split and we report ECE and
                                    effective remedy                   a reliability diagram

   Li et al., MARL                  Five complexity dimensions;        Justifies the FPS complexity band,
   review                           CTDE dominates; OpenAI Five        caps the compute budget as a first-
                                    needed 128k CPU cores, putting     class objective (O7), and scopes
                                    advanced RL out of reach of        multi-agent work out of Phase 1
                                    small studios

   Ahmad and                        AI-assisted building has no        Motivates Phase 2; Phase 1 emits
   Mubin                            autonomous test loop               bug reports in exactly the schema
                                                                       the Phase-2 builder will consume

### The Code-A1 transfer, stated precisely

Code-A1 is the closest architectural analogue to our project, so the mapping is worth
spelling out: it is the clearest novelty claim the project can make. Their framework
optimises a Code LLM and a Test LLM with opposing objectives, stabilised by a Mistake

Book of historically failed tests and a composite reward balancing validity against
adversarial difficulty.

   Code-A1 component                                  Our analogue                      Phase 1 status

   Code LLM, producing the                            Level generator / game            Phase 2. In Phase 1 the
   artefact under test                                builder                           builder is a fixed set of 7
                                                                                        authored maps.

   Test LLM, producing                                RL tester agent                   Built in Phase 1
   adversarial probes

   White-box access, tester                           Tester reads privileged           Built in Phase 1. This is our
   inspects candidate code                            geometry and object state         white-box channel, and the
                                                      from ViZDoom                      decoupled architecture is
                                                                                        what makes it safe.

   Self-collusion risk under a                        A single model that both          Avoided structurally: tester
   unified model                                      generates and tests levels        and builder are separate
                                                      would learn to emit trivially     components from day one
                                                      testable levels

   Mistake Book: per-question                         Bug Book: per-map archive         Built in Phase 1
   archive of failed tests with a                     of states, action prefixes and
   frequency counter,                                 oracle triggers that
   incremented on new fails,                          previously exposed
   decremented on passes,                             suspicious behaviour, with
   removed at zero                                    exactly the same frequency
                                                      rule

   Composite reward R =                               Tester reward R =                 Built in Phase 1, with alpha
   alpha*Valid + (1-alpha)*Adv,                       alpha*R_valid + (1-               swept over {0, 0.25, 0.5,
   with alpha = 0.5 best                              alpha)*R_adv, where R_valid       0.75, 1.0} as an ablation
                                                      is legitimate-play plausibility   mirroring their Figure 3
                                                      and R_adv is oracle-trigger
                                                      plus novelty

   Experience replay prevents                         Replaying Bug Book                Built in Phase 1
   forgetting of resolved bugs                        trajectories on a rebuilt map
                                                      is our regression oracle

The insight being ported is their central one: effective verifiable rewards require a
dynamic interplay between the artefact's robustness and the tester's rigour. Phase 1
builds the tester half and the replay machinery; Phase 2 closes the loop.

One caution carried over from Beukman et al. and from Code-A1's own alpha ablation: a
tester rewarded only for failure (alpha = 0) becomes unstable and starts producing
nonsense probes. Our validity term is what keeps the tester honest, and the alpha sweep
is how we demonstrate it.

## System architecture

The governing principle is single source of truth: one MapSpec file per level is compiled
into the playable WAD, the ViZDoom config, and the ground-truth JSON the oracles check
against. Nothing about a level is written down twice, so the oracle can never silently
disagree with the level the agent is actually playing. This is the ViZDoom replacement for
the navigation mesh Lu et al. relied on, and it is stronger: we know not just where the
player should be able to stand, but which doors should open, which keys gate what, and
exactly which defect was injected where.

Phase 1 architecture - 4 bands, 12 components, 1 feedback loop

The only cycle in the system is the dashed one: the Bug Book feeds historically interesting
states back to the agent as reset targets and as a reward term, which is the Code-A1
Mistake Book mechanism transplanted into a spatial environment.

### Layer responsibilities and boundaries

   Layer                    Owns                                     Must not know about

   Authoring                Level semantics, geometry, injected      RL, rewards, ViZDoom process
                            defects                                  management

   Layer                    Owns                                     Must not know about

   Environment              ViZDoom lifecycle, observation           Which defects exist, how bugs are
                            construction, action mapping, trace      scored
                            recording

   Tester agent             Policy, exploration strategy, reset      Ground truth; it receives only
                            scheduling                               reward scalars and observations

   Analysis                 Oracles, archive, clustering,            How the policy was trained
                            calibration, reporting

The agent MUST NOT read groundtruth.json . If it could, it would learn to walk to the
injected defects directly and the whole evaluation would be meaningless. Enforce this
with a code review rule and a unit test that greps qatester/agents/ for imports of
qatester.groundtruth .

### Process topology

One training run is a single Python process holding a SubprocVecEnv of N worker
processes, where N follows the host's CPU core count: 2 on Kaggle, 8-16 on a larger
machine. Each worker owns one ViZDoom instance (itself a child process spawned by the
ViZDoom library). Workers write trace records to their own Parquet shard and push oracle
events onto a multiprocessing queue drained by the main process, which owns the shared
cell archive and the Bug Book. The archive is shared rather than per-worker because Lu et
al.'s parallel speedup depends on agents not re-exploring each other's territory.

## Environment, part A: the MapSpec schema

A MapSpec is a YAML file that describes a level in semantic terms, never in engine terms.
It is the only hand-written artefact per level. Everything else — geometry, WAD binary,
ViZDoom config, oracle ground truth — is derived.

### Design rules for the schema

 1. Rooms are axis-aligned rectangles. This is a deliberate restriction. It makes polygon
     containment a constant-time test, makes the WAD compiler simple enough to be
     correct, and loses nothing for Phase 1. Non-rectangular rooms are composed as a union
     of overlapping rectangles sharing a sector.

 2. Portals are first-class. A portal is a connection between two rooms with a type ( open ,
      door , locked_door , lift , teleport ) and an optional requires key. The portal list is
     the intended connectivity graph, which the unreachable-region oracle BFS-searches.
 3. Defects are declared, not hidden. Each injected defect names the element it corrupts
     and the mutation applied. The compiler applies the mutation to the geometry and
     simultaneously records it in groundtruth.json . A defect you cannot declare is a
     defect you cannot score.
 4. A spec with an empty defects list MUST compile to a clean, fully playable, fully
    reachable level. This is the false-positive control and is tested in CI.

### Schema

Validate with a Pydantic v2 model in qatester/mapgen/spec.py . Field names are
normative.

   Field                  Type                       Notes

    name                  str                        Becomes the map lump name, uppercased, e.g.
                                                     MAP03 via slot

    slot                  int 1-32                   Map slot inside the WAD

    cell_size             int, default 64            Coverage grid resolution in map units

    rooms                 list[Room]                 At least one

    portals               list[Portal]               May be empty for a single-room map

    things                list[Thing]                Player start is mandatory and unique

    goal                  Goal                       Exit condition and its preconditions

    defects               list[Defect]               Empty for the clean control map

    seed                  int                        Used for any randomised cosmetic choices so
                                                     builds are reproducible

Room — id: str , x0,y0,x1,y1: int (map units, x1 > x0), floor: int (height),
 ceiling: int , tag: int | null (sector tag for specials), damage: int default 0,
 light: int default 160.

Portal — id: str , a: room_id , b: room_id , kind: open | door | locked_door |
lift | teleport , width: int default 96, requires: key_id | null , at: [x, y] |
null (the midpoint of the shared wall; computed automatically when the rooms are
adjacent).

Thing — id: str , type: player_start | key_red | key_blue | health | armour |
ammo | monster_imp | exit_switch | marker , room: room_id , pos: [x, y] ,
 interactive: bool (counts toward entity coverage).

Goal — kind: reach_exit | collect_all | survive , target: thing_id | null ,
 requires: list[key_id] , time_limit_tics: int .

Defect — id: str , kind: <one of the seven classes> , target: room_id |
portal_id | thing_id , params: dict , expected_oracle: str (which oracle family
should catch it), severity: low | medium | high , notes: str .

### Worked example: maps/m03_leak.yaml

   name: LEAK
   slot: 3
   cell_size: 64
   seed: 1337

   rooms:
     - {id: start,                   x0: -320, y0: -320, x1: 0,                 y1: 0,     floor: 0,   ceiling:
   128}
     - {id: hall,                    x0: 0,              y0: -224, x1: 640,     y1: -96,   floor: 0,   ceiling:
   128}
     - {id: vault,                   x0: 640,            y0: -320, x1: 960,     y1: 0,     floor: 0,   ceiling:
   128, tag: 10}
     - {id: balcony, x0: 0,                              y0: 64,     x1: 320,   y1: 256,   floor: 64, ceiling:
   192}

   portals:
       - {id: p_start_hall,                      a: start, b: hall,         kind: open,        width: 128}
       - {id: p_hall_vault,                      a: hall, b: vault,         kind: locked_door, width: 96,
   requires: key_red}
     - {id: p_start_balc,                        a: start, b: balcony, kind: lift,               width: 96}

   things:
       - {id: spawn,                   type: player_start, room: start,   pos: [-160, -160]}
       - {id: redkey,                  type: key_red,      room: balcony, pos: [160, 160],

   interactive: true}
       - {id: exitsw,                  type: exit_switch,            room: vault,   pos: [800, -160],
   interactive: true}
     - {id: medkit,   type: health,                                  room: hall,    pos: [320, -160],
   interactive: true}

   goal:
       kind: reach_exit
       target: exitsw
       requires: [key_red]
       time_limit_tics: 21000

   defects:
       - id: d_leak_hall_north
          kind: missing_blocking
          target: hall
          params: {wall: north, segment: [192, 320]}
          expected_oracle: out_of_bounds
          severity: high
          notes: >
            Blocking flag cleared on a 128 mu segment of the hall's north wall.
              The player can walk out of the level into undefined space.

### Validation rules the loader MUST enforce

     Exactly one player_start .
     Every portal.a / portal.b names an existing room; the two rooms must share a wall
     segment at least width long, unless kind: teleport .
     Every requires key names a key_* thing that exists.
     With defects removed, the portal graph must be connected and the goal reachable
     from the spawn room; spec.check_reachable() runs a BFS and raises otherwise.
     No two rooms overlap unless they are explicitly declared as a union_with pair (used
     for L-shaped spaces).
     Every defect's target exists and its kind is in the registry.

All six rules get a unit test in tests/test_spec_validation.py with one passing and one
failing fixture each.

## Environment, part B: the MapSpec to WAD compiler

This is the component that decides whether the project is reproducible or a pile of hand-
edited binaries. Build it first and test it hardest.

### Why UDMF, and why no external node builder

ViZDoom is a ZDoom derivative. The ViZDoom documentation is explicit on two points
that together make programmatic map generation easy:

     Use UDMF format for maps, because other formats do not support scripting. UDMF is
     a plain-text lump called TEXTMAP . Generating text is vastly easier and more
     debuggable than packing the binary VERTEXES / LINEDEFS / SIDEDEFS / SECTORS
     structures of the original Doom format.
     A node builder is not required. The documentation states that when an editor asks for
     a node builder you may select none, because ViZDoom has one built in. This removes
     the only genuinely hard dependency (ZDBSP/glBSP) from the pipeline.

The practical consequence: a map is a text string plus a tiny binary wrapper. The entire
compiler is a few hundred lines of pure Python with no third-party dependency. Do not
use omgifol — it is oriented at binary Doom-format maps and adds a dependency for no
gain here.

### WAD container format

A PWAD is trivially simple and you SHOULD write the 60-line writer rather than take a
library.

   Part                           Layout

   Header, 12 bytes               ASCII PWAD , then int32 little-endian lump count,
                                  then int32 little-endian directory offset

   Lump data                      Raw bytes, concatenated, in any order

   Directory                      One 16-byte entry per lump: int32 offset, int32 size,
                                  8 bytes ASCII name padded with NUL

A UDMF map inside a WAD is a run of lumps in this exact order: a map marker lump named
e.g. MAP03 with zero length, then TEXTMAP holding the UDMF text, then optionally
SCRIPTS and BEHAVIOR for ACS, then ENDMAP with zero length. Everything between the

marker and ENDMAP belongs to that map. Several maps simply follow one another in the
same WAD; Phase 1 ships one WAD, build/qa_scenarios.wad , containing MAP01 through
MAP07.

### Geometry generation

For each room rectangle the compiler emits one sector and four linedefs. Shared walls
between adjacent rooms become two-sided linedefs. The algorithm:

 1. Collect wall segments. For every room, walk its four edges and split each edge at the
    x- or y-coordinates of every other room's edges that touch it. This produces a set of
    maximal segments, each of which is either a boundary of exactly one room (one-sided)
    or shared by two rooms (two-sided).
 2. Punch portals. For each portal between adjacent rooms, find the shared segment and
    split it so that a width -long sub-segment sits at the portal midpoint. That sub-
     segment becomes the passable linedef; the rest of the shared wall stays blocking and
     is given an impassable flag.
 3. Deduplicate vertices into a dict keyed by (x, y) ; the index into the insertion-ordered
    list is the UDMF vertex index.
 4. Emit sidedefs. One-sided linedefs get a single front sidedef with texturemiddle . Two-
     sided linedefs get front and back sidedefs with texturetop and texturebottom set
     so that floor- and ceiling-height differences are not rendered as holes.
 5. Emit the linedef specials for doors, locked doors, lifts and teleports (numbers are
    listed in Appendix A).
 6. Emit things with the Doom editor numbers from the type registry, with skill1
     through skill5 and single flags set so they appear at every difficulty.

### Critical correctness invariants

These are the mistakes that produce a map which loads but is subtly broken. Assert each
one in qatester/mapgen/validate.py before writing the WAD.

     Every linedef's front sidedef exists and references a valid sector.
     A two-sided linedef MUST have the twosided flag and MUST NOT have blocking —
     except where a defect deliberately violates this.
     No two vertices are closer than 1 mu; no zero-length linedefs.

     Linedef winding is consistent: front side faces the sector on the right of the vector
     from v1 to v2. Getting this backwards makes walls invisible from one side, which looks
     exactly like a rendering bug and will waste a day.
     Every sector is closed: walking its linedefs returns to the start.
     Ceiling height exceeds floor height by at least 56 mu wherever the player must stand.
     No thing is placed within 16 mu of a wall.

### Derived outputs

From the same spec object the compiler also writes:

      build/MAPnn.cfg — the ViZDoom config, with doom_scenario_path , doom_map ,
      episode_timeout from goal.time_limit_tics , the standard button and game-
     variable lists, and the buffer flags.
      build/MAPnn.groundtruth.json — room polygons, the walkable mask rasterised at
      cell_size , the intended portal graph, the thing inventory, goal preconditions, and the
     full defect list with world positions. This is the oracle's input and the evaluation's
     answer key.
      build/MAPnn.preview.svg — a top-down render of the rooms, portals and defect
     locations. Generate it from day one; you will read it constantly while debugging
     geometry.

### Optional ACS

ACS is only needed if a defect requires in-engine logic that linedef specials cannot
express. Phase 1 is designed so that none do. If you later need it, compile SCRIPTS with
 acc into a BEHAVIOR lump and insert both between TEXTMAP and ENDMAP . Record the
 acc binary as a TODO(human) dependency; do not make the default build depend on it.

## The bug corpus: seven scenario maps

Seven maps, one WAD. Six carry injected defects of a distinct class; one is clean. Wuji's
authors spent roughly 20 person-days and over $5,000 preparing historical bugs for their
evaluation — generating them from a spec costs us an afternoon, which is the main
practical advantage of the compiler.

   Slot            Map                     Size                      Defects           Role

   MAP01           CLEAN                   4 rooms,                  none              False-positive control. Any
                                           1200x900 mu                                 report here is a false alarm.

   MAP02           WEDGE                   6 rooms +                 2 geometry        Stuck oracle
                                           alcoves                   traps

   MAP03           LEAK                    4 rooms                   1 missing         Out-of-bounds / exploit
                                                                     blocking flag     oracle

   MAP04           ORPHAN                  8 rooms                   1 mistagged       Unreachable-region oracle
                                                                     door trigger

   MAP05           PHANTOM                 5 rooms                   2 logic defects   Invariant oracle

   MAP06           SOFTLOCK                6 rooms                   2 trap states     Softlock / hang oracle

   MAP07           SPRAWL                  12 rooms,                 4 mixed defects   Scale test and the coverage
                                           1800x1400 mu                                headline number

### Defect definitions

MAP02 WEDGE — geometry_trap . Two defects. (a) A dead-end alcove 40 mu wide,
narrower than is comfortable for the 32 mu player diameter at the approach angle, with a
48 mu step at its mouth — above the 24 mu auto-step height, so the player can drop in and
cannot climb out. (b) A pit sector whose floor is 96 mu below the surrounding floor with
no lift. Compiler mutation: lower rooms[alcove].floor and raise the neighbouring floor.
Expected oracle: stuck.

MAP03 LEAK — missing_blocking . One linedef segment on the hall's outer wall has its
 blocking flag cleared while remaining one-sided. In ZDoom the player walks through it
into the void beyond the level. This is a faithful reproduction of Bergdahl et al.'s Exploit
sandbox, where a missing collision mesh let an RL agent find a shortcut a scripted agent
never could. Expected oracle: out_of_bounds.

MAP04 ORPHAN — wrong_tag . The switch intended to open the door to the east wing is
given sector tag 11 while the door's sector carries tag 10. The switch animates and does
nothing; the east wing (3 rooms, 1 key, 1 secret) is permanently unreachable. This is the
class Ferdous et al. found by inspecting learning logs — an incorrect door-button
connection. Expected oracle: unreachable.

MAP05 PHANTOM — wrong_special and bad_pickup . (a) A door declared
 locked_door requires: key_blue is emitted with lock argument 0, so it opens without
the key: a sequence-break. (b) A health pickup is given an amount that takes the player
above the 100 maximum. Both are only visible as property violations, not as anything
spatial. Expected oracle: invariant.

MAP06 SOFTLOCK — one_way_lift and teleport_loop . (a) A lift lowers the player into
a pit and has no return trigger on the lower sector: the episode can never progress. (b)
Two teleport linedefs whose destinations are each other's pads, so a player crossing one
oscillates indefinitely. Expected oracle: softlock, with the teleport loop also caught by the
stuck oracle's displacement test.

MAP07 SPRAWL — mixed. One of each of missing_blocking , wrong_tag ,
 geometry_trap and one_way_lift , placed far from the spawn behind a locked door, so
that reaching them requires completing the intermediate key task. This is Wuji's Block
Maze argument made concrete: a purely random agent cannot complete the intermediate
task and so cannot reach the defects at all, while a reward-maximising agent beelines
past them.

### Build matrix

Each map is built in two variants from the same spec:

      buggy — defects applied. The main evaluation target.

      fixed — defects: [] . Used for (i) the false-positive measurement, and (ii) the
     regression test: Bug Book trajectories recorded on buggy are replayed on fixed and
     MUST now pass, which is how we verify the Bug Book's decrement-and-remove rule
     actually works.

That gives 14 playable maps across two WADs, qa_scenarios_buggy.wad and
 qa_scenarios_fixed.wad . Keep slot numbers identical across the two so a config differs
only in doom_scenario_path .

## ViZDoom configuration and the environment wrapper

### DoomGame settings (normative)

Set these in code in QADoomEnv.__init__ , not only in the .cfg , so that a misplaced
config file cannot silently change the experiment.

   Setting                                                           Value         Why

    set_mode(Mode.PLAYER)                                            synchronous   Deterministic
                                                                                   stepping; ASYNC
                                                                                   modes drop tics

    set_screen_resolution(RES_160X120)                               smallest      Throughput. We
                                                                     supported     downsample to
                                                                                   64x64 anyway.

    set_screen_format(ScreenFormat.GRAY8)                            1 channel     The policy uses
                                                                                   depth, not colour

    set_depth_buffer_enabled(True)                                   on            Primary visual
                                                                                   observation;
                                                                                   geometry-focused
                                                                                   and texture-invariant

    set_labels_buffer_enabled(True)                                  on            Lets the entity-
                                                                                   coverage term know
                                                                                   which objects are in
                                                                                   view

    set_objects_info_enabled(True)                                   on            Every actor's
                                                                                   position and name
                                                                                   each tic, for the
                                                                                   invariant oracle

    set_sectors_info_enabled(True)                                   on            Live floor/ceiling
                                                                                   heights, so the oracle
                                                                                   sees a lift that never
                                                                                   returns

    set_automap_buffer_enabled(False)                                off           Costs frames, adds
                                                                                   nothing we do not
                                                                                   get from sectors

   Setting                                                           Value        Why

    set_notifications_buffer_enabled(True)                           on, size 4   ViZDoom 1.3.0+.
                                                                                  Gives in-game
                                                                                  message text such
                                                                                  as pickup and
                                                                                  locked-door
                                                                                  messages, which is a
                                                                                  clean event stream
                                                                                  for the invariant
                                                                                  oracle. Degrade
                                                                                  gracefully if the
                                                                                  installed version
                                                                                  lacks it.

    set_render_hud/weapon/crosshair/decals/part                      all False    Several percent of
   icles/effects_sprites/corpses/screen_flashe                                    frame time each, and
   s                                                                              all are noise in the
                                                                                  depth buffer

    set_window_visible(False)                                        off          Headless

    set_sound_enabled(False)                                         off          Avoids
                                                                                  ALSA/OpenAL errors
                                                                                  in containers

    set_console_enabled(False)                                       off          Keeps stdout clean
                                                                                  for the trace logger

    set_doom_skill(3)                                                NORMAL       Monsters present
                                                                                  but not the dominant
                                                                                  signal

    set_episode_timeout(goal.time_limit_tics)                        from spec    Truncation, not
                                                                                  termination

    set_episode_start_time(14)                                       14 tics      Skips the weapon-
                                                                                  raise animation

    set_living_reward(0)                                             0            All reward shaping
                                                                                  happens in Python,
                                                                                  never in the engine

   Setting                                                                        Value        Why

    set_seed(worker_seed)                                                         per worker   Reproducibility;
                                                                                               record it in the run
                                                                                               manifest

Game variables to request: POSITION_X , POSITION_Y , POSITION_Z , ANGLE , VELOCITY_X ,
VELOCITY_Y , VELOCITY_Z , HEALTH , ARMOR , AMMO2 , SELECTED_WEAPON , KILLCOUNT ,
 ITEMCOUNT , SECRETCOUNT , DEAD , ON_GROUND , ATTACK_READY , and USER1 – USER4
reserved for any later ACS signalling.

### Action space

Available buttons: MOVE_FORWARD , MOVE_BACKWARD , MOVE_LEFT , MOVE_RIGHT , TURN_LEFT ,
TURN_RIGHT , USE , ATTACK , JUMP . Expose a Discrete(12) space of hand-chosen
combinations rather than 2^9 = 512 raw combinations:

   Index        Combination                        Index         Combination

   0            no-op                              6             forward + turn right

   1            forward                            7             use

   2            backward                           8             forward + use

   3            strafe left                        9             attack

   4            strafe right                       10            jump

   5            forward + turn left                11            forward + jump

 USE MUST be in the action set — without it the agent cannot open doors or press
switches, and three of the six defect classes become unreachable. JUMP is included
deliberately: it is the action most likely to expose geometry exploits, which is exactly what
a tester wants.

Frame skip 4. Each env.step calls make_action(buttons, tics=4) . Oracles still see
every tic through advance_action(1) only in the dedicated replay/verification runs;
during training the oracle consumes the 4-tic-granularity trace, which is enough for every
threshold defined below.

### Determinism and recording

     Per-worker set_seed(base_seed + worker_index) ; the base seed goes in
      runs/<id>/manifest.json along with the git SHA and the WAD's SHA-256.
     Episodes that trigger an oracle are re-run with new_episode(recording_path) so a
      .lmp demo exists for every reported bug. A bug report without a reproducible demo is
     of little use to a developer, and Wuji's case studies show the demo is what makes a
     report credible.
      DoomGame.save(path) / DoomGame.load(path) back the Go-Explore archive. Treat
     save files as opaque, keep them in a per-run tmpdir, and cap the archive to 20,000 cells
     with LRU eviction — each save is tens of kilobytes and an uncapped archive will fill the
     disk overnight.

### Robustness and throughput

ViZDoom spawns a child process per instance and that process can die. Wuji's authors ran
a 30-second health-check supervisor; do the equivalent:

     Wrap every ViZDoom call in the worker. On vizdoom.ViZDoomErrorException or
     ViZDoomIsNotRunningException , record a crash oracle event with the current trace,
     then close and re-init the game and start a fresh episode. An engine crash is a finding,
     not an inconvenience — this is Wuji's crash oracle.
     Retry init up to 5 times with exponential backoff starting at 2 s.
     Hard-kill any worker that does not return from a step within 30 s and record a hang
     event.
     Target throughput: measure with scripts/bench_env.py . Measure on the machine
     you will actually train on. On a Kaggle 4-vCPU GPU session expect roughly 300 steps/s
     per job with 2 workers, about 600 steps/s aggregate across two concurrent jobs. On an
     8-core desktop expect 400-700 steps/s per worker at frame-skip 4. If Kaggle gives you
     under 200 steps/s per job, the usual culprits are OMP_NUM_THREADS left unset so
     torch starves the workers, a visible window, the automap buffer, or RGB rather than
     GRAY8.

## Observation, action and reward design

### Observation space

A Dict space with two keys.

 depth — Box(0, 255, (4, 64, 64), uint8) . The ViZDoom depth buffer at 160x120,
centre-cropped to 120x120 and resized to 64x64, with a 4-frame stack. Depth rather than
colour because the tester's job is geometry, not aesthetics, and depth transfers across
retextured levels for free.

 scalars — Box(-1, 1, (18,), float32) , all egocentric and all normalised:

   Slots       Content

   0-1          sin(angle) , cos(angle)

   2-4         velocity in the agent's own frame, divided by 32
               and clipped

   5            on_ground

   6           speed ratio: realised speed over commanded
               speed, the stuck signal the policy can act on

   7-9         health/100, armour/200, ammo/50

   10-12       keys held, one-hot over red/blue/yellow

   13          fraction of episode elapsed

   14          local novelty: 1/sqrt(1 + n) for the current cell's
               visit count

   15          mean novelty of the 8 neighbouring cells

   16          number of interactive objects in the labels buffer,
               divided by 8

   17          1 if a USE -able object is within 64 mu and in front,
               else 0

What is deliberately absent: absolute POSITION_X/Y/Z . Earle et al. found that global-
observation agents exploit border information to infer coordinates and overfit to a single
layout, while small local observations generalise better to larger and out-of-distribution
maps. A tester that memorises MAP03 is worthless. Absolute position is written to the
trace and used by oracles, the archive and metrics — never fed to the network.

Slots 14 and 15 are the one concession: they leak a derived property of position. They are
included because without them the policy cannot represent "I have been here before" at
all, and Ferdous et al.'s result depends on the agent acting on novelty. Ablate them in the

Tier-3 ablation to show they earn their place.

### Reward

The total reward follows Code-A1's composite form so that the Phase-2 generator can
plug into the same signature:

   R_t \;=\; \alpha\,R^{\text{valid}}_t \;+\; (1-\alpha)\,R^{\text{adv}}_t,
   \qquad \alpha \in [0,1]

Validity term — keeps the tester playing the game legitimately rather than thrashing:

   Component                 Value                                        Note

   Task                      +1.0 per newly satisfied goal                Derived from the MapSpec, never
   progress                  precondition (key collected, gate            hand-written per map
                             opened)

   Goal                      +5.0, terminates the episode
   reached

   Survival                  -5.0 on death                                Dying early is a failed test, not a
                                                                          finding

   Time                      -0.0005 per step                             Mild, so exploration is not punished

Adversarial term — rewards finding things:

   Component                         Value                                 Note

   Cell novelty                       +0.3 / sqrt(1 + n_cell) on           Count-based, counts shared
                                     entering a cell                       across workers

   Abstract-state                     +0.5 / sqrt(1 + n_state)             State = (cell, keys held, door-open
   novelty                                                                 bitmask) — Ferdous et al.'s entity
                                                                           abstraction

   Entity interaction                +1.0 the first time each              Entity coverage
                                     interactive object is used

   Portal traversal                  +1.0 the first time each portal is    Connection coverage
                                     crossed

   Component                         Value                              Note

   Oracle trigger                    +3.0 per new suspicious event      Dedupe before rewarding, or the
                                     cluster                            agent farms one bug

   Bug Book hit                      +1.5 for reaching a state in the   The replay incentive
                                     Bug Book

 α = 0.5 is the default, matching Code-A1's finding that balanced weighting was optimal
for both the generator-side and tester-side metrics. The sweep over {0, 0.25, 0.5, 0.75, 1.0}
is a required ablation.

### Three predictable reward-hacking failures, and the countermeasures

 1. Oracle farming. The agent discovers one stuck pocket and repeatedly re-enters it for
     +3.0. Fix: the oracle-trigger reward is paid once per (oracle_family, cluster_id)
     per episode, and the cluster id comes from the same spatial clustering used in
     reporting.
 2. Novelty churn. The agent oscillates across a cell boundary to re-trigger novelty. Fix:
    counts are per-episode-persistent and global; the bonus uses the visit count, not a
     first-visit flag, so the second crossing is already worth less, and a cell must be exited by
     more than one cell width before re-entry counts.
 3. Suicide-for-reset. If dying ended the episode cheaply, the agent would die to escape a
     hard region. Fix: the -5.0 death penalty, plus episodes truncate rather than terminate
     on timeout so there is no bootstrap discontinuity.

Each countermeasure gets a regression test in tests/test_reward_hacking.py driving a
scripted adversarial policy and asserting that total return stays under a bound.

## The oracle layer

The survey's first identified gap is that no existing method combines efficient exploration
with an automatic, game-agnostic bug oracle. Existing work uses hand-set thresholds
(Wuji), a navigation mesh (Lu et al.), scenario assertions (Mastain and Petrillo) or human
log inspection (Ferdous et al.). This layer is the project's main technical contribution, so
specify it precisely and test it independently of the RL agent.

### Common interface

Every oracle implements:

   class Oracle(Protocol):
       name: str
          def reset(self, episode: EpisodeContext) -> None: ...
          def observe(self, step: StepRecord) -> list[Finding]: ...
          def finalize(self) -> list[Finding]: ...

 StepRecord is the single source of per-step truth: tic, position, velocity, angle, on_ground,
health, armour, keys, action index, commanded-move flag, sector id, visible object ids,
sector floor/ceiling heights, notification text, and the Doom-level flags. Finding carries
 oracle , kind , position , tic_range , confidence in [0,1], evidence (a dict), and
 demo_path .

Oracles MUST be pure functions of the trace plus groundtruth.json . Never let an oracle
call into ViZDoom; that keeps them testable on recorded traces, which is how you will
actually debug them.

### Family 1 — Crash and hang

Triggers on a ViZDoom exception, on the child process exiting, or on a step exceeding 30 s
wall-clock. Also fires on a tic stall: get_episode_time() not advancing across three
consecutive steps while the engine reports itself running. Confidence 1.0 — this is never a
false positive, and a crash during testing is unambiguously a defect. This is Wuji's crash
oracle, unchanged.

### Family 2 — Stuck (intent versus effect)

Wuji compared the current state against a moving average of the previous n states and
flagged a bug when the difference fell below a threshold. That oracle cannot distinguish
an agent that chose to stand still from one the game will not let move, which is the main
source of its false alarms. Our version compares commanded motion with realised
motion, which is strictly more informative and costs nothing.

For a window of W = 70 tics (2 seconds):

      intent = fraction of steps in the window whose action includes a movement button.
      displacement = the diameter of the position set in the window, in the xy-plane.

      escape_attempts = number of distinct movement directions commanded, out of 4.

Fire when intent > 0.8 and displacement < 24 mu and escape_attempts >= 3 and
the agent is on_ground and alive. Confidence rises with the duration of the condition:
 conf = min(1.0, duration_tics / 350) , so a 10-second wedge reports at 1.0 and a 2-

second snag at 0.2.

Second, distinct trigger — vertical trap: the agent is in a sector whose floor is more than
24 mu below every adjacent sector's floor, with no lift special on any bounding linedef.
This is a static check against the ground truth and fires with confidence 0.9 the moment
the agent enters.

### Family 3 — Out of bounds and geometry violation

Ground truth gives the union of room polygons as a rasterised walkable mask plus the
exact rectangles. Three tests per step:

 1. Outside the union. The player's (x, y) is not inside any room rectangle expanded by
     the 16 mu player radius. Confidence 1.0; this is Lu et al.'s nav-mesh test with exact
     polygons instead of a mesh approximation, so there is no 3 m tolerance to tune.
 2. Below floor or above ceiling. z < sector_floor - 8 or z + 56 > sector_ceiling +
     8 while on_ground . Catches clipping into geometry.
 3. Wall traversal. Between two consecutive records the straight segment between
    positions crosses a linedef flagged blocking in the ground truth, and the displacement
     exceeds what the maximum run speed permits over the elapsed tics. Catches both
     teleport-like clipping and speed exploits.

Test 1 is what catches MAP03; it is the single highest-value oracle in the suite.

### Family 4 — Unreachable region

Runs at finalize() , over the whole run rather than one episode. Compare three sets of
cells:

     I — cells the ground truth marks walkable (intended).
     V — cells any agent visited.
     P — cells reachable in the ground-truth portal graph from the spawn, ignoring
     geometry.

Then I \ V is unvisited, and the interesting question is why. Partition it:

   Condition                                           Interpretation               Reported as

   cell in I \ V , room not in P                       The portal graph itself is   unreachable , confidence 0.95
                                                       broken

   Condition                                           Interpretation            Reported as

   cell in I \ V , room in P ,                         The trigger is broken —   unreachable , confidence 0.85
   every portal into that room                         this is MAP04
   was attempted (the agent
   pressed USE on it ≥ 5 times
   across the run) and never
   traversed

   cell in I \ V , room in P ,                         The tester did not get    Not a bug. Logged as
   portals never attempted                             there                     coverage_gap for the coverage
                                                                                 metric only

That third row is the distinction the literature glosses over, and getting it right is what
separates a tester from a coverage counter. Report coverage_gap separately and never
count it in precision or recall.

### Family 5 — Invariants (the declarative DSL)

Wuji needed developer-supplied assertions for logical bugs; Mastain and Petrillo needed a
scenario per feature. We write assertions once per property, in
 qatester/oracles/invariants.py , and they apply to every map:

   INVARIANTS = [
          Inv("health_cap",      lambda s: s.health <= 100 or s.had_powerup),
          Inv("health_monotone", lambda s, p: s.health <= p.health or
   s.picked_up_health),
       Inv("ammo_nonneg",     lambda s: s.ammo >= 0),
       Inv("no_phantom_keys", lambda s, p: s.keys >= p.keys or s.used_key_door),
          Inv("locked_needs_key", lambda s, gt: all(
              gt.portal(p).requires is None or gt.portal(p).requires in s.keys
              for p in s.portals_traversed)),
          Inv("kills_le_spawned", lambda s, gt: s.killcount <= gt.monster_count),
          Inv("z_in_sector",     lambda s: s.sector.floor - 8 <= s.z <=
   s.sector.ceiling - 48),
       Inv("lift_returns",     lambda s, gt: not s.on_lift_sector or
   s.lift_has_return_trigger),
   ]

 locked_needs_key catches MAP05's sequence break; health_cap catches the bad
pickup. Confidence 1.0 for a violated invariant — an invariant is either true or false. The
engineering work is making portals_traversed and picked_up_health reliable, and the

notifications buffer is the cheapest way to get the pickup events.

### Family 6 — Softlock and regression

Softlock. At finalize() , an episode is a softlock candidate when it timed out, the agent
spent the last 40% of the episode confined to a set of cells constituting one room or
fewer, the room is not the goal room, and no portal out of that region was successfully
traversed in that window. Confidence 0.8, raised to 0.95 if independent episodes from
different seeds end confined in the same room. The teleport loop additionally shows as a
high-frequency periodic position signal: flag when the autocorrelation of the position
sequence at lag < 70 tics exceeds 0.9.

Regression. Given a Bug Book entry with a recorded .lmp , replay it on a new build with
 replay_episode() . If the finding no longer reproduces, the defect is fixed; decrement its
frequency. If a finding appears that did not before, report it as a regression with
confidence 0.9. This is the direct analogue of Code-A1 re-evaluating against historical
failures to prevent previously fixed bugs from being reintroduced.

### Testing the oracles

Oracles are tested on synthetic traces, not by running the agent.
tests/fixtures/traces/ holds hand-written StepRecord sequences: one positive and
two near-miss negatives per oracle. The near-misses matter most — an agent standing
still by choice, a player legitimately on a ledge, a health pickup that takes the player to
exactly 100. Target: zero findings on the entire MAP01 CLEAN trace corpus.

## Coverage accounting and the Bug Book

### Four coverage measures

Wuji's most useful negative result is that line coverage was identical across strategies that
differed 16-fold in bugs found, while state coverage tracked bug detection closely
(roughly 807k versus 85k state buckets on NSH). We therefore never report line coverage
and always report these four:

   Measure                             Definition                        Denominator

   Spatial cell                        distinct 64x64 mu cells entered   walkable cells in the ground-
   coverage                                                              truth raster

   Measure                             Definition                                   Denominator

   Entity coverage                     distinct interactive things used or          things with interactive:
                                       picked up                                    true

   Connection                          distinct portals traversed                   portals in the spec
   coverage

   Abstract-state                      distinct (cell, keys_bitmask,                unbounded, reported as a
   coverage                            doors_bitmask) tuples                        raw count

The first three have a true denominator because we authored the map; that is rare in this
literature and worth stating in the report. The fourth is Ferdous et al.'s entity-and-
property state applied to Doom, and it is the one that correlates with finding logic bugs
rather than geometry bugs.

All four are maintained in qatester/coverage.py by a single CoverageTracker whose
counts live in shared memory so all 16 workers see one global picture — without that,
parallel workers duplicate each other's exploration and the Lu et al. speedup disappears.

### The cell archive (Go-Explore)

One entry per discovered cell:

   Field                                        Purpose

    cell                                         (ix, iy, iz_bucket, keys_bitmask) — z
                                                bucketed at 64 mu so multi-level areas separate

    save_path                                   ViZDoom save file for the best-known way to arrive
                                                here

    visits                                      Times entered

    resets_from                                 Times chosen as a reset point

    new_cells_after_reset                       Total new cells discovered within 500 steps of a
                                                reset from here

    best_return                                 Score of the trajectory that reached it, for
                                                replacement

    last_improved_step                          For staleness

Reset selection. Lu et al. compared three heuristics and found all beat uniform sampling.
Implement all three behind --reset-heuristic :

 1. count — sample with weight 1/sqrt(1 + visits) .
 2. frontier — weight by proximity to the nearest unvisited walkable cell, 1/(1 +
   d_nearest_unvisited) .
 3. mixed — the product of the two, which is the default.

The ReMiDi guard. Beukman et al. showed regret-based selection stagnates by
repeatedly sampling levels with high irreducible regret. The spatial analogue is a cell the
agent keeps resetting to that never yields anything — a dead end against a wall that
merely looks like a frontier. Multiply every weight by a productivity factor (1 +
new_cells_after_reset) / (1 + resets_from) , so a cell reset to 20 times with no
payoff falls out of the sampling distribution without being deleted. Make this an ablation:
it is a small change with a literature justification and is likely to show a real effect on
MAP07.

### The Bug Book

Structurally identical to Code-A1's Mistake Book, including the update rule, which is worth
keeping faithful so the comparison in the write-up is honest. Persisted as JSON after
every training iteration so a run can resume.

   {
       "MAP07": [
           {
               "entry_id": "map07-stuck-0003",
               "oracle": "stuck",
               "cell": [21, -7, 0, 1],
               "position": [1376.0, -448.0, 0.0],
               "save_path": "archive/map07/c21_-7_0_1.sav",
               "demo_path": "demos/map07/stuck_0003.lmp",
               "action_suffix": [1, 1, 5, 1, 1, 10],
               "frequency": 7,
               "first_seen_step": 412330,
               "last_confirmed_step": 1883200
           }
       ]
   }

Update rule (mirroring theirs exactly):

     A new oracle finding not matching an existing entry is added with frequency = 1 .
     A finding matching an existing entry increments frequency , capped at 20.
     Replaying an entry that no longer reproduces decrements frequency .
     At frequency == 0 the entry is removed — the defect is considered mastered.

Its three roles, matching the three Code-A1 lists:

 1. A stable baseline for reward. Oracle-trigger reward is computed against the Bug Book,
     so the signal does not swing with the stochasticity of what the agent happens to find
     this iteration.
 2. A curriculum signal. The ratio of new findings to Bug Book re-confirmations tells you
     whether the tester is still discovering or has saturated. Log it every iteration; in Phase 2
     it becomes the generator's difficulty signal.
 3. Anti-forgetting. Re-running Bug Book entries each iteration stops the tester from
     drifting away from regions it has already proven productive. Code-A1 found this
     mattered mainly for the Code LLM (HumanEval+ 72.69 to 69.66 when ablated) and only
     marginally for the Test LLM (Mul 7.14 to 6.99). Expect the same asymmetry: in Phase 1,
     with no builder, the Bug Book's measurable benefit will come mostly from role 1 and
     from regression catching, not from a large jump in discovery rate. Say so in the write-up
     rather than overclaiming.

Matching rule. Two findings are the same entry when their oracle family matches and
their positions are within 96 mu and, for invariant findings, the invariant name matches.
Use the same rule here and in report clustering so the numbers are consistent.

## The RL agent: five tiers

Build the tiers in order. Each is a usable system and each is a row in the final results table,
so nothing here is throwaway scaffolding.

   Tier       Agent                                                  What it exists to show                   Build
                                                                                                              effort

   T1         Uniform random policy                                  Floor. Wuji's random baseline found      1 hour
                                                                     0.6 bugs in 10 h against its 9.6-17.2.

   T2         Scripted patrol: wall-follow plus                      The "scripted testing" straw man         half a
              USE on contact                                         Bergdahl et al. compare against          day

   Tier       Agent                                                    What it exists to show                 Build
                                                                                                              effort

   T3a        PPO, validity reward only (alpha =                       The winning-is-not-testing result.     1 day
              1)                                                       This agent plays well and tests
                                                                       badly.

   T3b        PPO, composite reward with                               The curiosity tester; the Ferdous et   1 day
              count-based curiosity                                    al. comparison

   T4         T3b plus Go-Explore archive and                          The Lu et al. comparison; the          2-3 days
              resets                                                   coverage headline

   T5         T4 plus an 8-policy population with                      CUT on the Kaggle budget: 15 GPU-      3-4 days
              NSGA-II selection                                        h, the least load-bearing claim

T5 is already cut from the Kaggle plan: at 8M steps across 3 seeds it is a third of the entire
quota. T1-T4 support every headline claim. Build T5 only if you later get a machine with 16
or more CPU cores, or if the grid finishes early.

### Network architecture (T3-T5)

A two-branch encoder feeding a shared trunk. Small on purpose — Earle et al. found that
small local-observation models generalise better, and that result held even when hidden
dimensions were enlarged to equalise parameter counts, so there is no reason to reach for
a large network here.

   depth (4,64,64) uint8 -> /255
     Conv2d(4, 32, k=8, s=4) + ReLU                                     -> (32,15,15)
        Conv2d(32, 64, k=4, s=2) + ReLU                                 -> (64, 6, 6)
        Conv2d(64, 64, k=3, s=1) + ReLU                                 -> (64, 4, 4)
        Flatten -> Linear(1024, 256) + ReLU

   scalars (18,) -> Linear(18, 64) + ReLU -> Linear(64, 64) + ReLU

   concat(256, 64) = 320
     -> GRU(320, 256)                                           # recurrent: partial observability, Li et al.
        -> policy head               Linear(256, 12)
        -> value head                Linear(256, 1)                  [extrinsic]
        -> value head                Linear(256, 1)                  [intrinsic, separate as in RND-style
   setups]

Total roughly 1.3M parameters. Two value heads with separate discount factors
( gamma_ext = 0.995 , gamma_int = 0.99 ) because the novelty return is non-stationary
and mixing it into one head destabilises the extrinsic value estimate. Orthogonal
initialisation, gain sqrt(2) on hidden layers, 0.01 on the policy head.

Use the GRU, not a frame-stack-only model. Doom is partially observable, which the
MARL review lists as a core difficulty for FPS games, and the stuck oracle's signal — "I
have been pushing forward for two seconds and nothing is happening" — is inherently
temporal. With sb3-contrib 's RecurrentPPO this is a configuration change, not an
implementation.

### T3b: count-based curiosity

Prefer counts over RND. Three reasons: Lu et al. found random exploration after a good
reset beat RND even when RND used the same resetting scheme; counts need no second
network and no reward normalisation; and the count table is the same structure the
archive and coverage tracker already maintain, so it is free.

The novelty bonus is beta / sqrt(1 + N(abstract_state)) with beta = 0.5 , where N
is the global shared count. Decay beta linearly to 0.1 over the first 60% of training so
late training optimises the composite objective rather than pure novelty.

### T4: the Go-Explore loop

   loop:
     if rng() < p_reset and archive is non-empty:
              cell <- sample_cell(heuristic, productivity_guard)
              game.load(cell.save_path)       # ViZDoom save/load
           set env step counter to cell.depth
       else:
           game.new_episode()

       run the PPO policy for up to H = 500 steps
       for each step:
              update coverage, counts, oracles
              if the current cell is new or reached with a better return:
             game.save(new_save_path); archive.insert(cell)
     attribute new cells discovered back to the reset cell
   (new_cells_after_reset)

 p_reset = 0.5 , horizon H = 500 steps. Two engineering notes that will otherwise cost
you a day each: a reset mid-rollout breaks PPO's trajectory continuity, so treat a reset as
an episode boundary for GAE and set dones = True ; and the number of save files grows
fast, so cap the archive at 20,000 cells with LRU eviction and store saves on local disk,
never on a network mount.

### T5: population and multi-objective selection

Wuji maintained a population evolved by binary tournament on Pareto dominance with
crossover and Gaussian mutation, trained each offspring further with A2C, and selected
the next generation by non-dominated sorting with crowding distance. Yuan et al.
independently found the RL-plus-evolution combination explores better than either alone.
Our lighter version:

     Population of 8 PPO policies sharing one archive, one count table and one Bug Book.
     Each policy trains for 100k steps per generation.
     Objectives per policy: (new cells discovered, task preconditions satisfied, distinct
     oracle clusters triggered).
     Non-dominated sorting plus crowding distance keeps the top 4; the bottom 4 are
     replaced by copies of survivors with Gaussian noise sigma = 0.01 added to all
     weights.
     10 generations, so 8 Mstep total — budget for this before committing.

The diversity benefit is the point, not the evolutionary search itself: a population that
spreads along the Pareto front covers behaviours a single policy collapses away from,
which is precisely Wuji's argument for why its testers beat a single DRL agent.

## Training curriculum and compute budget

### Curriculum

Train on a curriculum of maps rather than one map at a time. A tester that only ever sees
MAP07 overfits to MAP07, which is exactly the failure Gisslén et al. motivate ARLPCG
with: an agent trained on a fixed level set overfits while levels change daily.

   Stage            Env steps                         Maps sampled           Purpose

   Warm-            0 - 100k                          MAP01 only             Learn to move, turn and use.
   up                                                                        Diagnose the pipeline
                                                                             cheaply.

   Core             100k - 600k                       MAP01-06, uniform      Learn general exploration and
                                                                             interaction

   Scale            600k - 1M                         MAP01-07, MAP07        Transfer to the large map
                                                      weighted 0.4

   Hold-            evaluation only,                  MAP07 fixed variant,   Generalisation check
   out              CPU session                       plus MAP08 from an
                                                      unseen spec

Build MAP08 from a spec the agent never trains on. It is the cheapest possible
generalisation result and directly tests Earle et al.'s local-observation claim in our setting.

### Core hyperparameters

Tuned for a Kaggle 4-vCPU session running two jobs concurrently. If you later get a
machine with 16+ cores, raise n_envs to 8-16 and drop n_steps back to 256; nothing else
needs to change.

   Parameter                         Value                                      Note

   Algorithm                          RecurrentPPO (sb3-contrib),
                                      MultiInputLstmPolicy

   Parallel envs                     2 per job, two jobs per session            CPU-bound: 4 vCPU total,
                                                                                learner needs the rest

    n_steps per env                  1024                                       2 x 1024 = 2048 transitions
                                                                                per rollout, which keeps
                                                                                the update stable at 2
                                                                                envs

   Minibatch                         256

   Epochs                            4

   Learning rate                     2.5e-4, linear to 0

   Parameter                         Value                                     Note

    gamma                            0.995 extrinsic, 0.99 intrinsic           A 21000-tic episode is
                                                                               5250 steps at skip 4

    gae_lambda                       0.95

    clip_range                       0.2, linear to 0.1

   Entropy                           0.01, decayed to 0.003 after 500k steps   Exploration matters more
   coefficient                                                                 than usual here; do not
                                                                               drop it to zero

   Value coefficient                 0.5

    max_grad_norm                    0.5

   Reward                             VecNormalize(norm_reward=True,           Never normalise the depth
   normalisation                     norm_obs=False)                           observation

   Total steps                       1M per seed                               T5 and its 8M budget are
                                                                               cut

   Seeds                             5 for E1/E2, 3 for ablations              Reduced from 10/5 to fit
                                                                               the quota; state this in the
                                                                               report

   Checkpoint                        100k steps, to /kaggle/working/ckpt/      Mandatory: 12 h session
   interval                                                                    cap

   Threading                          OMP_NUM_THREADS=1 ,                      Stops torch starving the
                                      torch.set_num_threads(1)                 env workers

### Compute budget: Kaggle, 2x T4, 30 GPU-hours per week

The whole grid is sized to this platform. Read this before changing any step count.

The binding constraint is CPU, not GPU. ViZDoom runs entirely on CPU, one process per
environment worker. A Kaggle GPU session gives 4 CPU cores and ~29 GB RAM. The
policy is 1.3M parameters, so a single T4 is idle most of the time. You cannot run 16 workers
here; you can run 2-3.

The lever that makes this fit: a T4 x2 session costs the same quota as a single-GPU
session. Quota is counted as session wall-clock, not per device. So run two independent
training configs concurrently in one notebook, one pinned to each GPU with

 CUDA_VISIBLE_DEVICES , 2 env workers each. You get two results per quota-hour instead
of one, and the overlap between one job's GPU update and the other's env stepping
recovers some of the CPU you lose by splitting.

Zero-quota work. T1 random, T2 scripted, all evaluation rollouts, E6 generalisation, E7
regression replay, oracle processing, clustering, calibration and figure generation are
forward-pass-or-less. Run them in CPU-only Kaggle sessions, which do not touch the
GPU quota. That removes roughly a third of the workload from the 30 hours.

   Item                                                              Figure

   Kaggle GPU session                                                4 vCPU, ~29 GB RAM, 2x T4, 12 h max
                                                                     session, 20 GB /kaggle/working

   Workers per job                                                   2 ( n_envs = 2 ), two jobs concurrently

   Throughput                                                        ~300 steps/s per job, ~600 steps/s aggregate

   Run length                                                        1M steps, down from 2M

   One run                                                           ~56 min; two concurrent runs finish together

   Quota cost per run                                                ~0.65 GPU-hours including setup and eval

   Week 1: E1 main comparison, 3 configs x 5                         15 runs, ~10 GPU-h
   seeds

   Week 2: E3 alpha (6) + E4 reset (6) + E5                          21 runs, ~14 GPU-h
   ablations (9)

   T1, T2, all evals, E6, E7, figures                                CPU-only sessions, 0 GPU-h

   Total GPU quota                                                   ~24 h across two weeks, against a 60 h
                                                                     allowance

That leaves roughly 36 hours of slack across the two weeks, which you will need: expect to
lose a third of your first week's quota to crashed sessions, a bug found mid-grid, and
reruns.

### What was cut to fit, and what it costs

   Cut                          From                          To                         Cost to the result

   Steps per run                2M                            1M                         Lower absolute coverage.
                                                                                         Mitigated by shrinking
                                                                                         MAP07.

   MAP07 size                   24 rooms,                     12 rooms, 1800x1400 mu     Less dramatic scale story;
                                3000x2400 mu                                             1M steps is now enough to
                                                                                         saturate it

   T5 population                8M steps x 3                  Dropped entirely           Loses the Wuji/MOPCGRL
   agent                        seeds                                                    hybrid-search comparison.
                                                                                         The biggest single saving
                                                                                         (~15 GPU-h) and the least
                                                                                         load-bearing claim.

   Seeds, E1 and                10                            5                          Wider confidence
   E2                                                                                    intervals. Still enough for
                                                                                         Mann-Whitney at the
                                                                                         effect sizes we expect.

   Seeds,                       5                             3                          Report as indicative, not
   ablations                                                                             significant, and say so

   Alpha sweep                  5 values                      3: {0, 0.5, 1.0}           Still shows the inverted-U
                                                                                         that Code-A1 reports; loses
                                                                                         resolution at the shoulders

   Reset-                       4 variants                    3: mixed, frontier,        Drops count -only
   heuristic                                                  uniform
   ablation

   E5 ablations                 5 configs                     3: no Bug Book, no         Keeps the three with
                                                              productivity guard,        literature backing; drops
                                                              absolute-position policy   no-GRU and no-abstract-
                                                                                         novelty

Keep the absolute-position ablation whatever else goes. It is one 3-seed config and it is
the cleanest cheap result in the project.

### Kaggle operating rules

 1. Run in commit mode ("Save & Run All"), not interactively. Interactive GPU sessions die
     after ~20 minutes idle and a browser disconnect kills your run.
 2. Checkpoint every 100k steps to /kaggle/working/ckpt/ and make train.py
     resume from the latest checkpoint automatically. The 12-hour session cap and random
     session deaths make this non-optional, not a nicety.
 3. Publish checkpoints as a Kaggle Dataset at the end of every session and mount it as
    input to the next. /kaggle/working does not survive across sessions.
 4. Go-Explore saves go to /kaggle/temp , not /kaggle/working . Working is capped at
     20 GB and is saved as notebook output; you do not want 5,000 save files in your output.
     Cap the archive at 5,000 cells on this platform, down from 20,000.
 5. Turn Internet on in notebook settings before the first run, or pip install vizdoom
     fails silently at setup.
 6. Set SDL_VIDEODRIVER=dummy and OMP_NUM_THREADS=1 before importing anything.
     Without the second one, torch grabs all 4 cores for its own threading and starves the
     ViZDoom workers — this alone can halve your throughput.
 7. Pin each job to a GPU with CUDA_VISIBLE_DEVICES=0 and =1 , launched as two
     background processes from one notebook cell, with both logging to separate files that
     you tail at the end.
 8. Budget the first GPU session purely to measuring real throughput with bench_env.py .
    If you get less than 200 steps/s per job, stop and fix that before spending quota on
     science.

### The training loop, concretely

   # scripts/train.py
   cfg   = load_config(args.config)                                  # Hydra/YAML
   vec   = make_vec_env(cfg, n_envs=16)                              # QADoomEnv + wrappers
   vec        = VecNormalize(vec, norm_reward=True, norm_obs=False)

   archive          = CellArchive(cfg.archive)                       # shared, in the main process
   bugbook = BugBook.load_or_new(cfg.run_dir)
   coverage = CoverageTracker(groundtruth)

   model = RecurrentPPO(QAPolicy, vec, **cfg.ppo)

   callbacks = [
       OracleDrainCallback(queue, oracles, bugbook, coverage),                      # drains worker

   events
       GoExploreCallback(archive, p_reset=0.5, horizon=500),           # T4+
       CoverageLogCallback(coverage, every=10_000),
          BugBookCheckpoint(bugbook, every=50_000),
          CurriculumCallback(cfg.curriculum),
          EvalCallback(eval_vec, n_eval_episodes=20, every=100_000),
   ]

   model.learn(total_timesteps=cfg.total_steps,
   callback=CallbackList(callbacks))
   write_manifest(cfg.run_dir, git_sha(), wad_sha256(), cfg)

### What to log, every 10k steps

To TensorBoard and to runs/<id>/metrics.jsonl : the four coverage measures;
cumulative distinct oracle clusters by family; Bug Book size and the new-findings-to-
reconfirmations ratio; episode return split into validity and adversarial components;
episode length; death rate; archive size; resets performed; mean productivity factor; env
steps per second. The coverage-versus-steps curve and the bugs-versus-wall-clock curve
are the two plots that go in the report, so make sure both are recoverable from this file
alone.

## From findings to a usable bug report

A 2M-step run will emit tens of thousands of raw findings for a handful of real defects.
Raw findings are not a deliverable; none of the surveyed testing papers addresses this,
which makes it cheap novelty for us and also the difference between a demo and
something a developer would use.

### Pipeline

1. Spatial-semantic clustering. DBSCAN over findings of the same oracle family, with eps
= 96 mu (1.5 door widths) and min_samples = 3 , on the feature (x, y, z/4) . Invariant
findings cluster on the invariant name instead of position. Output: one candidate defect
per cluster.

2. Representative selection. Within a cluster pick the finding with the highest
confidence, breaking ties by the shortest demo — the shortest reproduction is the most
useful one to hand a developer.

3. Severity. severity = w_o * oracle_weight + w_b * blocking_factor + w_f *
frequency_factor :

   Factor                              Definition

    oracle_weight                      crash 1.0, softlock 0.9, out_of_bounds 0.8, invariant
                                       0.7, unreachable 0.6, stuck 0.5

    blocking_factor                    1.0 if the defect sits on the only path to the goal in
                                       the ground-truth portal graph, 0.3 otherwise

    frequency_factor                    min(1, cluster_size / 10)

Weights 0.5 / 0.3 / 0.2 . Rank the report by severity, not by discovery order.

4. Confidence calibration. Oracle confidences are hand-set priors and will be
miscalibrated. Huang et al. show that agents learning from their own feedback become
systematically overconfident, with rising ECE and worse accuracy precisely on high-
confidence predictions, and that calibration after self-improvement is the effective
remedy. Apply it:

       Hold out MAP02, MAP04 and MAP06 as the calibration split; fit on their labelled
       outcomes, evaluate on MAP03, MAP05 and MAP07.
       Fit a single temperature T on the logit of the raw confidence by minimising NLL
       against the ground-truth label (true defect or false alarm).
       Report ECE with 10 bins, before and after, plus a reliability diagram. Target ECE < 0.1
       post-calibration.

This is a small amount of work that converts a cited gap in the survey into an actual
measured result, and it is the easiest novelty claim in the project to defend.

5. Report emission. results/<run_id>/report.json and report.html . The JSON
schema is fixed now because the Phase-2 builder consumes it:

   {
       "run_id": "2026-11-04T09-12-33Z_t4_seed3",
       "map": "MAP07",
       "wad_sha256": "...",
       "git_sha": "...",
       "env_steps": 2000000,
       "wall_clock_s": 2210,
       "coverage": {"cell": 0.883, "entity": 1.0, "connection": 0.92,

   "abstract_states": 184203},
     "findings": [
          {
              "id": "MAP07-oob-01",
              "oracle": "out_of_bounds",
              "severity": 0.84,
              "confidence_raw": 1.0,
              "confidence_calibrated": 0.97,
              "position": [1984.0, -512.0, 0.0],
              "room": "east_hall",
              "cluster_size": 47,
              "first_seen_step": 318450,
         "repro": {"demo": "demos/MAP07/oob_01.lmp", "save": "archive/...",
   "steps": 212},
         "evidence": {"linedef": 214, "exit_point": [1984.0, -512.0],
   "depth_beyond_mu": 340},
         "summary": "Player left the level through the east hall's north wall
   and travelled 340 mu into undefined space."
         }
       ],
     "coverage_gaps": [{"room": "north_store", "cells_unvisited": 41,
   "portals_attempted": 0}]
   }

Note coverage_gaps is a separate top-level list, never mixed into findings . Conflating
"the tester did not go there" with "the game is broken there" is the single easiest way to
inflate your own results, and reviewers will look for it.

The HTML report is a single self-contained file: the map preview SVG with findings plotted
as markers, a sortable table, and the per-finding evidence inline. Generate it with Jinja2
from the same JSON, so the two can never disagree.

## Evaluation protocol

### Metrics

   Metric                                    Definition                         Why this one

   Defects found @                           Distinct ground-truth defects      Wuji's headline metric, reported
   budget                                    with at least one matching         against both budgets
                                             cluster, at 0.5M / 1M / 2M steps
                                             and at 1 h / 4 h wall clock

   Precision                                 true clusters / all clusters       Nobody in the surveyed testing
                                                                                papers reports this. We can,
                                                                                because we have ground truth.

   Recall                                    true clusters / injected
                                             defects

   F1                                        harmonic mean                      Single headline number

   False alarms on                           clusters reported on MAP01         The cleanest honesty check in
   CLEAN                                                                        the whole study

   Reports per true                          clusters / true defects            Measures whether
   defect                                                                       deduplication works. Target <
                                                                                1.5.

   Cell / entity /                           as defined earlier                 Ferdous et al. and Lu et al.
   connection coverage                                                          comparison

   Abstract-state                            distinct tuples                    Wuji's state coverage analogue
   coverage

   Time to first detection                   env steps until each defect is     Shows the exploration
                                             first reported                     advantage sharply

   ECE                                       10-bin expected calibration        Huang et al.
                                             error, pre and post

   Cost                                      wall clock, env steps, GPU-        Objective O7
                                             hours

A cluster counts as true when its representative is within 128 mu of the injected defect's
recorded position and its oracle family matches the defect's expected_oracle . Fixing
both the distance and the family prevents the embarrassing case of claiming a detection
because an unrelated oracle happened to fire nearby.

### Experiment grid

   Experiment                       Configurations                   Seeds   GPU cost     Purpose

   E1 Main                          T1, T2 (CPU only) + T3a,         5       15 runs,     The headline table
   comparison                       T3b, T4                                  ~10 h

   E2 Winning vs                    T3a vs T3b on MAP07              5       reuses E1    Wuji's central claim,
   testing                                                                                replicated

   E3 Alpha sweep                   alpha in {0, 0.5, 1.0} on T3b    3       6 new        Code-A1's Figure 3
                                                                             runs, ~4     in a spatial setting
                                                                             h

   E4 Reset                         mixed (from E1), frontier,       3       6 new        Lu et al.'s ablation
   heuristic                        uniform on T4                            runs, ~4
                                                                             h

   E5 Component                     no Bug Book; no                  3       9 runs, ~6   Each isolates one
   ablations                        productivity guard; policy               h            design claim
                                    given absolute position

   E6 Generalisation                T4 from E1 evaluated on          5       CPU only,    Earle et al.'s local-
                                    unseen MAP08, no                         0h           observation claim
                                    training

   E7 Regression                    Bug Book from buggy              5       CPU only,    Does decrement-
                                    replayed on fixed                        0h           and-remove work

Total: 36 GPU runs, roughly 24 GPU-hours across two weekly quotas. Everything else
runs on CPU-only sessions.

E5's absolute-position row is the important one for the write-up: give the policy absolute
 POSITION_X/Y and expect in-distribution performance to hold or improve while MAP08
generalisation drops. That is a direct replication of Earle et al.'s border-tile overfitting
finding in a new domain, and it is a clean, cheap result.

### Statistics

Five seeds for E1 and E2, three for ablations, reported as median with interquartile range
rather than mean with standard deviation, because bug counts are small integers with
heavy tails. Pairwise comparisons use a two-sided Mann-Whitney U test with Holm-
Bonferroni correction across the comparison family, and report the Vargha-Delaney A12
effect size, which is the standard in search-based software engineering and is what a
software-testing venue will expect. State the significance level before running anything
and do not change it afterwards.

### Result tables to produce

 1. Main table — rows T1-T5, columns: defects found (median, IQR), precision, recall, F1,
    false alarms on CLEAN, cell coverage, wall clock. This is the paper's Table 1.
 2. Per-defect table — rows = the 12 injected defects, columns = agent tiers, cells =
     median steps to first detection or a dash. This is where the "T3a never finds the off-
     path defect" story becomes visible at a glance.
 3. Alpha sweep plot — defects found and cell coverage against alpha, mirroring Code-
     A1's Figure 3 layout so the comparison is immediate.
 4. Coverage curves — cell coverage against env steps, one line per tier, shaded IQR.
 5. Calibration — reliability diagram plus ECE before and after.

### Threats to validity to state explicitly

     Injected defects are not natural defects. Wuji's authors faced the same issue and
     spent real money mitigating it. We cannot; say so. Mitigate partially by having one team
     member author the defect specs and another author the oracles, without discussing
     thresholds.
     Single engine, single genre. No claim about transfer beyond Doom-like FPS levels.
     The oracles know the ground truth. They are automatic given a spec, not automatic
     given an arbitrary game. The honest framing is: automatic for any game that can export
     its intended level semantics — which is every game with a level editor.
     Seven maps is a small corpus. Report per-map results, not only aggregates.

## Repository layout and module contracts

   rl-qa-tester/
       pyproject.toml
       README.md
       docs/decisions.md             # every deviation from this plan, dated
       maps/
         m01_clean.yaml m02_wedge.yaml m03_leak.yaml m04_orphan.yaml
         m05_phantom.yaml                    m06_softlock.yaml       m07_sprawl.yaml   m08_holdout.yaml
       qatester/
         __init__.py
          mapgen/
            spec.py                            # Pydantic models + validation + check_reachable()
              geometry.py                      # rectangles -> vertices, linedefs, sidedefs, sectors
              udmf.py                          # TEXTMAP emitter
              wad.py                           # PWAD binary writer (no third-party deps)
         defects.py                            # defect kind registry: each is a spec -> spec
   mutation
         groundtruth.py                        # walkable raster, portal graph, defect positions
              preview.py                       # top-down SVG render
              cli.py                           # python -m qatester.mapgen build maps/ -o build/
          env/
            vizdoom_env.py                     # QADoomEnv(gymnasium.Env)
            observation.py                     # depth crop/resize, scalar vector assembly
              actions.py                       # the 12 discrete combinations
              reward.py                        # validity + adversarial composition
              trace.py                         # StepRecord, Parquet writer, .lmp recording
            supervisor.py                      # crash/hang detection, restart with backoff
          oracles/
              base.py         # Oracle protocol, Finding, EpisodeContext
              crash.py stuck.py bounds.py reach.py invariants.py softlock.py
              registry.py
          coverage.py                          # CoverageTracker, shared counts
          archive.py                           # CellArchive, reset heuristics, productivity guard
          bugbook.py                           # BugBook: add/confirm/decrement/remove, persistence
          agents/
            random_agent.py                    scripted_agent.py
              policy.py                        # the two-branch network
              ppo_runner.py                    # T3a/T3b
              goexplore.py                     # T4 callback
            population.py   # T5 NSGA-II selection
          report/
            cluster.py severity.py calibrate.py emit.py

   templates/report.html.j2
     scripts/
       build_maps.py                   train.py           evaluate.py   bench_env.py   replay.py
   make_figures.py
     tests/
       test_spec_validation.py                            test_wad_roundtrip.py
   test_geometry_invariants.py
          test_env_contract.py                            test_oracles.py         test_reward_hacking.py
          test_bugbook_rule.py                            test_clustering.py
          fixtures/traces/
       runs/                    # gitignored
       build/                   # gitignored
       results/

### Key signatures (normative — downstream code depends on these)

   # qatester/mapgen/cli.py
   def build(spec_dir: Path, out_dir: Path, variant: Literal["buggy", "fixed"])
   -> BuildResult:
       """Writes <out>/qa_scenarios_<variant>.wad, one .cfg and one
       .groundtruth.json per map, and one .preview.svg per map."""

   # qatester/env/vizdoom_env.py
   class QADoomEnv(gymnasium.Env):
          def __init__(self, cfg_path: Path, groundtruth: GroundTruth,
                       coverage: CoverageTracker, oracle_queue: mp.Queue,
                       seed: int, record_dir: Path | None = None): ...
          def reset(self, *, seed=None, options=None) -> tuple[ObsDict, dict]: ...
          def step(self, action: int) -> tuple[ObsDict, float, bool, bool, dict]:
   ...
          def load_state(self, save_path: Path) -> ObsDict:
              """Go-Explore reset. Calls DoomGame.load and rebuilds the frame
   stack."""
       def save_state(self, save_path: Path) -> None: ...

   # qatester/oracles/base.py
   @dataclass(frozen=True)
   class Finding:
          oracle: str; kind: str; position: tuple[float, float, float]
          tic_range: tuple[int, int]; confidence: float
          evidence: dict[str, Any]; demo_path: Path | None

   # qatester/archive.py
   class CellArchive:
       def insert(self, cell: Cell, save_path: Path, score: float) -> bool: ...
          def sample_reset(self, heuristic: str, rng: np.random.Generator) ->
   CellEntry | None: ...
       def attribute(self, from_cell: Cell, new_cells: int) -> None: ...

   # qatester/bugbook.py
   class BugBook:
          def add_or_confirm(self, f: Finding, map_name: str) -> str: ...
          def decrement(self, entry_id: str) -> None: ...
          def entries_for(self, map_name: str) -> list[BugBookEntry]: ...
          def save(self, path: Path) -> None: ...

### Hard rules for the implementation

 1. qatester/agents/** MUST NOT import qatester.mapgen.groundtruth . Enforced by
       tests/test_no_groundtruth_leak.py .
 2. Oracles MUST NOT import vizdoom . Enforced by the same test file.
 3. Every random source goes through an explicitly seeded np.random.Generator passed
     in; no module-level random or np.random calls.
 4. All paths come from config; no literal paths outside scripts/ .
 5. Type-annotate everything and run mypy --strict on qatester/mapgen/ and
       qatester/oracles/ at minimum. Those two packages are where a silent type error
     turns into a wrong experimental result.

## Build order: M0 to M9

Each milestone has an acceptance test that MUST pass before the next begins. The
acceptance command is what you run; if it is green, move on.

   M          Milestone                 Deliverable                   Acceptance test                       Days

   M0         Environment               Repo skeleton, pinned deps,   pytest tests/test_smoke.py runs       1
              up                        ViZDoom running a stock       1000 random steps in basic.cfg at >
                                        scenario headless             300 steps/s

   M          Milestone                 Deliverable                       Acceptance test                              Days

   M1         WAD writer                qatester.mapgen.wad +             scripts/build_maps.py                        2
                                        udmf producing a one-room         maps/m01_clean.yaml then ViZDoom
                                        map                               loads MAP01, the agent spawns,
                                                                           POSITION_X is finite, 500 random steps
                                                                          run without error

   M2         Full compiler             All 8 specs compile; portals,     pytest                                       3
                                        doors, lifts, teleports work;     tests/test_geometry_invariants.py
                                        preview SVGs render               green; every map manually walked once
                                                                          in SPECTATOR mode with --render

   M3         Ground truth              groundtruth.json per              pytest tests/test_defects.py : each          2
              and defects               map; all 7 defect kinds inject    defect changes exactly the intended
                                        correctly                         linedef/sector and appears in the ground
                                                                          truth; the fixed variant of every map is
                                                                          fully reachable by BFS

   M4         Oracle layer              All 6 families against            pytest tests/test_oracles.py green;          3
                                        synthetic traces                  zero findings on the entire MAP01
                                                                          CLEAN trace corpus; every injected
                                                                          defect detected in its hand-authored
                                                                          positive trace

   M5         Gym                       QADoomEnv with obs,               pytest tests/test_env_contract.py            3
              environment               actions, reward, trace            (Gymnasium API conformance) and
                                        recording, supervisor              tests/test_reward_hacking.py green;
                                                                           scripts/bench_env.py reports > 400
                                                                          steps/s single worker

   M6         T1-T3a                    Random, scripted, task-only       A report.html exists for MAP03 from a        3
              baselines                 PPO; the full logging and         random run; T3a reaches the MAP01 exit
                                        report pipeline end to end        in > 80% of eval episodes

   M7         T3b curiosity             Count-based novelty,              Cell coverage on MAP07 at 1M steps is        3
              tester                    shared counts, Bug Book           statistically higher for T3b than T3a over
                                        wired in                          5 seeds

   M8         T4 Go-Explore             Archive, save/load resets,        Cell coverage on MAP07 at 1M steps is at     4
                                        three heuristics,                 least 80%; pytest
                                        productivity guard                tests/test_bugbook_rule.py green

   M9         Evaluation and            E1-E7 run; all five figures and   scripts/make_figures.py regenerates          5
              write-up                  tables; calibration fitted        every figure from results/ alone, with
                                                                          no manual steps

Total about 29 working days for one person, or roughly 3 calendar weeks for a four-person
team working in parallel across the three tracks below.

### Parallel tracks for a four-person team

   Track                      Milestones                             Who

   A—                         M1, M2, M3                             One person owns the compiler end to end;
   Authoring                                                         it is the critical path

   B — Oracles                M4, and the report pipeline            A different person from track A, by design:
                              in M6                                  the defect author and the oracle author
                                                                     should not agree thresholds between
                                                                     themselves, or the evaluation is circular

   C — RL                     M5, M6, M7, M8                         Two people; one on the env and policy, one
                                                                     on the archive and population

   D—                         M9                                     Whoever is free; starts at M6 by building
   Evaluation                                                        the figure scripts against fake data

Track D building figure scripts against synthetic results before any real run exists is worth
the half-day: it forces the metric definitions to be concrete early and means M9 is
assembly rather than invention.

### What to do when a milestone is late

In order of what to cut: T5 population first, then the MAP07 scale map (fall back to 12
rooms), then the alpha sweep down to three values, then MAP08 generalisation. Do not
cut M4's clean-map zero-false-positive requirement or the 10-seed protocol — those are
what make the numbers mean anything.

## Development environment setup

### Platform

Use Linux, or WSL2 on Windows. ViZDoom ships Linux wheels for Python 3.8+ and builds
most reliably there; headless operation with no X server is a documented, supported
configuration. Native Windows wheels exist but multiprocessing with SubprocVecEnv ,
headless rendering and save-file paths all behave differently and will cost you time that
belongs to the project. Python 3.11.

### Dependencies

   vizdoom>=1.3.0                                # 1.3.0 gives the notifications buffer and
   set_config(dict)
   gymnasium>=0.29
   stable-baselines3>=2.3
   sb3-contrib>=2.3                              # RecurrentPPO
   torch>=2.2
   numpy scipy pandas pyarrow
   pydantic>=2.6
   scikit-learn                                  # DBSCAN, calibration
   matplotlib
   jinja2
   pyyaml
   hydra-core                                    # or plain argparse + YAML if you prefer
   pytest pytest-cov mypy ruff

No omgifol , no node builder, no acc unless the optional ACS path is taken.

### Commands

   git clone <repo> && cd rl-qa-tester
   python3.11 -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"

   # TODO(human): place an IWAD next to the package.
   # freedoom2.wad is free and legally redistributable; use it.
   # wget https://github.com/freedoom/freedoom/releases/.../freedoom-0.13.0.zip
   # Do NOT commit doom2.wad to the repository.

   python -c "import vizdoom; print(vizdoom.__version__)"
   pytest tests/test_smoke.py -q                                         # M0 gate

   python scripts/build_maps.py maps/ -o build/ --variant buggy
   python scripts/build_maps.py maps/ -o build/ --variant fixed
   open build/MAP07.preview.svg                                          # eyeball the geometry

   python scripts/bench_env.py --cfg build/MAP01.cfg --steps 20000
   python scripts/train.py --config configs/t4_goexplore.yaml --seed 0
   python scripts/evaluate.py --run runs/<id> --maps MAP01..MAP08
   python scripts/make_figures.py --results results/

### Headless notes

Set SDL_VIDEODRIVER=dummy before importing ViZDoom in containers without a display. If
ViZDoom still complains about audio, confirm set_sound_enabled(False) is applied
before init() ; it is the single most common container failure. Keep
 set_window_visible(False) — ViZDoom with the window disabled is explicitly
supported on Linux systems without an X server.

## Risk register

Ordered by expected cost. The first four are the ones that actually sink student projects
of this shape.

   #          Risk                             Signal it is happening                      Mitigation / fallback

   R1         Geometry bugs                    Oracles fire on MAP01 CLEAN; walls          This is why M4's gate
              in your own                      invisible from one side; the player falls   is zero findings on
              compiler look                    through the floor at spawn                  CLEAN. Render
              exactly like game                                                             preview.svg for
              bugs                                                                         every build, walk every
                                                                                           map once in
                                                                                           SPECTATOR mode,
                                                                                           and assert the winding
                                                                                           and closure invariants
                                                                                           before writing the
                                                                                           WAD. Budget a full day
                                                                                           for linedef winding
                                                                                           alone.

   #          Risk                             Signal it is happening               Mitigation / fallback

   R2         The agent never                  Coverage plateaus below 50%; the     Shrink MAP07 to 12
              reaches the                      only defects found are near spawn    rooms. Raise
              interesting parts                                                      p_reset . Add
              of MAP07                                                              intermediate marker
                                                                                    things that award
                                                                                    task-progress reward
                                                                                    so the goal is not one
                                                                                    sparse terminal event.
                                                                                    Wuji's Block Maze
                                                                                    makes the same point:
                                                                                    an agent that cannot
                                                                                    complete intermediate
                                                                                    tasks never reaches
                                                                                    the deep bugs.

   R3          DoomGame.save /                 Go-Explore resets dominate wall      Benchmark save/load
               load is slower or               clock; loads occasionally return a   in M5 before building
              less reliable than               corrupt state                        T4 on top of it.
              expected                                                              Fallback: replay-based
                                                                                    resets — store the
                                                                                    action prefix instead
                                                                                    of a save file and re-
                                                                                    execute it from
                                                                                     new_episode() with a
                                                                                    fixed seed. Slower per
                                                                                    reset but has no
                                                                                    engine dependency,
                                                                                    and you need the
                                                                                    action prefix for the
                                                                                    demo anyway.

   #          Risk                             Signal it is happening                   Mitigation / fallback

   R4         Oracle                           Thresholds changed after seeing          Freeze all thresholds
              thresholds are                   detection rates                          at the end of M4,
              tuned until the                                                           record them in
              results look good                                                         docs/decisions.md
                                                                                        with the date, and
                                                                                        have the oracle author,
                                                                                        not the defect author,
                                                                                        own them. State in the
                                                                                        report that they were
                                                                                        frozen before
                                                                                        evaluation.

   R5         ViZDoom child                    Workers silently stop producing steps;   The supervisor in
              processes crash                  throughput decays over hours             env/supervisor.py :
              or hang under 16-                                                         30 s step watchdog, 5
              way parallelism                                                           retries with backoff,
                                                                                        crash recorded as a
                                                                                        finding. Test it by
                                                                                         SIGKILL -ing a worker
                                                                                        mid-run.

   R6         Disk fills with                  Training dies overnight with ENOSPC      Cap the archive at
              save files and                                                            20,000 cells with LRU
              demos                                                                     eviction; keep demos
                                                                                        only for findings that
                                                                                        survive clustering; put
                                                                                         runs/ on local disk
                                                                                        and gitignore it.

   R7         Reward hacking                   Coverage high, defects found low,        The three
              produces                         episode returns climbing smoothly        countermeasures and
              impressive-                                                               their regression tests.
              looking but                                                               Always read coverage
              meaningless                                                               and defects found
              coverage                                                                  together; never report
                                                                                        coverage alone.

   #          Risk                             Signal it is happening               Mitigation / fallback

   R8         ACS turns out to                 A defect class cannot be expressed   All seven Phase-1
              be needed and                    with linedef specials                defect classes are
              acc will not build                                                    expressible without
                                                                                    ACS by construction.
                                                                                    If an eighth is wanted,
                                                                                    drop it rather than
                                                                                    taking on the
                                                                                    dependency.

   R9         Notifications                     set_notifications_buffer_enabled    Feature-detect with
              buffer absent in                 raises                               hasattr . Fallback for
              the installed                                                         pickup events: watch
              ViZDoom                                                               ITEMCOUNT , HEALTH
                                                                                    and the objects list for
                                                                                    disappearing actors.
                                                                                    Slightly noisier,
                                                                                    entirely workable.

   R10        Ten seeds x                      Grid still running the week of the   Reduce seeds to 5 for
              seven                            report                               ablations but keep 10
              experiments                                                           for E1 and E2, which
              does not fit the                                                      are the headline
              deadline                                                              claims. Record the
                                                                                    reduction in the
                                                                                    report.

   R11        Depth buffer                     Depth looks wrong or randomised      viz_nocheat adds
              contains noise                                                        noise to the depth and
                                                                                    labels buffers. Ensure
                                                                                    it is off; it is off by
                                                                                    default, so suspect a
                                                                                    stray add_game_args .

## Phase 2 hooks

Phase 1 is deliberately shaped so that Phase 2 is an addition, not a rewrite. Five seams are
left open.

1. The Builder consumes MapSpec , not WADs. A generator in Phase 2 emits MapSpec
objects and the existing compiler turns them into playable levels with ground truth
attached. This is the key structural decision: because the spec, not the binary, is the
interface, a generator gets free oracles.

2. The reward already has the alpha form. R = alpha * R_valid + (1 - alpha) *
R_adv is Code-A1's signature. In Phase 2 the Builder takes the opposing objective —
rewarded when the Tester fails to achieve coverage or goal progress, penalised when its
levels are unplayable — and the same alpha controls its validity/difficulty balance. Gisslén
et al.'s ARLPCG is the proof that this works in a level-generation setting, with the caveat
they identified: a fixed, non-adaptive Solver was exploited and the Generator converged
to impossible tracks, so the Tester must keep training alongside.

3. The Bug Book is already the experience-replay buffer. Code-A1's ablation showed the
Mistake Book mattered mainly for the Code LLM (the artefact producer), not the Test
LLM. The analogous prediction for us is that the Bug Book's benefit will be modest in
Phase 1 and substantial in Phase 2, where it stops the Builder from regenerating defects
the Tester has already mastered. Phase 1 measures the small effect so Phase 2 can
measure the large one against a known baseline.

4. The curriculum signal is already logged. The ratio of new findings to Bug Book
reconfirmations, logged every iteration, is directly usable as the Phase-2 difficulty
controller. Guo et al.'s GenEnv targets an intermediate success rate of roughly 50% — the
agent's zone of proximal development — and that is the set point to aim the Builder at.
Beukman et al.'s warning applies: without a guard, selecting purely for Tester failure stalls
on levels that are unlearnable rather than merely hard, which is why the archive's
productivity guard is built in Phase 1 and generalises directly.

5. The report schema is the Builder's input format. report.json already carries
positions, rooms, defect kinds and severities in machine-readable form. The Phase-2 loop
is: Builder emits spec, compiler builds, Tester tests, reporter emits JSON, Builder reads
JSON and mutates the spec. Closing that loop is what addresses the survey's second gap
— that AI-assisted building and RL testing are studied independently and test results are
never fed back to modify the game.

### Phase 3 and beyond, briefly

Multi-agent testers following cMarlTest's active/passive role split; porting the oracle
interface to a second engine (Godot is the realistic candidate, since it can export level
semantics as JSON); and learned oracles that generalise beyond declared ground truth.

The calibration work from Phase 1 becomes load-bearing there: a learned oracle that is
confidently wrong is worse than no oracle, which is exactly Huang et al.'s warning about
self-improving agents.

## Appendix A: UDMF reference

Verify every number below against the ZDoom wiki before relying on it; they are correct
to the best of my knowledge but the wiki is authoritative and this is exactly the kind of
detail worth double-checking once, early, in M1.

### Lump order for one map inside a PWAD

 MAP01 (0 bytes) → TEXTMAP (the UDMF text) → [ SCRIPTS , BEHAVIOR if using ACS] →
 ENDMAP (0 bytes). Repeat for each map.

### Minimal TEXTMAP

   namespace = "zdoom";

   vertex { x = -320.0; y = -320.0; }
   vertex { x =                   0.0; y = -320.0; }
   vertex { x =                   0.0; y =    0.0; }
   vertex { x = -320.0; y =                              0.0; }

   linedef { v1 = 0; v2 = 1; sidefront = 0; blocking = true; }
   linedef { v1 = 1; v2 = 2; sidefront = 1; blocking = true; }
   linedef { v1 = 2; v2 = 3; sidefront = 2; blocking = true; }
   linedef { v1 = 3; v2 = 0; sidefront = 3; blocking = true; }

   sidedef { sector = 0; texturemiddle = "BRICK7"; }
   sidedef { sector = 0; texturemiddle = "BRICK7"; }
   sidedef { sector = 0; texturemiddle = "BRICK7"; }
   sidedef { sector = 0; texturemiddle = "BRICK7"; }

   sector {
       heightfloor = 0; heightceiling = 128;
       texturefloor = "FLOOR4_8"; textureceiling = "CEIL3_5";
       lightlevel = 160;
   }

   thing { x = -160.0; y = -160.0; type = 1; angle = 0;
                  skill1 = true; skill2 = true; skill3 = true;
                  skill4 = true; skill5 = true; single = true; }

That is a complete, loadable map. Build exactly this in M1 before attempting anything
larger.

### Two-sided linedef (a portal opening)

   linedef { v1 = 4; v2 = 5; sidefront = 8; sideback = 9; twosided = true; }
   sidedef { sector = 0; texturetop = "BRICK7"; texturebottom = "BRICK7"; }
   sidedef { sector = 1; texturetop = "BRICK7"; texturebottom = "BRICK7"; }

No blocking on a two-sided linedef the player should pass through. Setting blocking =
true on a two-sided line is a legitimate construct (an invisible wall) and is precisely the
inverse of the MAP03 defect.

### Linedef specials used by the compiler

   Portal kind                   Special                             Args               Trigger flags

    open                         0                                   —                  two-sided, no
                                                                                        blocking

    door                         12 ( Door_Raise )                   tag=0, speed=16,   playeruse = true
                                                                     delay=150          on both sides, or
                                                                                        Generic_Door

    locked_door                  13 ( Door_LockedRaise )             tag=0, speed=16,   playeruse = true ;
                                                                     delay=150, lock=   lock 1 = red card, 2 =
                                                                     <n>                blue card, 3 = yellow
                                                                                        card

    lift                         62                                  tag=<sector        playeruse = true,
                                 ( Plat_DownWaitUpStay )             tag>, speed=32,    repeatspecial =
                                                                     delay=105          true

    teleport                     70 ( Teleport )                     tid=<destination   playercross = true
                                                                     tid>, tag=0

   Portal kind                   Special                             Args                Trigger flags

   switch for a                  11 ( Door_Open ) or 13              tag=<sector          playeruse = true .
   remote door                                                       tag>, speed=16      The MAP04 defect is
                                                                                         this tag not
                                                                                         matching the door
                                                                                         sector's tag .

For teleports, place a TeleportDest thing (type 14) at the destination with a thing.id
(TID) referenced by the special's first arg.

### Thing types used

   Spec type                                                                Doom editor number

    player_start                                                            1

    key_red / key_blue / key_yellow (cards)                                 13 / 5 / 6

    health (medikit)                                                        2012

    armour (green)                                                          2018

    ammo (clip)                                                             2007

    monster_imp                                                             3001

    teleport_dest                                                           14

    marker (invisible, use a non-blocking decoration or                     9001
   a custom MapSpot , type 9001)

 exit_switch is not a thing: it is a linedef with special 243 ( Exit_Normal ) or an
 ExitSuper / ExitLevel line. The compiler emits it as a one-sided wall segment with
 playeruse = true in the goal room.

### Flags worth knowing

 blocking , blockmonsters , twosided , dontpegtop , dontpegbottom , secret ,
 blocksound , impassable for linedefs; playeruse , playercross , monstercross ,
 repeatspecial for triggers; skill1 - skill5 , single , coop , dm , ambush for things. Set
all five skill flags and single on every thing, or things silently vanish at the skill level you
train at.

### Defect injection, mechanically

   Defect kind                         Mutation applied by defects.py

    missing_blocking                   On the named wall segment, set blocking =
                                       false while leaving it one-sided

    wrong_tag                          Change the switch linedef's arg0 to a tag no
                                       sector carries

    wrong_special                      Set Door_LockedRaise arg3 (lock) to 0

    geometry_trap                      Lower a sector's heightfloor by more than 24
                                       below all neighbours

    one_way_lift                       Emit the Plat_DownWaitUpStay trigger on the
                                       upper sector's linedef only

    teleport_loop                      Point two Teleport specials at each other's
                                       destination TIDs

    bad_pickup                         Replace the medikit with a custom actor granting
                                       150 health, or place a Soulsphere (type 2013) where
                                       the spec asked for a medikit

## Appendix B: configuration templates

### build/MAP03.cfg (generated, do not hand-edit)

   doom_scenario_path = qa_scenarios_buggy.wad
   doom_map = map03
   doom_skill = 3

   mode = PLAYER
   screen_resolution = RES_160X120
   screen_format = GRAY8
   window_visible = false
   sound_enabled = false
   console_enabled = false

   depth_buffer_enabled = true
   labels_buffer_enabled = true

   objects_info_enabled = true
   sectors_info_enabled = true
   automap_buffer_enabled = false

   render_hud = false
   render_minimal_hud = false
   render_weapon = false
   render_crosshair = false
   render_decals = false
   render_particles = false
   render_effects_sprites = false
   render_messages = false
   render_corpses = false
   render_screen_flashes = false

   episode_start_time = 14
   episode_timeout = 21000
   living_reward = 0

   available_buttons = {
          MOVE_FORWARD MOVE_BACKWARD MOVE_LEFT MOVE_RIGHT
          TURN_LEFT TURN_RIGHT USE ATTACK JUMP
   }

   available_game_variables = {
          POSITION_X POSITION_Y POSITION_Z ANGLE
          VELOCITY_X VELOCITY_Y VELOCITY_Z
          HEALTH ARMOR AMMO2 SELECTED_WEAPON
          KILLCOUNT ITEMCOUNT SECRETCOUNT
          DEAD ON_GROUND ATTACK_READY
          USER1 USER2 USER3 USER4
   }

### configs/t4_goexplore.yaml

   run_name: t4_goexplore
   total_steps: 1_000_000                                       # Kaggle budget; was 2M
   n_envs: 2                                                    # 4 vCPU, two jobs per session
   seed: 0
   device: cuda:0                                               # the sibling job uses cuda:1

   kaggle:

        ckpt_dir: /kaggle/working/ckpt
        ckpt_every: 100_000
        resume: auto                                            # pick up the newest checkpoint on start
        scratch_dir: /kaggle/temp                               # archive saves and demos live here, NOT in
   working
        max_session_hours: 11.5                                 # stop cleanly before the 12 h cap

   maps:
        wad: build/qa_scenarios_buggy.wad
        curriculum:
          - {until: 100_000,                       maps: [MAP01],                           weights: [1.0]}
          - {until: 600_000,                       maps: [MAP01, MAP02, MAP03, MAP04, MAP05, MAP06],
                                                   weights: [0.1, 0.18, 0.18, 0.18, 0.18, 0.18]}
          - {until: 1_000_000, maps: [MAP01, MAP02, MAP03, MAP04, MAP05, MAP06,
   MAP07],
                                                   weights: [0.05, 0.11, 0.11, 0.11, 0.11, 0.11, 0.40]}

   observation:
        depth_size: 64
        frame_stack: 4
        frame_skip: 4
        include_absolute_position: false                              # MUST stay false except in ablation
   E5

   reward:
     alpha: 0.5
        validity:
          precondition: 1.0
          goal: 5.0
          death: -5.0
          time: -0.0005
        adversarial:
          cell_novelty: 0.3
          state_novelty: 0.5
          entity_first_use: 1.0
          portal_first_cross: 1.0
          oracle_cluster: 3.0
          bugbook_hit: 1.5
        novelty_beta_start: 0.5
        novelty_beta_end: 0.1
        novelty_decay_frac: 0.6

   archive:

       enabled: true
       cell_size: 64
       z_bucket: 64
       max_cells: 5_000                                         # was 20k; /kaggle/temp is finite
       p_reset: 0.5
       horizon: 500
       reset_heuristic: mixed                                   # mixed | frontier | uniform
       productivity_guard: true

   bugbook:
       enabled: true
       match_radius_mu: 96
       freq_cap: 20
       checkpoint_every: 50_000

   ppo:
       policy: MultiInputLstmPolicy
       n_steps: 1024                                            # 2 envs x 1024 = 2048 per rollout
       batch_size: 256
       n_epochs: 4
       learning_rate: 2.5e-4
       lr_schedule: linear
       gamma: 0.995
       gamma_int: 0.99
       gae_lambda: 0.95
       clip_range: 0.2
       clip_range_end: 0.1
       ent_coef: 0.01
       ent_coef_end: 0.003
       ent_decay_after: 500_000
       vf_coef: 0.5
       max_grad_norm: 0.5

   oracles:
     stuck:
          window_tics: 70
          intent_threshold: 0.8
          displacement_mu: 24
          min_escape_directions: 3
          conf_saturation_tics: 350
       bounds:
         player_radius_mu: 16
          z_tolerance_mu: 8

       softlock:
          tail_fraction: 0.4
          autocorr_threshold: 0.9
          max_lag_tics: 70
       reach:
         use_attempts_for_broken_trigger: 5

   report:
     dbscan_eps_mu: 96
       dbscan_min_samples: 3
       severity_weights: {oracle: 0.5, blocking: 0.3, frequency: 0.2}
       calibration_maps: [MAP02, MAP04, MAP06]

### Launching two jobs in one Kaggle cell

   export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

   CUDA_VISIBLE_DEVICES=0 nohup python scripts/train.py \
       --config configs/t4_goexplore.yaml --seed 0 \
       > /kaggle/working/logs/job0.log 2>&1 &

   CUDA_VISIBLE_DEVICES=1 nohup python scripts/train.py \
       --config configs/t3b_curiosity.yaml --seed 0 \
       > /kaggle/working/logs/job1.log 2>&1 &

   wait
   tail -n 40 /kaggle/working/logs/job0.log /kaggle/working/logs/job1.log

### Frozen thresholds

Every number under oracles: is frozen at the end of M4. Copy this block verbatim into
 docs/decisions.md with the freeze date and never edit it in response to a result. If a
threshold genuinely must change, that is a new experiment with its own name, re-run
from scratch, and both versions appear in the report.

## Appendix C: Claude Code task prompts

Export this document to Markdown, drop it in the repo as docs/PLAN.md , and start each
session by pointing Claude Code at it. One milestone per session; do not ask for two.

Session opener (use every time):

    Read docs/PLAN.md in full before writing any code. It is the build specification for
    this repository and is normative: MUST means a hard requirement, TODO(human)
    means stop and ask me. We are working on milestone Mx only. Do not start the next
    milestone. When you believe Mx is done, run its acceptance command from the build-
    order table and show me the output.

### Per-milestone prompts

M0 — Set up the repository skeleton exactly as the layout section specifies:
pyproject.toml with the pinned dependencies from the setup section, the full package
tree with empty __init__.py files, .gitignore covering runs/ , build/ and *.wad ,
and tests/test_smoke.py which runs 1000 random steps against ViZDoom's stock
 basic.cfg headless and asserts throughput above 300 steps/s. Report the ViZDoom
version you installed and whether the notifications buffer is available.

M1 — Implement qatester/mapgen/wad.py (a PWAD binary writer with no third-party
dependencies) and qatester/mapgen/udmf.py (a TEXTMAP emitter). Use the minimal
TEXTMAP in Appendix A as the target output. Then write scripts/build_maps.py so
that python scripts/build_maps.py maps/m01_clean.yaml -o build/ produces a
WAD ViZDoom can load. Add tests/test_wad_roundtrip.py which writes a WAD, re-
reads the directory and asserts every lump's name, offset and size. Do not implement
portals or defects yet.

M2 — Implement spec.py , geometry.py and preview.py . Geometry: the wall-segment
splitting algorithm from the compiler section, then portal punching, then vertex
deduplication, then sidedefs and specials. Write tests/test_geometry_invariants.py
asserting all seven correctness invariants from that section. Then author
 maps/m01_clean.yaml through maps/m07_sprawl.yaml to the definitions in the bug-
corpus section, but with empty defects lists for now. Every map must compile and every
preview.svg must render. Flag anything in the specs that you think is geometrically
impossible.

M3 — Implement defects.py with all seven defect kinds as spec-to-spec mutations,
using the injection table in Appendix A, and groundtruth.py emitting the walkable
raster, portal graph, thing inventory and defect list. Add the defects blocks to the map
specs. Write tests/test_defects.py asserting that each mutation changes exactly the
intended element and nothing else, and that the fixed variant of every map passes
 check_reachable() .

M4 — Implement the full qatester/oracles/ package to the oracle-layer section. Build
the synthetic trace fixtures first, in tests/fixtures/traces/ : one positive and two near-
miss negatives per oracle family, written by hand as StepRecord sequences. Do not write
any oracle before its fixtures exist. The acceptance bar is zero findings across the entire
MAP01 CLEAN fixture corpus and a detection in every positive fixture. Then freeze every
threshold into docs/decisions.md with today's date.

M5 — Implement qatester/env/ : QADoomEnv , observation assembly, the 12-action space,
the composite reward, trace recording to Parquet, .lmp recording, and the crash/hang
supervisor. Write tests/test_env_contract.py using Gymnasium's own env checker,
and tests/test_reward_hacking.py with the three scripted adversarial policies from
the reward section. Run scripts/bench_env.py and report steps/s for 1, 4 and 16
workers.

M6 — Implement T1 (random), T2 (scripted wall-follower with USE on contact) and T3a
(PPO, alpha = 1.0), plus the whole qatester/report/ pipeline: clustering, severity,
emission, and the Jinja2 HTML template. Run T1 on MAP03 for 200k steps and show me
the resulting report.html . The report must contain at least one out-of-bounds finding
with a working demo path.

M7 — Add the count-based novelty bonus with globally shared counts, the abstract-state
key, and the Bug Book with its exact add/confirm/decrement/remove rule from the
coverage section. Write tests/test_bugbook_rule.py covering all four transitions
including removal at frequency zero. Then run T3a and T3b on MAP07 for 600k steps
across 3 seeds and report median cell coverage for each with a Mann-Whitney U test.

M8 — Implement qatester/archive.py and qatester/agents/goexplore.py : the cell
archive over DoomGame.save / load , all three reset heuristics plus uniform, the ReMiDi-
style productivity guard, and LRU eviction at 20,000 cells. Benchmark save and load
latency before building the training loop on top of them and tell me the numbers; if a
load takes more than 50 ms we switch to the replay-based fallback in risk R3. Treat every
reset as an episode boundary for GAE.

M9 — Implement scripts/evaluate.py and scripts/make_figures.py , then run
experiments E1 through E7 from the evaluation section. make_figures.py must
regenerate every table and figure from results/ alone with no manual steps. Fit the
temperature calibration on MAP02/04/06 and report ECE before and after on
MAP03/05/07.

### Working agreements to repeat to the agent

     Never let anything under qatester/agents/ import ground truth.
     Never change an oracle threshold after M4 without recording it in
      docs/decisions.md .
     Every new random source takes an explicit seeded Generator .
     When something in the plan turns out to be wrong — and some of it will be — say so
     and propose the change rather than silently working around it.
