# The project explained: concepts, how we built it, what we found

*RL-based QA tester for games, Phase 1 (ViZDoom). Team 19.*

This document explains the whole project in plain language: the ideas behind each part,
how each part was put together, what we measured, and what we could **not** conclude.
It deliberately avoids code. For the code, see [ARCHITECTURE.md](ARCHITECTURE.md); for
every individual decision and its evidence, see [decisions.md](decisions.md) (cited below
as D1-D51).

---

## Contents

1. [The problem and the goal](#1-the-problem-and-the-goal)
2. [The central idea: planted bugs and a hidden answer key](#2-the-central-idea-planted-bugs-and-a-hidden-answer-key)
3. [Building the levels](#3-building-the-levels)
4. [The game as a reinforcement-learning environment](#4-the-game-as-a-reinforcement-learning-environment)
5. [Detecting bugs: the oracles](#5-detecting-bugs-the-oracles)
6. [The testers](#6-the-testers)
7. [From detections to a bug report, and scoring it](#7-from-detections-to-a-bug-report-and-scoring-it)
8. [Repairing bugs: the Fixer loop](#8-repairing-bugs-the-fixer-loop)
9. [How we worked: compute, validation, honesty](#9-how-we-worked-compute-validation-honesty)
10. [Milestones and results](#10-milestones-and-results)
11. [What went wrong, and what we learned from it](#11-what-went-wrong-and-what-we-learned-from-it)
12. [What we could not identify or conclude](#12-what-we-could-not-identify-or-conclude)
13. [Glossary](#13-glossary)

---

## 1. The problem and the goal

**Game testing (QA) is mostly manual.** People play builds for hours looking for things
that are broken: a wall you can walk through, a pit you can't climb out of, a door that
never opens, a key that doesn't do anything. Automating this is attractive and hard: a
tester has to *play* the game well enough to reach the broken parts, and *recognise* that
something is broken without anyone telling it what "broken" looks like on that level.

**Our goal for Phase 1:** build an automatic tester for 3D first-person levels, using the
classic game Doom through **ViZDoom** (a research platform that lets programs play Doom),
and **measure honestly** how well it finds bugs, including on levels it has never seen.

Two questions run through the whole project:

- **Reaching:** can the tester get to the parts of the level where bugs are?
- **Judging:** when it gets there, can it tell a real bug from normal play, without
  crying wolf?

The original build plan ([PLAN.md](PLAN.md)) proposed a ladder of testers, from random
button-pressing up to reinforcement learning with curiosity and Go-Explore, plus a set of
experiments to compare them. We followed it milestone by milestone (M0-M9), recording
every change from the plan and why.

---

## 2. The central idea: planted bugs and a hidden answer key

You can't measure a bug finder on a level whose bugs you don't know. So we **make the
bugs ourselves**:

1. Build a level as intended (the **fixed** build).
2. Deliberately break it in known ways (the **buggy** build), and write down exactly what
   was broken and where: the **answer key** (we call it the *ground truth*).
3. Let a tester play the buggy build **without ever seeing the answer key**.
4. Afterwards, compare the tester's bug reports with the answer key.

This gives two numbers, the standard pair for any detector:

| Measure | Question it answers | Bad when… |
|---|---|---|
| **Precision** | Of the bugs it *reported*, how many were real? | it cries wolf (false alarms) |
| **Recall** | Of the bugs we *planted*, how many did it find? | it misses things |

A tester that reports everything has perfect recall and useless precision; one that
reports nothing has no false alarms and finds nothing. We want both high.

**Keeping the key hidden is enforced, not just promised.** The tester and the detectors
receive only a description of the *intended* level (where the rooms and doors should be),
never the list of planted bugs. An automated check fails the build if any tester code
even imports the answer key (D8, D22).

The **fixed** build gives us a second, simple test: run any tester on a bug-free level,
and *every* report it makes is, by definition, a false alarm.

---

## 3. Building the levels

### 3.1 How a Doom level is made

A Doom level is a 2D floor plan with heights. Its building blocks:

- **Sectors:** areas of floor, each with a floor height and a ceiling height. A room is
  one or more sectors; a door is a thin sector whose ceiling starts at the floor (shut)
  and rises when opened; a lift is a sector whose floor moves.
- **Lines (linedefs):** the edges between sectors. A one-sided line is a solid wall; a
  two-sided line can be walked across, unless it is flagged *blocking*.
- **Specials:** behaviour attached to lines, such as "when pressed, open door with tag
  12", "when walked over, teleport", "when pressed, lower lift". Specials find their target
  through **tags** (a number shared by the switch and the sector it controls).
- **Things:** objects placed in the level: the player start, keys, health packs, switches,
  the exit, monsters.

### 3.2 From a simple description to a real level

Writing Doom geometry by hand is error-prone, so we describe a level at a higher level,
as a list of rectangular **rooms** (with floor and ceiling heights), **portals** between
them (an opening, a door, a locked door that needs a key, a lift, a teleporter, or a door
opened by a remote switch), **things**, and the **goal** (reach the exit, perhaps after
collecting a key). A compiler turns this into real Doom geometry: it lays the rooms on a
grid, separates neighbours by a thin wall strip, turns each portal into the right sectors
and lines with the right specials and tags, and writes a level file ViZDoom can load,
plus the description of the intended level that the detectors use.

### 3.3 Hand-made maps and generated maps

- **8 hand-made maps** (MAP01-MAP08), each built to exercise particular features: MAP01 is
  clean (no bugs at all, a false-alarm check), MAP07 SPRAWL is the big 12-room benchmark
  with 4 planted bugs, and MAP08 HOLDOUT is never used for training.
- **A procedural generator** makes unlimited new levels from a number (the *seed*): it
  picks a grid of 5-14 rooms, connects them with doors, locked doors and keys, switch
  doors, lifts and teleporters, makes sure the goal is reachable, and plants whichever bug
  types are requested.
- **Train/test split.** Generator seeds are divided into a *training* set and a *test*
  set. Testers may be developed and trained on training maps; the test maps are kept for
  measuring **generalisation**, i.e. how well a tester does on levels nobody tuned it on
  (D18, D31). This was a hard requirement from the start: a tester that only works on
  the maps it was trained on is not a tester.

### 3.4 The seven bug types

Each bug is planted by changing the *compiled* level in one specific way (D20):

| Bug | What it is in the game | How it's planted | Which detector should catch it |
|---|---|---|---|
| **Leaky wall** | you can walk through a wall and out of the level | a wall segment loses its *blocking* property, with a pocket behind it | out of bounds |
| **Trap pit** | you fall into a pit and can never climb out | a room's floor is lowered beyond the maximum step-up height, with no lift | stuck |
| **Dead switch** | a switch that opens nothing | the switch's tag points at the wrong sector | unreachable region |
| **Broken lock** | a locked door opens without its key | the door's lock requirement is removed | game-rule violation |
| **One-way lift** | a lift takes you down but can't be called back up | the lift's trigger on the lower side is removed | softlock |
| **Teleport loop** | two teleporters bounce you back and forth forever | each teleporter's destination is moved in front of the other's pad | softlock |
| **Bad pickup** | a health pickup pushes health above the maximum | the item is swapped for an over-the-cap one | game-rule violation |

Every planted bug is also **confirmed in the real game engine** by an automated scenario
(for example: walk to the leaky wall and check the player ends up outside), so we know
each bug really exists and really behaves as described.

---

## 4. The game as a reinforcement-learning environment

### 4.1 Reinforcement learning in one paragraph

In **reinforcement learning (RL)** an *agent* repeatedly observes the situation, chooses
an *action*, and receives a *reward*. One play-through, from start to the end (goal,
death, or time running out), is an *episode*. Over many episodes the agent learns a
*policy* (a way of choosing actions) that collects as much reward as possible. The
reward is the only thing that tells the agent what we want, so designing it well is
half the work.

### 4.2 What our agent sees, does, and gets

- **Actions:** 14 button combinations (move forward or back, strafe, turn, use, jump,
  attack, and combinations like forward-and-use). Each chosen action is held for 4 game
  frames (*frame skip*), which makes play faster and smoother; the game runs at 35 frames
  ("tics") per second.
- **Observation:** a small **depth image** (how far away walls are in each direction,
  64×64 pixels, the last few frames stacked) plus **18 numbers**: health, armour, ammo,
  keys held, velocity, whether it is on the ground, time used, how new its current spot
  is, and whether something usable (a door, lift or switch) is right in front of it.
- **Reward:** described in section 6.3.

The environment also does the bookkeeping on every step: it records a precise snapshot
of the player (position, speed, health, keys, which buttons were pressed, the current
heights of every door and lift), feeds it to the bug detectors, tracks how much of the
map has been visited, records a replayable recording of the episode, and restarts the
game engine if it ever hangs.

### 4.3 Coverage: how much of the level was explored

Coverage is measured on a grid of 64×64-unit floor **cells**: the share of walkable cells
the tester has visited. We also track *entity coverage* (share of interactive objects
used), *connection coverage* (share of doors/lifts/teleporters crossed) and the number of
distinct *situations* seen (place × keys held × which doors are open). Visit counts are
shared between all parallel copies of the game, so "new" means new for the whole team of
workers.

---

## 5. Detecting bugs: the oracles

### 5.1 What a test oracle is

In software testing, an **oracle** is whatever decides whether an observed behaviour is
correct. Our oracles watch the stream of player snapshots and fire when something
contradicts how the level is *supposed* to work. They know the intended level (where
rooms, walls and doors are) but never the planted bugs.

### 5.2 The six detector families

| Family | Fires when… | Typical bug |
|---|---|---|
| **Crash** | the engine crashes, a step hangs, or game time stops advancing | engine faults |
| **Stuck** | the player keeps pushing in several directions on the ground and doesn't move; or stands in an area much lower than everything around it, with no lift | trap pit |
| **Out of bounds** | the player is outside every room, below a floor or above a ceiling, or crosses a wall faster than running allows | leaky wall |
| **Unreachable region** | at the end of a run: some rooms were never visited, and every way into them was *properly tried and failed* repeatedly | dead switch |
| **Game-rule violation** | an invariant breaks: health above the cap, keys appearing from nowhere, a locked door passed without its key, more kills than monsters | broken lock, bad pickup |
| **Softlock** | an episode ends by time-out with the player confined to one room for its final stretch, after properly trying every exit; or the player bounces between teleporters | one-way lift, teleport loop |

### 5.3 The key principle: evidence about the level, not about the player

A naive detector would flag "the player stayed in one room for a long time" as stuck.
But an agent that simply isn't very good also stays in one room. That would make the
detector measure the *tester's* skill, not the *level's* correctness.

So every detector requires **evidence that the level is broken**. The clearest example is
the idea of a **proper failed attempt**, used by the unreachable and softlock families. A
door only counts as "tried and failed" if:

- the player actually *pressed* USE (Doom acts on the press, not while the button is held);
- the press really reached the door's usable face (not a corner, not the wall beside
  it, not a switch on the other side of a wall) from where the player stood when the game
  processed it;
- the door or lift wasn't already moving;
- for a locked door, the player held the key;
- and the door then didn't move.

Most of our detector improvements over the project (D36, D41, D44, D47) tightened exactly
this definition, each time after checking in the real engine what Doom actually does.

### 5.4 Frozen thresholds

Detectors have numeric thresholds (how many tics of pushing count as stuck, how far is
"within reach"…). We **froze** them at the end of milestone M4 (D27). Any later change
needs a recorded decision backed by engine evidence, never "it improves the score",
because tuning a detector until the results look good would make the results
meaningless.

---

## 6. The testers

We built testers of increasing sophistication, so each could be compared with the simpler
ones (the plan calls them tiers T1-T4).

### 6.1 T1 Random and T2 Scripted: the baselines

- **Random** presses a random button each step. It is the floor every other tester must
  beat, and it is surprisingly good at stumbling into some bugs.
- **Scripted** is a hand-written rule-follower: keep a wall on your right, turn into
  openings, press USE on anything usable or on a wall at arm's length, back off when stuck.
  It needs no learning and is a "classic automated testing" strawman.

### 6.2 T3a: a learning agent (PPO with memory)

**PPO (Proximal Policy Optimisation)** is a widely used RL algorithm: the agent plays a
batch of steps, then nudges its policy towards actions that turned out better than
expected, but only a small step at a time, which keeps training stable.

Our agent is a small neural network: an image part reads the depth frames, a second part
reads the 18 numbers, and an **LSTM** (a memory layer) carries information across steps,
because a game is only *partially observable*: you can't see the key you passed two
rooms ago, but you need to remember it. Training ran on Kaggle GPUs.

**T3a is rewarded only for playing properly**: getting keys, opening the right doors,
reaching the exit, not dying, not wasting time. It is the "good player" baseline.

### 6.3 Reward design: what we reward and how we stop cheating

The reward is a weighted mix of two parts: `reward = α × (play properly) + (1 − α) × (explore / hunt bugs)`.

- **Play properly:** a bonus for each key or critical door, a big bonus for the exit, a
  penalty for dying, a tiny time cost.
- **Shaping:** a reward for getting closer to the goal along the level's route. This is
  *potential-based shaping*, a standard technique where the bonus for moving closer is
  exactly cancelled by the penalty for moving away, so the agent can't farm it by pacing
  back and forth, and the best policy is unchanged. It was essential: without it the
  agent learned nothing on even the simplest map (D35).
- **Explore / hunt bugs:** bonuses for new places, new situations, first use of an object,
  first crossing of a door or lift, each new bug found, and returning to known bugs (the
  Bug Book, below).

RL agents are notorious for **reward hacking**, finding loopholes that score points
without doing what we meant. We built in three protections, each with tests: a spot only
pays its novelty bonus again after the agent has properly left it (no pacing on a
threshold), a bug pays only once per episode (no farming the same bug), and death is
penalised (no suiciding to restart).

### 6.4 T3b: curiosity and the Bug Book

**Count-based curiosity** rewards the agent for going where it has rarely been: the bonus
for a place shrinks as `1/√(visits)`. The idea, from the exploration literature, is that a
curious agent covers more of the level and therefore reaches more bugs.

The **Bug Book** is a memory of bugs found so far (modelled on the "Mistake Book" in
earlier work). A newly found bug is added; finding it again raises its count (up to a
cap); a replay that no longer reproduces it lowers the count, and at zero it is removed.
It gives the agent a reward target (revisit known bugs to confirm them) and is the basis
of a regression check (do bugs disappear once the level is fixed?).

T3b = PPO + curiosity + Bug Book, with α = 0.5. Whether this beats T3a was milestone M7.

### 6.5 Curriculum: learning across many levels

To learn something general, the curiosity agent trained on a **curriculum**: first the
simple MAP01, then a mix of hand-made maps and 64 generated training maps, with MAP07
added at the end. The environment switches levels between episodes according to the
current stage's weights.

### 6.6 T4: Go-Explore

**Go-Explore** (Ecoffet et al.) tackles the hardest exploration problems with a simple
idea: **remember where you've been, return there, and explore from there.** Instead of
always starting from the spawn point, it keeps an **archive** of interesting places
together with a way to get back to each one. ViZDoom can *save* and *load* the game, so
returning is instant.

The loop:

1. Half the time, pick a place from the archive and **load** its save; otherwise start a
   fresh episode.
2. **Explore** from there for 500 steps.
3. Every time a **new place** is reached (or a known one reached earlier than before),
   **save** the game into the archive.
4. Credit any new places found back to the place it restarted from.

The details that make it work:

- **What counts as a "place" (a cell):** the 64-unit floor square, the height level, the
  keys held, **and which doors are open**. The door state was our addition: on MAP07
  the first door closes about 37 steps after opening, and random play rarely gets through
  in time. With the door state in the cell, the very moment a door opens is saved, so
  later restarts begin with it already open (D46).
- **Choosing where to restart:** prefer places that were visited rarely, places next to
  unexplored areas (the *frontier*), and places whose restarts have paid off before. The
  last one, the **productivity guard** (after ReMiDi), stops the archive wasting restarts
  on a dead-end corner that only *looks* like a frontier.
- **The explorer between restarts** is not a neural network: it repeats its last action
  most of the time (smoother movement) and presses USE whenever something usable is in
  front. This is the original Go-Explore's exploration phase. It needs no GPU, and it
  shows clearly that the *archive* is what solves exploration.
- **Patience** (D48, added by our teammate): a "stuck forever" bug can only be confirmed
  when an episode *times out* with the player stuck, but Go-Explore restarts every 500
  steps. Patience gives 10% of the steps to runs that play until time runs out, starting
  from places whose restarts never got far (places that look like traps). With patience,
  Go-Explore found all 4 of MAP07's bugs.

Saving takes about 4 ms and loading about 9 ms per state, and a save is 37 KB, so
returning anywhere is effectively free.

---

## 7. From detections to a bug report, and scoring it

- **Clustering:** one real bug triggers many detections (every time the tester walks
  through the leaky wall). Detections of the same family close together are grouped into
  one candidate bug (DBSCAN clustering, 96-unit radius).
- **Severity:** candidates are ranked by detector family, whether they block progress, and
  how often they occur.
- **The report:** a machine-readable `report.json` and a human-readable `report.html` per
  level, with a map, each bug's location, its evidence, and a **replay** (a Doom recording
  that reproduces it), so a developer can watch exactly what happened.
- **Scoring against the answer key:** a reported bug counts as finding a planted bug only
  if it comes from the detector family that bug *should* trigger (a trap reported as
  "confined" is also accepted) **and** lies within 128 map units of it. Requiring the
  right family prevents claiming a detection because an unrelated detector happened to
  fire nearby.

---

## 8. Repairing bugs: the Fixer loop

*(Built by our teammate; D49-D51, [FIXER_PLAN.md](FIXER_PLAN.md).)*

Finding bugs is half of QA; the natural next step is fixing them. The Fixer loop is:

```
QA (Go-Explore + detectors) reports a bug
   → the Fixer proposes an edit to the level
   → the level is rebuilt and QA re-tests
   → keep the edit only if the bug is gone, coverage didn't collapse,
     and no new bug appeared; otherwise undo it
```

- **What the Fixer may do:** six generic edit operations on the compiled level: set a
  line's flag, change a special's argument (for example a tag), copy a special to another
  line, change a floor height, change an item's type within its class, move a teleport
  destination. None is specific to a bug type, and **nothing can be deleted**, because
  deleting is the easiest way to *hide* a bug instead of fixing it.
- **The edit menu:** from one QA report and the level alone, list the parts of the level
  worth editing near the report (plus switches whose tag points there, since a dead switch
  is reported at the door it fails to open, which may be rooms away), and the candidate
  edits for each. The correct fix was in the menu for 114 of 114 test bugs.
- **The current Fixer is not learned:** it tries candidates nearest-first. A learned Fixer
  (a language model trained with reinforcement learning, GRPO, rewarded by QA's re-test)
  is designed but not built.
- **Cheating found and closed (D51):** the first run "fixed" a teleport loop by walling
  off the teleporter (which also hid another bug behind it), and "fixed" a one-way lift by
  raising the room under it so the lift became pointless. Both passed the automated
  check. The menu no longer offers such edits. This is a textbook example of an optimiser
  exploiting a loose reward, caught by reading the fixes.
- **Results:** on 8 generated levels with 16 planted bugs, playing rounds until QA is
  quiet, **13 of 16 bugs were truly repaired** (checked by replaying each bug's engine
  scenario on the final level), and every level could still be finished.

---

## 9. How we worked: compute, validation, honesty

### 9.1 Where things ran

- **Local laptop (CPU):** building maps, the detectors, all tests, the demo, short tester
  runs.
- **Kaggle GPU sessions (2× T4, 30 hours per week):** PPO training.
- **Kaggle CPU sessions (free):** long random, scripted and Go-Explore runs.

A launcher script packages the current code, generates a notebook, runs a **pre-flight
test** before any long job, can **stop a GPU job early** if the agent isn't making progress
(a progress gate), and has a **watchdog** for runs that finish but don't exit. These were
added after early GPU runs were lost to crashes, hangs and wrong rewards.

### 9.2 How we checked the detectors

Detector correctness was verified in four independent ways:

1. **Hand-written traces:** for each detector, a scripted sequence that *should* fire it,
   and near-misses that *shouldn't* (a legitimate teleport, healing to exactly 100…).
2. **Engine scenarios:** each planted bug played in the real engine and detected.
3. **Replays:** every bug report from a real training run replayed through the current
   detectors (89 of 90 replay exactly). When we changed a detector, real finds had to
   stay and false ones had to go.
4. **Bug-free sweeps:** testers on the fixed builds of 104 levels for 1.76 million steps.
   Any report there is a false alarm. Final result: **0**.

### 9.3 Statistics

Comparisons between testers use **5 independent seeds** each and a one-sided
**Mann-Whitney U test** (which doesn't assume normally distributed results), plus the
**A12 effect size** (the probability that a random run of one tester beats a random run of
the other). With 5 seeds a side, the smallest possible p-value is 0.004; with 3 it would be
0.05, which can never be significant, so we used 5 (D42).

### 9.4 Honesty rules

- The answer key never reaches testers, detectors or the Fixer (checked automatically).
- Detector thresholds are frozen; changes need engine evidence.
- Every false-alarm fix has a regression test that fails without the fix.
- Failed experiments are reported as results, not hidden or re-tuned until they pass.
- Every deviation from the plan is logged with its reason (D1-D51).

---

## 10. Milestones and results

### 10.1 Milestones

| Milestone | What | Outcome |
|---|---|---|
| M0-M3 | setup, level compiler, 8 hand-made maps, planting bugs, answer keys, generator | ✅ |
| M4 | the six detector families | ✅ (later hardened) |
| M5 | RL environment, reward, coverage | ✅ |
| M6 | baselines running; PPO learns to finish MAP01 | ✅ 100% and 90% of evaluation episodes reach the exit |
| M7 | curiosity explores significantly more than plain PPO | ❌ not significant (see 10.2) |
| M8 | Go-Explore explores ≥ 80% of MAP07 | ✅ 92% on all 5 seeds |
| M9 | full evaluation | partial: main comparison, unseen maps, figures; ablations not run |
| — | Go-Explore patience; the Fixer loop; the QA Lab demo app | ✅ |

### 10.2 The main comparison: MAP07 (12 rooms, 4 planted bugs)

| Tester | Steps | Map explored (5 seeds) | Planted bugs found (of 4) | False alarms |
|---|---|---|---|---|
| Random | 1M | 40% | 0.6 | 0 |
| Scripted | 1M | 46% | 0.8 | 0 |
| PPO (play properly) | 600k | 64% | ~1 | 0 |
| PPO + curiosity | 600k | 60% | ~1.3 | 0 |
| **Go-Explore** | 1M | **92%** | **3 every seed** | **0** |
| **Go-Explore + patience** | 1M | **92%** | **4 every seed** | **0** |

- Go-Explore vs both PPO and Random: p = 0.004, A12 = 1.00 (every Go-Explore seed beats
  every other seed).
- Curiosity vs plain PPO: p = 0.50, A12 = 0.52 (no detectable difference). The result was
  **bimodal**: two curious runs were the best PPO runs of all (82%, 79%), one never left
  the starting third of the map (33%).

### 10.3 Maps no tester was tuned on (39 maps, 66 planted bugs)

| Tester | Precision | Bugs found |
|---|---|---|
| Random | 98% | 77% |
| Scripted | 98% | 73% |
| Go-Explore + patience (the 7 hand-made maps of the set) | 93% | 10 of 11 |

Easy bug types: trap pits, broken locks, teleport loops (about 90% found). Hardest: dead
switches (0-25% found), because the tester must press the right switch and then notice the
door didn't open, several times.

### 10.4 The trained curiosity agent across 71 training maps

Precision **100%** (51 of 51 reports real); recall **17-24%**. It judges well but doesn't
reach most bugs, and it never learned to finish these levels.

### 10.5 The Fixer loop

13 of 16 planted bugs repaired across 8 generated levels, all levels still finishable,
after closing two "cheating" repairs.

---

## 11. What went wrong, and what we learned from it

| Problem | What happened | What we did | Lesson |
|---|---|---|---|
| **Sparse reward** | the first PPO agent learned nothing on MAP01 | goal-distance shaping, turn-in-place actions (D35) | an agent can't learn from a reward it never sees |
| **Evaluation bug** | PPO looked like it failed (0-10%) while training said 91-99% | the evaluation was missing the agent's novelty inputs (D39) | test the measurement, not just the model |
| **Lost runs** | a session restart lost results; a finished run hung for 10 hours | unattended notebooks, a watchdog, forced exit (D37, D38) | long jobs need to survive the infrastructure |
| **Curiosity swamped the task** | novelty was 20-100× the task reward; the agent wandered and never used keys | scaled novelty to 5% (D44) | balance reward parts before training |
| **Half the bug reports were false** | 49 of 90 in one run: teleports mistaken for wall-walking, presses the engine never accepted… | replay-driven diagnosis, engine checks, tighter "proper attempt" (D44, D47) | an oracle must check for *evidence about the level*, verified in the engine |
| **Engine quirks** | long file paths hang ViZDoom; a save after recording freezes the clock; a save costs 2 tics; engines racing at start-up | short paths, no recordings in Go-Explore, tic-aware margins, staggered starts (D34, D43, D46, D47) | the simulator is part of the system under test |
| **Reward hacking in the Fixer** | walling off a teleporter, flattening a lift | menu rules forbidding such edits (D51) | read what the optimiser actually did |

---

## 12. What we could not identify or conclude

This section matters as much as the results. These are the questions the project leaves
open, and why.

### 12.1 About exploration and learning

1. **Does curiosity help at all?** We can only say that *this* curiosity design (count-based
   novelty at 5% weight, with a Bug Book) did not significantly beat plain PPO on *one*
   map at *600k steps* with *5 seeds*. A different bonus size, a longer budget, or a
   different novelty signal might behave differently; the bimodal result (two excellent
   runs, one stuck) suggests high variance rather than no effect. We did not run the
   reward-balance sweep (E3) that would show how sensitive this is, and re-tuning after
   seeing the result would have been tuning to the test.
2. **Would Go-Explore with the trained agent beat Go-Explore with random moves?** The
   plan's version explores from each restart with the PPO policy. We only built the
   random-explorer version, so we can't say how much a learned explorer would add.
3. **Which Go-Explore ingredients matter?** We never compared the restart heuristics
   (E4) or removed the productivity guard or the Bug Book (E5). The door-state cell is
   the only ingredient we showed to be necessary (exploration stalled at 33% without it).
4. **Does the trained agent generalise to unseen maps?** We measured generalisation for
   the random, scripted and Go-Explore testers, but the trained PPO agent was never
   evaluated on the unseen test maps. Its recall on its own training maps (17-24%) suggests
   it would not do well, but that is not measured.
5. **Why does exploration stall on some generated levels?** Some levels open a door with
   a switch in a *different* room; random-style exploration rarely presses it. We observed
   this but didn't quantify how often generated levels contain such puzzles, or how
   exploration scales with puzzle depth.

### 12.2 About detection

6. **Two unexplained false alarms.** On unseen generated maps with no planted bugs, one
   "confined" (in a lift's lower room) and one "wedge" (in a corner) were reported. We did
   not replay them, so we don't know whether each is a remaining detector edge case or a
   **real, unplanted flaw** produced by the level generator. Either answer is interesting.
7. **Is our scoring too strict?** On MAP06 the one-way lift was reported, at its exact
   location, by the "unreachable" detector ("the lift fails from the pit") instead of the
   expected "softlock" detector, and so counted as a false alarm. Whether several
   detector families should be accepted for a bug type is a judgement call we left open
   rather than change the rule after seeing results.
8. **How trustworthy are the reported confidence values?** Each report carries a
   confidence, but we never **calibrated** them (fitting how often a 0.9-confidence report
   is actually right). They are rankings, not probabilities.
9. **Do fixed bugs stop being reported?** The Bug Book's "decrement and remove" path, and
   the regression experiment that replays a buggy build's bugs on the fixed build (E7),
   were not run at scale.
10. **Dead switches remain hard.** Recall for dead switches stayed between 0% and 25%.
    We don't know whether the bottleneck is reaching the switch, pressing it correctly,
    or the five-failed-attempts rule the detector uses.

### 12.3 About the Fixer

11. **Are the repairs what a designer would do?** A kept fix removes the bug and keeps the
    level playable, but may not match the designer's intent (a pickup "fixed" by swapping
    it for a smaller one, not the original item). One extra edit was kept in a later round
    after a report whose reality wasn't checked.
12. **Would a learned Fixer beat the simple one?** The non-learned Fixer already succeeds
    for most bug types because the right fix is usually the nearest; the learned Fixer
    (an LLM trained with GRPO) is designed but not built. In our teammate's separate
    LLM experiments, the trained planner barely beat the untrained model, and the full
    co-evolution run (Fixer vs a bug-planting adversary) was not completed, so whether
    co-evolution works here is **open**.
13. **Damage the guard can't see.** The coverage check (keep ≥ 80% of the old coverage) was
    too weak to catch the "wall off the teleporter" shortcut. A stronger check (no room
    that was reachable may become unreachable) was designed, not built.

### 12.4 About the evaluation itself

14. **Statistical power.** 5 seeds per tester and one main benchmark map (MAP07) are enough
    to show large effects (Go-Explore) but not small ones (curiosity). Some numbers rest
    on a single seed (the unseen-map runs, the Fixer batches).
15. **Our bugs, our levels.** All bugs are of 7 types we defined, planted in levels our
    own generator made. Real game bugs are messier (physics glitches, scripting errors,
    performance problems, visual artefacts), and real levels are far more complex. We
    can't claim these results transfer to commercial games.
16. **Compute limits.** PPO budgets (600k-2M steps) were set by the weekly GPU quota.
    Longer training might change the PPO results; we can't say in which direction.
17. **Go-Explore on all 39 unseen maps.** At the time of writing, the run of Go-Explore +
    patience on the 32 generated test maps was still in progress; only its 7 hand-made
    maps are reported.

### 12.5 The honest one-line summary

> We built a complete, honest test bench for automatic game QA. On it, **detecting** bugs
> reliably is solved for our bug types (near-zero false alarms), **reaching** bugs is the
> real problem, curiosity alone did not solve it, and Go-Explore's "remember and return"
> did. Whether learning-based testers can match or beat Go-Explore, and whether any of this
> transfers to real games, remain open.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| A12 | effect size: probability that a random run of A beats a random run of B (0.5 = no difference) |
| Answer key / ground truth | the record of which bugs were planted where; hidden from testers |
| Archive (Go-Explore) | the saved places Go-Explore can return to |
| Bug Book | memory of found bugs, used as a reward target and for regression checks |
| Cell | a 64×64-unit square of floor (plus height, keys and open doors in Go-Explore) |
| Coverage | share of the level's walkable cells visited |
| Curriculum | training on easier levels first, then a growing mix |
| Episode | one play-through, from start to goal, death or time-out |
| Frame skip | holding each action for 4 game frames |
| GRPO | a reinforcement-learning method for training language models from rewards |
| LSTM | a neural-network layer with memory across time steps |
| Mann-Whitney U | a statistical test comparing two groups without assuming normal distributions |
| Oracle | a check that decides whether observed behaviour is a bug |
| Patience | Go-Explore runs that play until the time-out, to confirm "stuck forever" bugs |
| Potential-based shaping | a progress reward that can't be farmed and doesn't change the best policy |
| PPO | Proximal Policy Optimisation, the RL algorithm used for the learning testers |
| Precision / Recall | share of reports that are real / share of planted bugs found |
| Productivity guard | lowers the chance of restarting from places that never lead anywhere new |
| Reward hacking | an optimiser scoring points through a loophole instead of the intended behaviour |
| Softlock | the player is alive but can never progress |
| Tag | the number linking a switch or line to the sector it controls |
| Tic | one Doom game frame (1/35 s) |
| UDMF | the text format of the Doom levels we compile |
