# Fixer loop: endless QA → fix → re-test co-evolution (design, for sign-off)

Status: **proposal**, nothing built yet. Builds on the Phase-1 stack in this repo (levels,
planted bugs, answer keys, detectors, Go-Explore) and reuses the LLM + GRPO trainer from
`game-qa-coevolution` (`qa_llm.py`).

## 1. Goal

A loop that never stops:

```
 ┌─► generator: new level + hidden planted bugs (procgen seed)
 │        │
 │        ▼
 │   QA: Go-Explore + the 6 detectors play the level -> bug reports
 │        │                                 (answer key never seen)
 │        ▼
 │   FIXER (LLM): reads one report + the level near it -> edits the level
 │        │
 │        ▼
 │   rebuild the WAD, QA re-tests -> reward the Fixer
 │        │
 │   inner loop: next report, until no reports are left or the attempt budget runs out
 └── outer loop: next level, forever. The Fixer (and later the Planter) keeps training.
```

The answer key is used **only to score** the loop for us (did the fix really repair the
planted bug?). It never reaches the Fixer, its reward, or QA (same rule as Phase 1, D22).

## 2. The roles

| Role | What it is | Learns? | Stage |
|---|---|---|---|
| Generator | `procgen.generate(seed)` with random bug kinds | no | 1 |
| QA | Go-Explore (T4, 92% of MAP07) + the 6 frozen detectors (D27) | no | 1 |
| **Fixer** | Qwen2.5 + LoRA adapter, GRPO | **yes** | 1 |
| **Planter** | second LoRA adapter on the same Qwen: chooses which bugs to plant | **yes** | 2 |
| QA explorer | a learned explorer instead of Go-Explore | optional | 3 |

Why QA stays frozen at first: the detectors' thresholds are frozen by rule (D27) and have
0 false alarms on 104 bug-free maps, and Go-Explore is our best tester. A frozen, honest
judge is what makes the Fixer's reward trustworthy. The co-evolution (the arms race) comes
from the Planter in stage 2: it is rewarded for bugs the Fixer fails to repair, the Fixer
is rewarded for repairing them, and neither runs out of new problems.

## 3. What the Fixer edits

Bugs are planted in the **compiled level**, not the YAML (D20): a wall's flags, a switch's
target tag, a door's lock number, a sector's floor height, a thing's type or position. The
YAML still describes the *intended* level, so editing YAML could not fix anything.
The Fixer therefore edits the compiled level's `udmf.TextMap`
(`CompiledMap.textmap()`), the same "source code" a Doom level designer edits, then
`wad.write_wad` rebuilds the map.

### Edit operations (generic, none is specific to a bug type)

| Op | Fields | Fixes, for example |
|---|---|---|
| `set_flag(linedef, flag, on)` | one of `udmf.LINEDEF_FLAGS` | a wall you can walk through (missing_blocking) |
| `set_arg(linedef, i, value)` | special argument i | a switch with the wrong tag (wrong_tag), a door's lock (wrong_special) |
| `copy_special(from_linedef, to_linedef)` | special + args | a lift with no trigger on one side (one_way_lift) |
| `set_floor(sector, height)` | heightfloor | a pit with no way out (geometry_trap) |
| `set_thing_type(thing, type)` | within the thing's class (health items, keys…) | an over-cap health pickup (bad_pickup) |
| `move_thing(thing, x, y, angle)` | for teleport destinations | teleporters that loop (teleport_loop) |

Deleting walls, sectors or things is **not** offered: it is the easiest way to make a bug
unobservable instead of fixed.

### The edit menu (how an LLM picks edits with exact probabilities)

For a report at position p, `menu.py` lists the entities within 256 mu of p (linedefs,
sectors, things) and, for each, the candidate edits built from **the level alone**:
- linedefs: toggle each blocking flag; set a door/switch argument to the tag of each nearby
  door sector, or the lock to each key the level contains; copy the special of the
  matching linedef on the other side of the same opening;
- sectors: set the floor to each neighbouring sector's floor;
- things: change to another type of the same class; move a teleport destination to a
  standard spot (the room centre side of its pad, facing into the room).

The Fixer chooses in two steps, each a single-token choice from a lettered list:
**entity** (≤ 26 options + `done`), then **edit** for that entity (≤ 26). Up to 3 edits per
attempt. This is exactly the choice-from-options form `TwoPlayerLLM.choose/update` already
supports (exact log-probabilities, one forward pass per decision, KL to the frozen base),
and it can never produce an invalid level. Free-form patch generation is a possible v2.

### What the Fixer reads (prompt)

The report (detector family, kind, position, a one-line evidence summary), then a short
text listing of the nearby entities with their fields ("linedef 412: two-sided, NOT
blocking, between sector 31 (floor 0) and sector 57 (floor 0, the outside)"). Never the
answer key, never the YAML defect list. A test (extending `test_no_groundtruth_leak.py`)
enforces that nothing under `qatester/fixer/` reads defects.

## 4. Verification and the Fixer's reward

Each attempt is checked by re-testing the patched level with **QA only**:

1. **Baseline:** before any fix, QA runs `k` seeds (start: `k = 3`, 20k steps each) and
   records which reports reproduce. Only reports found in at least 2 of 3 seeds are sent
   to the Fixer, so "gone after the fix" means something.
2. **Re-test:** the same `k` seeds and budget on the patched level.

Fixer reward per attempt:

| Term | Value | Why |
|---|---|---|
| target report gone in all k seeds | **+1.0** | the fix worked (as far as QA can tell) |
| level broken: exit no longer reached, or coverage drops > 20% | **−1.0** | stops "fixes" that wall off the problem area |
| each new report cluster not present before | **−0.5** | stops fixes that create bugs |
| each edit | **−0.05** | prefers the smallest fix |
| build fails / invalid level | **−0.2** | |

GRPO: a group = 4 attempts on the same report and level, advantage normalised within the
group, one LoRA update per group (as in `qa_train.phase2_train`).

**Known blind spot:** if QA cannot see a bug, nobody can reward its fix. QA finds 3 of 4
MAP07 bugs with Go-Explore and misses one-way lifts (it restarts every 500 steps, so it
rarely confirms "stuck forever"). The `CONTRIBUTING.md` task "Go-Explore patience" fixes
this and should come first or alongside.

## 5. The Planter (stage 2) and the arms race

The Planter picks the bugs for each new level from the generator's menu (bug kind ×
target × parameters). Reward per planted bug:

| Outcome | Reward | Why |
|---|---|---|
| QA detects it **and** the Fixer fails to repair it | **+1.0** | a fair bug the Fixer can't fix yet: exactly what it should train on |
| QA detects it, the Fixer repairs it | 0.0 | too easy |
| QA never detects it | **−0.2** | invisible bugs teach nothing and would let it cheat |
| plant fails (invalid spec) | −0.2 | |

So the Planter is pushed toward bugs that are **detectable but hard to repair**, and the
Fixer's training data gets harder as it improves. That is the endless co-evolution.

## 6. How we measure it (answer key, evaluation only)

On held-out procgen **test** seeds the Fixer never trained on (D31 split):

- **Repair rate:** planted bugs no longer matched by any post-fix report (`scoring.matches`)
  while the exit is still reachable.
- **True repair rate:** the edited fields equal the fixed build's values for the bug's
  affected entities (functionally equivalent fixes also count via the first metric).
- **Damage rate:** fixes that broke the level or added new bugs.
- **Edits per repair.**
- Baselines: random menu choice; **nearest-first** (the nearest entity's first edit, no
  learning); "oracle-menu" upper bound (the correct edit is in the menu: measures the
  menu, not the Fixer); untrained Qwen; GRPO-trained Qwen.
  F1 measured (114 bugs, 50 generated maps): menus have a median of 5 entities / 12
  edits, and for 5 of the 7 kinds the right entity is the nearest one, so nearest-first
  should already repair those. The Fixer has to beat nearest-first to show any value; the
  real test is one_way_lift (3 edits, right entity ~4th) and teleport_loop (2 edits,
  right entity ~9th), and multi-bug levels where the nearest entity belongs to another bug.

## 7. Code layout (new package `qatester/fixer/`)

| File | Role |
|---|---|
| `edits.py` | the 6 edit ops on `udmf.TextMap` + apply + rebuild WAD |
| `menu.py` | candidate edits around a report, from the level only |
| `prompt.py` | report + nearby entities → text |
| `verify.py` | baseline / re-test protocol, Fixer reward |
| `llm.py` | `TwoPlayerLLM` from game-qa-coevolution, roles `fixer` (+ `planter`) |
| `loop.py` | inner loop (reports of one level) and outer loop (levels forever), checkpoints |
| `scripts/fix_loop.py` | entry point; `configs/fix_*.yaml` |

## 8. Milestones (each with an acceptance test before the next starts)

| | What | Acceptance test |
|---|---|---|
| F0 | `edits.py` | for each of the 7 bug kinds, a hand-written patch made only of the 6 ops makes the bug disappear in the real engine and keeps the exit reachable (proves the op set **can** fix every kind) |
| F1 | `menu.py`, `prompt.py` | on 50 generated maps, the correct edit is in the menu for ≥ 95% of planted bugs; leak test passes |
| F2 | `verify.py` | per bug kind, measured reproduction rate at the baseline budget; reward-hacking tests: walling off the area or deleting access gives a negative reward |
| F3 | baselines | random-menu and oracle-menu repair rates on test seeds |
| F4 | untrained Qwen Fixer | zero-shot repair rate (no training) |
| F5 | GRPO Fixer, endless outer loop on train seeds | repair rate on test seeds above the untrained Fixer (one-sided Mann-Whitney over seeds) |
| F6 | Planter co-evolution | Fixer trained with the Planter beats one trained on random bugs, on the same test seeds |

## 9. Compute

- Re-testing dominates: QA runs at ~300-700 steps/s per CPU worker (PLAN.md), so one
  attempt (3 seeds × 20k steps) is ~1.5-3.5 min on one worker; a GRPO group of 4 attempts
  is ~2-4 min on 4 workers. F2 measures the smallest budget that still reproduces each
  bug kind, which sets the real cost.
- The LLM choices are cheap (2-6 decisions per attempt). Training (F5, F6) runs on Kaggle
  GPU with the CPU workers in the same session; F0-F4 run locally on CPU (Qwen 3B or
  smaller for F4 if no GPU).

## 10. Open questions for sign-off

1. Fixer model size: Qwen2.5-**3B** (faster, the co-evolution repo's main size) or 7B?
2. Start with **Go-Explore patience** (so one-way lifts and traps become visible to QA)
   before F5? Recommended.
3. Where the code lives: this repo (Satyajit-147/VizDoom) on a branch, with `llm.py`
   copied from game-qa-coevolution, or a new repo?
