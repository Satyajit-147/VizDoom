# Running the demo

The QA Lab app lets anyone pick or generate a Doom level, plant bugs in it, watch an
automatic tester hunt them, and then reveal the answer key: which planted bugs were
found, which were missed, and whether anything was reported that isn't a bug.

## Setup (once, ~5 minutes)

Linux or macOS with Python 3.11+ (Windows works too; see CONTRIBUTING).

```bash
git clone https://github.com/Satyajit-147/VizDoom.git rl-qa-tester
cd rl-qa-tester
python3 -m venv .venv
.venv/bin/pip install -e ".[demo]"      # ViZDoom ships its own engine; no torch needed
```

## Start it

```bash
.venv/bin/streamlit run scripts/qa_lab.py
```

Open **http://localhost:8501**. (If your shell exports a ROS or other `PYTHONPATH`, start
it with `env -u PYTHONPATH .venv/bin/streamlit run scripts/qa_lab.py`.)

## A 5-minute demo script

**1. The idea (30 s).** "We plant known bugs in a level, keep the answer key hidden from
the tester, let it play, and then check its reports against the key."

**2. Go-Explore finds the hard bugs (3 min).**
- Left panel → Map: **Hand-made map → MAP06 — SOFTLOCK** → **Build level**. The header
  says how many bugs are planted (2), never which ones or where.
- Tester: **Go-Explore**. Steps to play: **30,000**. Click **Start / continue** (about 2-3
  minutes).
- What to point at while it plays:
  - the **path on the map** jumps: each jump is Go-Explore reloading a saved spot and
    exploring onward from it;
  - **saved states** climbs: one save per new place reached;
  - detections appear in the table as they happen.
- Click **Finish and reveal the answer key**. Checked before the demo (2 seeds): **both
  bugs found** (a one-way lift and a teleport loop, the hardest kind: confirming "stuck
  forever" needs patience), 0 false alarms, 70-80% of the map explored.
- Equally reliable at 30,000 steps (2 of 2 bugs, 0 false alarms, both seeds): **MAP05**
  (a broken lock and an over-cap health pickup, ~95% explored) and **MAP02** (two trap
  pits, 80-100% explored).
- Don't use MAP07 live: Go-Explore needs about 150,000 steps to get past its first door.
  Its 92% figure comes from 1M-step runs: show `docs/results/map07_coverage.png`
  instead.

**3. A random level nobody has seen (2 min).**
- Map: **Surprise me** → **Build level**: a brand-new generated level with 2-3 random
  bug types, hidden.
- Tester: **Scripted** or **Go-Explore**, 20,000 steps, Start, then Finish.
- Expect 1-2 of 2-3 bugs found and no false alarms. If something is missed, say why:
  it was in a room the tester never reached (exploring is the hard part; judging what it
  sees is reliable).

**4. Close (30 s), with the MAP07 chart on screen.** "On the big 12-room map, in 1M-step
runs, Go-Explore explores 92% and finds 3 to 4 of the 4 bugs on every seed, while the
trained neural agents explore about 60%. On 39 maps none of our testers were tuned on,
even the simple testers find about three quarters of the planted bugs with 98%
precision."

## What the testers are

| Tester | How it plays |
|---|---|
| Go-Explore | saves the game at every new place, reloads saves and explores further; with *patience*, some runs play until time runs out so "stuck forever" bugs can be confirmed |
| Random | presses random buttons |
| Scripted | follows walls, presses USE on doors and switches |
| Trained AI (PPO) | appears only if PyTorch is installed and a checkpoint is under `checkpoints/` |

## Bug types you can plant

| In the app | What it is | Usually found? |
|---|---|---|
| Leaky wall | you can walk out of the level | yes |
| Trap pit | you fall in and can't climb out | yes |
| Broken lock | a locked door opens without its key | yes |
| Teleport loop | two teleporters bounce you back and forth | yes |
| One-way lift | a lift down with no way back up | often |
| Bad pickup | health goes above the maximum | often |
| Dead switch | a switch that opens nothing | hard: the tester must press it, then check the door |

For a dependable result on stage, use the hand-made **MAP06**, **MAP05** or **MAP02** with
Go-Explore and 30,000 steps (each found all its bugs, with no false alarm, in our checks),
or **Generate a new map** and pick *Broken lock + Trap pit*.

## Troubleshooting

- **Blank page / nothing happens:** the first level build takes a few seconds; watch the
  spinner. Reload the page if the app was restarted.
- **"Address not mapped… vizdoom-crash.log" in the terminal:** an old game engine
  crashing *while closing* when the app switches levels. Harmless; the run continues.
- **Port in use:** `streamlit run scripts/qa_lab.py --server.port 8502`.
- Runs write to `/tmp/qa_lab` (set `QA_LAB_DIR` to change it); each finished run has a
  full HTML report there, with the path printed under the answer key.
