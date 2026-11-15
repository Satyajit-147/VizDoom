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

**2. The big map: what Go-Explore does (2 min).**
- Left panel → Map: **Hand-made map → MAP07 — SPRAWL** → **Build level**. The header
  says how many bugs are planted (4), never which ones or where.
- Tester: **Go-Explore**. Steps to play: **40,000**. Click **Start / continue**.
- What to point at while it plays:
  - the **path on the map** jumps: each jump is Go-Explore reloading a saved spot and
    exploring onward from it;
  - **saved states** climbs: one save per new place reached;
  - **area explored** climbs past what the other testers manage (they plateau around
    40-65% on this map; Go-Explore reaches about 90%);
  - detections appear in the table as they happen.
- Click **Finish and reveal the answer key**: found / missed, false alarms (expect 0),
  precision and recall, and the map with the planted bugs drawn in red.

**3. A random level nobody has seen (2 min).**
- Map: **Surprise me** → **Build level**: a brand-new generated level with 2-3 random
  bug types, hidden.
- Tester: **Scripted** or **Go-Explore**, 20,000 steps, Start, then Finish.
- Expect 1-2 of 2-3 bugs found and no false alarms. If something is missed, say why:
  it was in a room the tester never reached (exploring is the hard part; judging what it
  sees is reliable).

**4. Close (30 s).** "On 39 maps none of our testers were tuned on, the simple testers
find about three quarters of the planted bugs with 98% precision. On the big map,
Go-Explore explores 92% and finds 3-4 of 4 bugs, while the trained neural agents
explore about 60%."

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

For a guaranteed result on stage, use **Generate a new map** and pick *Broken lock +
Trap pit*, or the hand-made **MAP03** (leaky wall) or **MAP02** (trap pits).

## Troubleshooting

- **Blank page / nothing happens:** the first level build takes a few seconds; watch the
  spinner. Reload the page if the app was restarted.
- **"Address not mapped… vizdoom-crash.log" in the terminal:** an old game engine
  crashing *while closing* when the app switches levels. Harmless; the run continues.
- **Port in use:** `streamlit run scripts/qa_lab.py --server.port 8502`.
- Runs write to `/tmp/qa_lab` (set `QA_LAB_DIR` to change it); each finished run has a
  full HTML report there, with the path printed under the answer key.
