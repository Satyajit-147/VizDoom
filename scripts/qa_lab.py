"""QA Lab: build a level, plant bugs, and watch a tester hunt for them.

    streamlit run scripts/qa_lab.py

1. Left panel: generate a new map from a seed with the bug types you tick, or pick one
   of the eight hand-made maps; then choose a tester (random, scripted, or a trained
   PPO checkpoint when PyTorch is installed).
2. Run: the game view and the top-down map update live, with the tester's path and a
   marker each time a bug detector fires.
3. Finish: the answer key is revealed — planted bugs found or missed, false alarms,
   precision and recall — and a full HTML bug report is written.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import random
import sys
import time
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from qatester.demo.lab import BUG_LABELS, LabLevel, Session, make_level, preview_svg

OUT = Path(os.environ.get("QA_LAB_DIR", "/tmp/qa_lab"))
CORPUS = {
    "MAP01": "CLEAN: 4 rooms, a key and a locked vault (no bugs: a false-alarm check)",
    "MAP02": "WEDGE: an alcove and a pit",
    "MAP03": "LEAK: a balcony on a lift",
    "MAP04": "ORPHAN: switch-operated doors",
    "MAP05": "PHANTOM: a locked door and a medkit",
    "MAP06": "SOFTLOCK: a pit lift and teleporters",
    "MAP07": "SPRAWL: 12 rooms, the big map",
    "MAP08": "HOLDOUT: never used in training",
}
ORACLE_NAMES = {"crash": "Crash", "stuck": "Stuck", "out_of_bounds": "Out of bounds",
                "unreachable": "Unreachable area", "invariant": "Broken game rule",
                "softlock": "Softlock"}  # fmt: skip
CHECKPOINTS = sorted(str(p.parent) for p in REPO.glob("checkpoints/**/model.zip"))


def torch_available() -> bool:
    try:
        import sb3_contrib  # noqa: F401
    except ImportError:
        return False
    return True


st.set_page_config(page_title="QA Lab", layout="wide")
st.title("QA Lab — an AI tester hunting bugs in Doom levels")
ss = st.session_state

# --- left panel: the level ------------------------------------------------------------
with st.sidebar:
    st.header("1. Level")
    source = st.radio("Map", ["Surprise me", "Generate a new map", "Hand-made map"],
                      horizontal=True)  # fmt: skip
    if source == "Surprise me":
        st.caption("A random new map with 2-3 random bug types planted. Which bugs, and "
                   "where, stays hidden until you reveal the answer key.")
    elif source == "Generate a new map":
        seed = int(st.number_input("Seed (any number makes a different map)", 0, 10**7,
                                   2_000_000, step=1))  # fmt: skip
        picked = st.multiselect("Bugs to plant", list(BUG_LABELS), format_func=BUG_LABELS.get,
                                default=["missing_blocking", "geometry_trap"])  # fmt: skip
    else:
        lump = st.selectbox("Map", list(CORPUS), format_func=lambda m: f"{m} — {CORPUS[m]}")
    if st.button("Build level", type="primary"):
        if "session" in ss:
            ss.session.close()
            del ss["session"]
        with st.spinner("Building the level..."):
            if source == "Surprise me":
                dice = random.Random()
                ss.level = make_level(OUT / "levels", seed=dice.randrange(10**7),
                                      kinds=dice.sample(list(BUG_LABELS), dice.randint(2, 3)))  # fmt: skip
            elif source == "Generate a new map":
                ss.level = make_level(OUT / "levels", seed=seed, kinds=picked)
            else:
                ss.level = make_level(OUT / "levels", corpus_map=lump, maps_dir=REPO / "maps")
        ss.result = None

    st.header("2. Tester")
    options = ["goexplore", "random", "scripted"] + (
        ["ppo"] if torch_available() and CHECKPOINTS else [])
    tester = st.radio("Who tests it", options, format_func={
        "goexplore": "Go-Explore (saves the game at every new place, jumps back to explore further)",
        "random": "Random (presses random buttons)",
        "scripted": "Scripted (follows walls, presses USE)",
        "ppo": "Trained AI tester (PPO)"}.get)  # fmt: skip
    ckpt = st.selectbox("Trained checkpoint", CHECKPOINTS) if tester == "ppo" else None
    if not torch_available():
        st.caption("The trained tester needs PyTorch installed on this machine.")
    budget = int(st.slider("Steps to play", 1_000, 60_000, 20_000, step=1_000))

level: LabLevel | None = ss.get("level")
if level is None:
    st.info("Build a level in the left panel to begin.")
    st.stop()

st.subheader(f"Level: {level.name} — {len(level.spec.rooms)} rooms, "
             f"{len(level.defects)} planted bug(s), hidden from the tester")  # fmt: skip
c1, c2, c3 = st.columns([1, 1, 1])
start = c1.button("Start / continue", type="primary")
finish = c2.button("Finish and reveal the answer key")
reset = c3.button("New run on this level")
if reset and "session" in ss:
    ss.session.close()
    del ss["session"]
    ss.result = None

if (start or "session" in ss) and "session" not in ss:
    ss.session = Session(level, tester=tester, checkpoint=Path(ckpt) if ckpt else None,
                         record_dir=OUT / "runs" / f"{level.name}_{tester}")  # fmt: skip
session: Session | None = ss.get("session")

left, right = st.columns([1, 1])
game = left.empty()
stats = left.empty()
mapview = right.empty()
st.markdown("#### Bugs detected so far")
table = st.empty()


def draw() -> None:
    assert session is not None
    if session.screen is not None:
        game.image(session.screen, caption="What the tester sees (colour view)",
                   width="stretch")  # fmt: skip
    cov = session.coverage_summary()
    saved = (f" · saved states **{session.saved_states:,}**"
             if session.tester == "goexplore" else "")
    stats.markdown(f"**Step {session.steps:,}** · episodes {session.episodes} · "
                   f"area explored **{100 * cov['cell_coverage']:.0f}%** · "
                   f"detections {len(session.findings)}{saved}")  # fmt: skip
    marks = [{**f, "label": ORACLE_NAMES[f["oracle"]]} for f in session.findings[-40:]]
    mapview.markdown(preview_svg(level, session.trail[-3000:], marks), unsafe_allow_html=True)
    table.dataframe([{"step": f["env_steps"], "detector": ORACLE_NAMES[f["oracle"]],
                      "what": f["kind"].replace("_", " "),
                      "where": f"({f['position'][0]:.0f}, {f['position'][1]:.0f})"}
                     for f in reversed(session.findings)], width="stretch")  # fmt: skip


if session is not None:
    draw()
    if start and not finish:
        target = session.steps + budget
        while session.steps < target:
            session.step(25)  # redraw every 25 steps: the browser is the slow part
            draw()
            time.sleep(0.01)

if finish and session is not None:
    with st.spinner("Judging the run and writing the report..."):
        ss.result = session.finish(OUT / "reports" / f"{level.name}_{session.tester}")

result = ss.get("result")
if result is not None:
    st.markdown("## The answer key")
    sc, rep = result["score"], result["report"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Planted bugs found", f"{len(level.defects) - len(sc['missed'])} / {len(level.defects)}")
    m2.metric("False alarms", len(sc["false_alarms"]))
    m3.metric("Precision", "—" if sc["precision"] is None else f"{100 * sc['precision']:.0f}%")
    m4.metric("Recall", "—" if sc["recall"] is None else f"{100 * sc['recall']:.0f}%")
    st.dataframe([{"planted bug": BUG_LABELS.get(d["kind"], d["kind"]),
                   "where": ", ".join(d["affected_rooms"]),
                   "found?": "FOUND" if sc["found"][d["id"]] else "missed",
                   "reported as": ", ".join(sc["found"][d["id"]])}
                  for d in level.defects], width="stretch")  # fmt: skip
    st.markdown("#### The map with the planted bugs revealed (red)")
    st.markdown(preview_svg(level, session.trail[-3000:] if session else [],
                            [{**f, "label": f["id"]} for f in rep["findings"]],
                            show_bugs=True), unsafe_allow_html=True)  # fmt: skip
    path = OUT / "reports" / f"{level.name}_{session.tester if session else ''}" / "report.html"
    st.caption(f"Full bug report with replays: {path}")
