"""The demo's core (qatester/demo/lab.py): build a level with chosen bugs, play it,
score it against the answer key. The Streamlit UI itself (scripts/qa_lab.py) is
clicked through with Streamlit's AppTest."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from qatester.demo.lab import Session, make_level, preview_svg

REPO = Path(__file__).resolve().parents[1]


def test_generated_level_has_exactly_the_chosen_bugs(tmp_path: Path) -> None:
    kinds = ["missing_blocking", "wrong_tag", "bad_pickup"]
    level = make_level(tmp_path, seed=2_000_123, kinds=kinds)
    assert sorted(d["kind"] for d in level.defects) == sorted(kinds)
    assert level.cfg_path.exists()


def test_play_finish_and_score_a_corpus_map(tmp_path: Path) -> None:
    """MAP03's leak, walked into by random play, is found and scored."""
    level = make_level(tmp_path / "lv", corpus_map="MAP03", maps_dir=REPO / "maps")
    s = Session(level, tester="random", record_dir=tmp_path / "runs", seed=1)
    try:
        s.step(20_000)
        assert s.screen is not None and s.screen.shape == (240, 320, 3)
        assert "polyline" in preview_svg(level, s.trail)
        res = s.finish(tmp_path / "report")
    finally:
        s.close()
    sc = res["score"]
    assert sc["found"]["d_leak_hall_north"], f"leak not found: {res['report']['findings']}"
    assert sc["false_alarms"] == []
    assert (tmp_path / "report" / "report.html").exists()


def test_streamlit_app_clicks_through(tmp_path: Path) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    os.environ["QA_LAB_DIR"] = str(tmp_path)
    at = testing.AppTest.from_file(str(REPO / "scripts" / "qa_lab.py"), default_timeout=300)
    at.run()
    at.sidebar.button[0].click().run()  # build the default generated level
    at.sidebar.slider[0].set_value(500).run()
    next(b for b in at.button if b.label.startswith("Start")).click().run()
    next(b for b in at.button if b.label.startswith("Finish")).click().run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Planted bugs found" in labels and "False alarms" in labels
