"""The demo's core (qatester/demo/lab.py): build a level with chosen bugs, play it,
score it against the answer key. The Streamlit UI itself (scripts/qa_lab.py) is
clicked through with Streamlit's AppTest."""

from __future__ import annotations

import sys
from collections.abc import Iterator
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


@pytest.fixture
def keep_main() -> Iterator[None]:
    """AppTest runs the app as `__main__`. Put the real one back afterwards: worker
    processes started by later tests re-import `__main__`, and would start the app."""
    main = sys.modules["__main__"]
    yield
    sys.modules["__main__"] = main


def test_streamlit_app_clicks_through(tmp_path: Path, keep_main: None,
                                      monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("QA_LAB_DIR", str(tmp_path))
    at = testing.AppTest.from_file(str(REPO / "scripts" / "qa_lab.py"), default_timeout=300)
    at.run()
    at.sidebar.button[0].click().run()  # build the default generated level
    at.sidebar.slider[0].set_value(500).run()
    next(b for b in at.button if b.label.startswith("Start")).click().run()
    next(b for b in at.button if b.label.startswith("Finish")).click().run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Planted bugs found" in labels and "False alarms" in labels


def test_goexplore_tester_in_the_lab(tmp_path: Path) -> None:
    """The demo's Go-Explore tester: it jumps between saved states (gaps in the trail),
    keeps the clock running after loads (D46), and finds MAP03's leak."""
    level = make_level(tmp_path / "lv", corpus_map="MAP03", maps_dir=REPO / "maps")
    s = Session(level, tester="goexplore", record_dir=tmp_path / "runs", seed=1)
    try:
        s.step(20_000)
        assert s.saved_states >= 20
        assert sum(x != x for x, _ in s.trail) >= 5  # NaN gaps: jumps to saved states
        res = s.finish(tmp_path / "report")
    finally:
        s.close()
    sc = res["score"]
    assert sc["found"]["d_leak_hall_north"], f"leak not found: {res['report']['findings']}"
    assert sc["false_alarms"] == []
