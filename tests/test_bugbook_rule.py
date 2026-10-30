"""M7 acceptance: the Bug Book's add / confirm / decrement / remove rule (PLAN: The Bug
Book), including removal at frequency zero, the frequency cap and persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from qatester.bugbook import BugBook
from qatester.oracles.base import Finding


def finding(oracle: str = "stuck", kind: str = "wedge", x: float = 100.0,
            y: float = 200.0) -> Finding:  # fmt: skip
    return Finding(oracle, kind, (x, y, 0.0), (10, 50), 0.9, {"episode": 3})


def test_a_new_finding_is_added_with_frequency_one() -> None:
    book = BugBook()
    eid = book.add_or_confirm(finding(), "MAP02", step=1000)
    (e,) = book.entries_for("MAP02")
    assert e.entry_id == eid == "map02-stuck-0001"
    assert (e.frequency, e.first_seen_step, e.last_confirmed_step) == (1, 1000, 1000)
    assert book.entries_for("MAP03") == []


def test_a_matching_finding_confirms_and_the_frequency_is_capped() -> None:
    book = BugBook(freq_cap=20)
    eid = book.add_or_confirm(finding(), "MAP02", step=1)
    # Same family within 96 mu: the same entry.
    assert book.add_or_confirm(finding(x=160.0, y=250.0), "MAP02", step=2) == eid
    e = book.get(eid)
    assert e is not None and e.frequency == 2 and e.last_confirmed_step == 2
    for s in range(3, 40):
        book.add_or_confirm(finding(), "MAP02", step=s)
    assert e.frequency == 20, "frequency is capped at 20"
    assert len(book.entries_for("MAP02")) == 1


def test_what_does_not_match_is_a_new_entry() -> None:
    book = BugBook()
    a = book.add_or_confirm(finding(), "MAP02")
    b = book.add_or_confirm(finding(x=400.0), "MAP02")  # too far
    c = book.add_or_confirm(finding(oracle="out_of_bounds", kind="outside_level"), "MAP02")
    d = book.add_or_confirm(finding(oracle="invariant", kind="health_cap"), "MAP02")
    e = book.add_or_confirm(finding(oracle="invariant", kind="health_cap", x=999.0), "MAP02")
    f = book.add_or_confirm(finding(oracle="invariant", kind="ammo_nonneg"), "MAP02")
    assert len({a, b, c, d, f}) == 5
    assert e == d, "invariants match on the invariant's name, not on position"


def test_decrement_and_removal_at_zero() -> None:
    book = BugBook()
    eid = book.add_or_confirm(finding(), "MAP02")
    book.add_or_confirm(finding(), "MAP02")  # frequency 2
    book.decrement(eid)
    e = book.get(eid)
    assert e is not None and e.frequency == 1
    book.decrement(eid)
    assert book.get(eid) is None, "removed at frequency 0: the defect is mastered"
    assert book.entries_for("MAP02") == []
    assert book.stats()["removed"] == 1
    with pytest.raises(KeyError):
        book.decrement(eid)
    # A later finding at the same place starts a fresh entry.
    assert book.add_or_confirm(finding(), "MAP02") == "map02-stuck-0002"


def test_targets_stats_and_round_trip(tmp_path: Path) -> None:
    book = BugBook()
    book.add_or_confirm(finding(), "MAP07", step=5)
    book.add_or_confirm(finding(), "MAP07", step=9)
    book.add_or_confirm(finding(oracle="invariant", kind="health_cap"), "MAP07")
    assert [t[0] for t in book.targets("MAP07")] == ["map07-stuck-0001"]  # places only
    assert book.stats() == {"entries": 2, "added": 2, "confirmed": 1, "removed": 0,
                            "new_to_confirm": 2.0}  # fmt: skip
    path = tmp_path / "bugbook.json"
    book.save(path)
    again = BugBook.load(path)
    assert again.to_json() == book.to_json()
    assert BugBook.load_or_new(tmp_path / "missing.json").stats()["entries"] == 0
