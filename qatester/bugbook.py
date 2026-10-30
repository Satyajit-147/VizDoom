"""The Bug Book (PLAN: Coverage accounting and the Bug Book), structured like Code-A1's
Mistake Book, with the same update rule:

- a finding that matches no entry is added with frequency 1;
- a finding that matches an entry raises its frequency by 1, capped at 20;
- replaying an entry that no longer reproduces lowers its frequency by 1;
- at frequency 0 the entry is removed: the defect is considered mastered.

Matching rule (the same as report clustering): same oracle family, positions within
`match_radius_mu`, and for invariants the same invariant name.

Its three roles: a stable target for the Bug Book reward (+1.5 for reaching a known
bug's place, `targets()`); a curriculum signal (new findings vs reconfirmations,
`stats()`); and regression catching (replay entries on a new build, E7). Persisted as
JSON so a run can resume.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qatester.oracles.base import Finding

ABBR = {"crash": "crash", "stuck": "stuck", "out_of_bounds": "oob", "unreachable": "unreach",
        "invariant": "inv", "softlock": "soft"}  # fmt: skip


@dataclass
class BugBookEntry:
    entry_id: str
    oracle: str
    kind: str
    position: list[float]
    frequency: int
    first_seen_step: int
    last_confirmed_step: int
    demo_path: str | None = None
    save_path: str | None = None  # a Go-Explore save at the bug (M8)
    evidence: dict[str, Any] = field(default_factory=dict)

    def matches(self, oracle: str, kind: str, x: float, y: float, radius: float) -> bool:
        if oracle != self.oracle:
            return False
        if oracle == "invariant":
            return kind == self.kind
        return math.hypot(x - self.position[0], y - self.position[1]) <= radius


class BugBook:
    def __init__(self, match_radius_mu: float = 96.0, freq_cap: int = 20) -> None:
        self.radius = match_radius_mu
        self.cap = freq_cap
        self.maps: dict[str, list[BugBookEntry]] = {}
        self.counters: dict[str, int] = {}
        self.added = 0
        self.confirmed = 0
        self.removed = 0

    # --- the update rule --------------------------------------------------------------
    def add_or_confirm(self, f: Finding | Mapping[str, Any], map_name: str,
                       step: int = 0) -> str:  # fmt: skip
        """Add a new entry (frequency 1) or confirm a matching one (frequency + 1, capped).
        Returns the entry id."""
        d = asdict(f) if isinstance(f, Finding) else dict(f)
        x, y = float(d["position"][0]), float(d["position"][1])
        entries = self.maps.setdefault(map_name, [])
        for e in entries:
            if e.matches(d["oracle"], d["kind"], x, y, self.radius):
                e.frequency = min(self.cap, e.frequency + 1)
                e.last_confirmed_step = step
                self.confirmed += 1
                return e.entry_id
        key = f"{map_name}-{ABBR.get(d['oracle'], d['oracle'])}"
        self.counters[key] = self.counters.get(key, 0) + 1
        demo = d.get("demo_path")
        entry = BugBookEntry(
            entry_id=f"{key.lower()}-{self.counters[key]:04d}", oracle=d["oracle"],
            kind=d["kind"], position=[float(v) for v in d["position"]], frequency=1,
            first_seen_step=step, last_confirmed_step=step,
            demo_path=str(demo) if demo else None,
            evidence=json.loads(json.dumps(d.get("evidence", {}), default=str)),
        )  # fmt: skip
        entries.append(entry)
        self.added += 1
        return entry.entry_id

    def decrement(self, entry_id: str) -> None:
        """A replay of this entry did not reproduce it. At 0 it is removed."""
        for entries in self.maps.values():
            for e in entries:
                if e.entry_id == entry_id:
                    e.frequency -= 1
                    if e.frequency <= 0:
                        entries.remove(e)
                        self.removed += 1
                    return
        raise KeyError(entry_id)

    # --- reading ----------------------------------------------------------------------
    def entries_for(self, map_name: str) -> list[BugBookEntry]:
        return list(self.maps.get(map_name, []))

    def get(self, entry_id: str) -> BugBookEntry | None:
        return next((e for es in self.maps.values() for e in es if e.entry_id == entry_id),
                    None)  # fmt: skip

    def targets(self, map_name: str) -> list[tuple[str, float, float]]:
        """(entry id, x, y) of the places the Bug Book reward pays for reaching."""
        return [(e.entry_id, e.position[0], e.position[1]) for e in self.maps.get(map_name, [])
                if e.oracle != "invariant"]  # fmt: skip

    def stats(self) -> dict[str, Any]:
        """The curriculum signal: new findings against reconfirmations."""
        total = sum(len(es) for es in self.maps.values())
        return {"entries": total, "added": self.added, "confirmed": self.confirmed,
                "removed": self.removed,
                "new_to_confirm": round(self.added / self.confirmed, 4) if self.confirmed
                else None}  # fmt: skip

    # --- persistence ------------------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {"match_radius_mu": self.radius, "freq_cap": self.cap,
                "counters": self.counters,
                "totals": {"added": self.added, "confirmed": self.confirmed,
                           "removed": self.removed},
                "maps": {m: [asdict(e) for e in es] for m, es in self.maps.items()}}  # fmt: skip

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=1) + "\n", encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> BugBook:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        book = cls(d["match_radius_mu"], d["freq_cap"])
        book.counters = dict(d["counters"])
        book.added, book.confirmed, book.removed = (d["totals"][k] for k in
                                                    ("added", "confirmed", "removed"))  # fmt: skip
        book.maps = {m: [BugBookEntry(**e) for e in es] for m, es in d["maps"].items()}
        return book

    @classmethod
    def load_or_new(cls, path: Path, match_radius_mu: float = 96.0,
                    freq_cap: int = 20) -> BugBook:  # fmt: skip
        return cls.load(path) if Path(path).exists() else cls(match_radius_mu, freq_cap)
