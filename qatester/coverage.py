"""The four coverage measures, with visit counts shared by every worker (PLAN: Coverage
accounting).

Counts live in one `multiprocessing.shared_memory` block that the creating process owns.
A tracker pickles to the block's name, so a copy sent to a subprocess (SB3's
SubprocVecEnv) attaches to the same counts. Increments are not atomic across processes:
two workers bumping one cell in the same instant lose one count. That only makes the
novelty bonus very slightly larger and is accepted (docs/decisions.md D29).

Abstract states (cell, keys, doors) have no fixed denominator, so they are hashed into a
fixed table; the number of non-zero buckets is the distinct-state count, exact until
collisions start (well past a million distinct states for the default table).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np

from qatester.mapgen.groundtruth import GroundTruth

Cell = tuple[int, int]
AbstractState = tuple[int, int, int, int]  # cell x, cell y, keys bitmask, doors bitmask
DEFAULT_STATE_BUCKETS = 1 << 20


@dataclass(frozen=True)
class _Layout:
    width: int
    height: int
    n_things: int
    n_portals: int
    state_buckets: int

    @property
    def sizes(self) -> tuple[int, int, int, int]:
        return (self.width * self.height, self.state_buckets, self.n_things, self.n_portals)

    @property
    def total(self) -> int:
        return sum(self.sizes)


def _state_bucket(s: AbstractState, buckets: int) -> int:
    h = (s[0] * 73856093) ^ (s[1] * 19349663) ^ (s[2] * 83492791) ^ (s[3] * 2654435761)
    return (h & 0x7FFFFFFFFFFFFFFF) % buckets


class CoverageTracker:
    def __init__(self, truth: GroundTruth, *, state_buckets: int = DEFAULT_STATE_BUCKETS,
                 _attach_to: str | None = None) -> None:  # fmt: skip
        self.truth = truth.intent_only() if "defects" in truth.data else truth
        d = self.truth.data
        w = d["walkable"]
        self.layout = _Layout(w["width"], w["height"], len(d["things"]), len(d["portals"]),
                              state_buckets)  # fmt: skip
        nbytes = self.layout.total * np.dtype(np.int32).itemsize
        self.owner = _attach_to is None
        if self.owner:
            self.shm = shared_memory.SharedMemory(create=True, size=nbytes)
        else:
            assert _attach_to is not None
            self.shm = _attach(_attach_to)
        flat = np.ndarray((self.layout.total,), np.int32, buffer=self.shm.buf)
        if self.owner:
            flat[:] = 0
        parts, start = [], 0
        for size in self.layout.sizes:
            parts.append(flat[start:start + size])
            start += size
        self._flat = flat
        self.cells = parts[0].reshape(self.layout.height, self.layout.width)  # [iy, ix]
        self.states, self.things, self.portals = parts[1], parts[2], parts[3]
        self.walkable = np.array([[c == "1" for c in row] for row in w["rows"]], bool)
        self.interactive = np.array([bool(t["interactive"]) for t in d["things"]], bool)
        self.portal_index = {p["id"]: i for i, p in enumerate(d["portals"])}
        self.thing_index = {t["id"]: i for i, t in enumerate(d["things"])}

    # --- pickling: a copy attaches to the same shared block --------------------------
    def __reduce__(self) -> tuple[Any, ...]:
        return (_reattach, (self.truth.data, self.shm.name, self.layout.state_buckets))

    @property
    def name(self) -> str:
        return self.shm.name

    # --- counts ----------------------------------------------------------------------
    def _in_grid(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.layout.width and 0 <= cell[1] < self.layout.height

    def cell_count(self, cell: Cell) -> int:
        return int(self.cells[cell[1], cell[0]]) if self._in_grid(cell) else 0

    def visit_cell(self, cell: Cell) -> int:
        """Count one entry; returns the count before it. Cells off the grid count 0."""
        if not self._in_grid(cell):
            return 0
        n = int(self.cells[cell[1], cell[0]])
        self.cells[cell[1], cell[0]] = n + 1
        return n

    def neighbour_novelty(self, cell: Cell) -> float:
        """Mean 1/sqrt(1 + n) over the walkable cells around `cell` (0 if none)."""
        x, y = cell
        x0, y0 = max(0, x - 1), max(0, y - 1)
        counts = self.cells[y0:y + 2, x0:x + 2]
        mask = self.walkable[y0:y + 2, x0:x + 2].copy()
        if self._in_grid(cell):
            mask[y - y0, x - x0] = False
        if not mask.any():
            return 0.0
        return float((1.0 / np.sqrt(1.0 + counts[mask])).mean())

    def visit_state(self, s: AbstractState) -> int:
        b = _state_bucket(s, self.layout.state_buckets)
        n = int(self.states[b])
        self.states[b] = n + 1
        return n

    def use_thing(self, thing_id: str) -> None:
        self.things[self.thing_index[thing_id]] += 1

    def traverse_portal(self, portal_id: str) -> None:
        self.portals[self.portal_index[portal_id]] += 1

    def summary(self) -> dict[str, float]:
        """The four measures (PLAN: Four coverage measures)."""
        walk = self.walkable
        visited = self.cells > 0
        n_walk = int(walk.sum())
        n_inter = int(self.interactive.sum())
        n_port = self.layout.n_portals
        return {
            "cell_coverage": float((visited & walk).sum() / n_walk) if n_walk else 0.0,
            "cells_visited": int(visited.sum()),
            "entity_coverage": (float(((self.things > 0) & self.interactive).sum() / n_inter)
                                if n_inter else 0.0),
            "connection_coverage": float((self.portals > 0).sum() / n_port) if n_port else 0.0,
            "abstract_states": int(np.count_nonzero(self.states)),
        }  # fmt: skip

    def room_cells(self) -> dict[str, tuple[int, int]]:
        """Per room: (walkable cells visited, walkable cells), by each cell's centre."""
        w = self.truth.data["walkable"]
        ox, oy, size = w["origin"][0], w["origin"][1], w["cell_size"]
        out = {r["id"]: [0, 0] for r in self.truth.data["rooms"]}
        for iy, ix in zip(*np.nonzero(self.walkable), strict=True):
            room = self.truth.room_at(ox + (ix + 0.5) * size, oy + (iy + 0.5) * size)
            if room is not None:
                out[room][1] += 1
                out[room][0] += int(self.cells[iy, ix] > 0)
        return {k: (v[0], v[1]) for k, v in out.items()}

    def save(self, path: Path) -> None:
        """Snapshot every count (for checkpoints: shared memory does not outlive a session)."""
        np.save(path, np.asarray(self._flat))

    def restore(self, path: Path) -> None:
        saved = np.load(path)
        if saved.shape != self._flat.shape:
            raise ValueError(f"{path}: {saved.shape} counts, expected {self._flat.shape}")
        self._flat[:] = saved

    def close(self) -> None:
        """Detach; the owner also frees the block."""
        self.cells = self.states = self.things = self.portals = self._flat = None  # type: ignore[assignment]
        self.shm.close()
        if self.owner:
            self.shm.unlink()


def _attach(name: str) -> shared_memory.SharedMemory:
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name=name, track=False)
    # Before 3.13 attaching also registers the block with the resource tracker. Workers
    # started by multiprocessing share the owner's tracker, where the name is already
    # registered, so this is harmless; unregistering here would break the owner's unlink.
    return shared_memory.SharedMemory(name=name)


def _reattach(data: dict[str, Any], name: str, state_buckets: int) -> CoverageTracker:
    return CoverageTracker(GroundTruth(data), state_buckets=state_buckets, _attach_to=name)
