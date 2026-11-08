"""The Go-Explore cell archive (PLAN: The cell archive; M8).

One entry per discovered cell (ix, iy, iz bucket, keys held, doors open), with a ViZDoom
save file
for the best-known way to arrive there: the earliest arrival, which leaves the most
episode time to explore from it. Reset points are drawn by one of four heuristics:

- count: weight 1 / sqrt(1 + visits);
- frontier: weight 1 / (1 + d), d the Chebyshev distance in grid cells to the nearest
  cell nobody has reached yet. The plan measures d to the nearest unvisited *walkable*
  cell, but walkability comes from the ground truth, which an agent may not read
  (tests/test_no_groundtruth_leak.py). Here d is measured against the archive itself,
  so a cell against a wall looks like a frontier until the productivity guard learns
  otherwise, which is the case the guard exists for (docs/decisions.md D46);
- mixed: the product of the two (the default);
- uniform.

The ReMiDi-style productivity guard multiplies every weight by
(1 + new_cells_after_reset) / (1 + resets_from). When the archive is full, the least
recently used entry is evicted with its save file.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

Heuristic = Literal["count", "frontier", "mixed", "uniform"]
Cell = tuple[int, int, int, tuple[str, ...], int]

CELL_MU = 64.0  # the coverage grid's cell size
Z_BUCKET_MU = 64.0  # so a balcony and the floor under it are different cells
FRONTIER_RADIUS = 4  # cells; farther than this from anything unreached counts as d = 5


def cell_of(x: float, y: float, z: float, keys: Iterable[str], doors: int = 0) -> Cell:
    """The plan's cell plus the open-door mask (D46), the env's abstract state: the
    moment a door opens is worth saving, so later resets start with it open."""
    return (math.floor(x / CELL_MU), math.floor(y / CELL_MU), math.floor(z / Z_BUCKET_MU),
            tuple(sorted(keys)), doors)  # fmt: skip


@dataclass
class Entry:
    cell: Cell
    save_path: Path
    depth: int  # episode tic of the saved arrival
    found_step: int  # explorer step at which the cell was first reached
    visits: int = 0
    resets_from: int = 0
    new_cells_after_reset: int = 0
    last_improved_step: int = 0
    # Farthest any run restarted from this cell got from it (Chebyshev, in grid cells).
    # A cell whose restarts never get far looks like a trap: patience starts there.
    max_reach: int = 0

    @property
    def best_return(self) -> float:
        """Replacement score: an earlier arrival is better."""
        return -float(self.depth)


class Archive:
    def __init__(self, save_dir: Path, max_cells: int = 20_000,
                 heuristic: Heuristic = "mixed", guard: bool = True) -> None:  # fmt: skip
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_cells = max_cells
        self.heuristic = heuristic
        self.guard = guard
        self.entries: OrderedDict[Cell, Entry] = OrderedDict()  # least recently used first
        self.reached_xy: set[tuple[int, int]] = set()
        self.evicted = 0
        self._next_file = 0

    def __len__(self) -> int:
        return len(self.entries)

    def visit(self, cell: Cell, depth: int, step: int,
              save: Callable[[Path], None]) -> bool:  # fmt: skip
        """The explorer is in `cell` at episode tic `depth`. Saves the state (via `save`)
        if the cell is new or reached earlier than before; True if it is new."""
        e = self.entries.get(cell)
        if e is not None:
            e.visits += 1
            self.entries.move_to_end(cell)
            if depth < e.depth:
                save(e.save_path)
                e.depth, e.last_improved_step = depth, step
            return False
        path = self.save_dir / f"c{self._next_file}.sav"
        self._next_file += 1
        save(path)
        self.entries[cell] = Entry(cell, path, depth, step, visits=1, last_improved_step=step)
        self.reached_xy.add((cell[0], cell[1]))
        while len(self.entries) > self.max_cells:
            _, old = self.entries.popitem(last=False)
            old.save_path.unlink(missing_ok=True)
            self.evicted += 1
        return True

    def _frontier_d(self, cell: Cell) -> int:
        ix, iy = cell[0], cell[1]
        for r in range(1, FRONTIER_RADIUS + 1):
            for dx in range(-r, r + 1):
                for dy in (-r, r) if abs(dx) != r else range(-r, r + 1):
                    if (ix + dx, iy + dy) not in self.reached_xy:
                        return r
        return FRONTIER_RADIUS + 1

    def weights(self, heuristic: Heuristic | None = None) -> np.ndarray:
        h = heuristic or self.heuristic
        out = []
        for e in self.entries.values():
            w = 1.0
            if h in ("count", "mixed"):
                w *= 1.0 / math.sqrt(1.0 + e.visits)
            if h in ("frontier", "mixed"):
                w *= 1.0 / (1.0 + self._frontier_d(e.cell))
            if self.guard:
                w *= (1.0 + e.new_cells_after_reset) / (1.0 + e.resets_from)
            out.append(w)
        return np.asarray(out, dtype=float)

    def sample(self, rng: np.random.Generator) -> Entry:
        """A reset point; counts the reset against it."""
        if not self.entries:
            raise ValueError("the archive is empty")
        w = self.weights()
        e = list(self.entries.values())[int(rng.choice(len(w), p=w / w.sum()))]
        e.resets_from += 1
        self.entries.move_to_end(e.cell)
        return e

    def discard(self, cell: Cell) -> None:
        """Drop an entry and its save file (a save that loads into an already finished
        episode, e.g. one made in the last tics before the time limit)."""
        e = self.entries.pop(cell, None)
        if e is not None:
            e.save_path.unlink(missing_ok=True)

    def sample_confined(self, rng: np.random.Generator, min_resets: int = 2) -> Entry | None:
        """A start for a patient run: among cells restarted from at least `min_resets`
        times, one whose restarts never got far, weight 1 / (1 + max_reach)^2. Uses only
        the archive's own trajectories, never the level. None if no cell qualifies yet.
        Counts the reset against it."""
        cands = [e for e in self.entries.values() if e.resets_from >= min_resets]
        if not cands:
            return None
        w = np.asarray([1.0 / (1.0 + e.max_reach) ** 2 for e in cands], dtype=float)
        e = cands[int(rng.choice(len(cands), p=w / w.sum()))]
        e.resets_from += 1
        self.entries.move_to_end(e.cell)
        return e

    def summary(self) -> dict[str, float | int]:
        es = list(self.entries.values())
        return {"cells": len(es), "xy_cells": len(self.reached_xy), "evicted": self.evicted,
                "resets": sum(e.resets_from for e in es),
                "productive_resets": sum(e.new_cells_after_reset > 0 for e in es)}  # fmt: skip
