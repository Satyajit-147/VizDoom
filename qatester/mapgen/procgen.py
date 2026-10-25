"""Seeded random MapSpecs for training and testing on unseen maps (docs/decisions.md D18).

Not the Phase-2 adversarial generator: there is no feedback loop. A seed fixes a map
completely, and the train and test splits draw from disjoint seed ranges.

How a map is made (`generate`):
1. Pick 0-3 defect kinds, distinct, uniformly from the 7 classes. The layout is then
   made to contain what each kind needs (a pit for one_way_lift, a teleport pair for
   teleport_loop, and so on), so every class is equally common.
2. Grow a connected set of grid cells; each cell is one rectangular room, and rooms in
   neighbouring cells share a full wall. Column widths and row heights are random.
3. A random spanning tree from the spawn room gives the portals. Some leaves become pits
   (floor -64) or balconies (+64), reached only by a lift. A few extra edges add loops.
   Portal kinds are drawn per edge: open, door, locked door (the key is placed where it
   can be reached without that door) or switch door.
4. An optional teleport pair joins two rooms; the exit switch goes in the room farthest
   from spawn; pickups are scattered.
5. Defect targets are drawn among valid ones, kept away from what the map needs to be
   played (spawn, exit, keys, switches) and from what another defect cuts off, so each
   defect stays findable and shows up as its own class.
6. The spec is validated and compiled, with and without defects. On any failure the
   generator retries with the next attempt seed, so `generate(seed)` is deterministic.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from qatester.mapgen.cli import compile_checked
from qatester.mapgen.geometry import GeometryError
from qatester.mapgen.groundtruth import ground_truth
from qatester.mapgen.spec import KEY_TYPES, DefectKind, MapSpec, OracleFamily, SpecError

Cell = tuple[int, int]
DIRS: dict[str, Cell] = {"east": (1, 0), "west": (-1, 0), "north": (0, 1), "south": (0, -1)}
DEFECT_KINDS: tuple[DefectKind, ...] = (
    "missing_blocking", "wrong_tag", "wrong_special", "geometry_trap", "one_way_lift",
    "teleport_loop", "bad_pickup",
)  # fmt: skip
EXPECTED_ORACLE: dict[str, OracleFamily] = {
    "missing_blocking": "out_of_bounds",
    "wrong_tag": "unreachable",
    "wrong_special": "invariant",
    "geometry_trap": "stuck",
    "one_way_lift": "softlock",
    "teleport_loop": "softlock",
    "bad_pickup": "invariant",
}
SEVERITY = {"missing_blocking": "high", "wrong_tag": "high", "wrong_special": "high",
            "geometry_trap": "medium", "one_way_lift": "high", "teleport_loop": "medium",
            "bad_pickup": "low"}  # fmt: skip
SPLIT_BASE = {"train": 0, "test": 1_000_000}  # disjoint seed ranges
MAX_ATTEMPTS = 50


class ProcgenError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcgenConfig:
    grid: int = 4  # the layout lives on a grid x grid board
    min_rooms: int = 5
    max_rooms: int = 9
    sizes: tuple[int, ...] = (320, 384, 448, 512)  # column widths / row heights, mu
    max_defects: int = 3
    p_off_level: float = 0.25  # a non-spawn leaf becomes a pit or balcony
    p_loop: float = 0.15  # an extra edge between neighbouring floor-0 rooms (max 2)
    p_teleport: float = 0.3
    kind_weights: dict[str, float] = field(default_factory=lambda: {
        "open": 0.5, "door": 0.3, "locked_door": 0.1, "switch_door": 0.1})  # fmt: skip


def seeds(split: str, count: int, start: int = 0) -> list[int]:
    """Seeds of a split. Train and test never share a seed."""
    if split not in SPLIT_BASE:
        raise ValueError(f"split must be one of {sorted(SPLIT_BASE)}")
    if start + count > SPLIT_BASE["test"]:
        raise ValueError("train seeds would run into the test range")
    return [SPLIT_BASE[split] + start + i for i in range(count)]


# --- layout ----------------------------------------------------------------------------


@dataclass
class _Edge:
    a: str
    b: str
    kind: str  # open / door / locked_door / switch_door / lift
    tree: bool
    requires: str | None = None  # locked_door: key type

    @property
    def id(self) -> str:
        return f"p_{self.a}_{self.b}"


@dataclass
class _Layout:
    rng: np.random.Generator
    cfg: ProcgenConfig
    cells: dict[str, Cell] = field(default_factory=dict)
    xs: list[int] = field(default_factory=list)
    ys: list[int] = field(default_factory=list)
    floor: dict[str, int] = field(default_factory=dict)
    ceiling: dict[str, int] = field(default_factory=dict)
    parent: dict[str, str | None] = field(default_factory=dict)
    edges: list[_Edge] = field(default_factory=list)
    teleports: list[tuple[str, str, str, tuple[int, int]]] = field(default_factory=list)
    things: list[dict[str, Any]] = field(default_factory=list)
    reserved_sides: dict[str, set[str]] = field(default_factory=dict)
    marks: dict[tuple[str, str], list[int]] = field(default_factory=dict)  # wall switches
    spawn: str = ""
    goal: str = ""
    exit_room: str = ""  # the deepest leaf, kept at floor 0 for the exit
    spare: set[str] = field(default_factory=set)  # behind the wrong_tag door: nothing needed
    trap_room: str | None = None  # kept as a plain dead end for geometry_trap

    # geometry helpers
    def rect(self, r: str) -> tuple[int, int, int, int]:
        i, j = self.cells[r]
        return self.xs[i], self.ys[j], self.xs[i + 1], self.ys[j + 1]

    def centre(self, r: str) -> tuple[int, int]:
        x0, y0, x1, y1 = self.rect(r)
        return (x0 + x1) // 2, (y0 + y1) // 2

    def at_cell(self, c: Cell) -> str | None:
        return next((r for r, rc in self.cells.items() if rc == c), None)

    def side_towards(self, a: str, b: str) -> str:
        (ax, ay), (bx, by) = self.cells[a], self.cells[b]
        return next(s for s, d in DIRS.items() if d == (bx - ax, by - ay))

    def portal_sides(self, r: str) -> set[str]:
        out = set()
        for e in self.edges:
            if r in (e.a, e.b):
                out.add(self.side_towards(r, e.b if e.a == r else e.a))
        return out

    def outer_sides(self, r: str) -> list[str]:
        """Sides with no room beyond them."""
        x, y = self.cells[r]
        return [s for s, (dx, dy) in DIRS.items() if self.at_cell((x + dx, y + dy)) is None]

    def degree(self, r: str) -> int:
        return sum(r in (e.a, e.b) for e in self.edges) + sum(
            r in (a, b) for _, a, b, _ in self.teleports
        )


def _grow(lay: _Layout, n: int) -> None:
    rng, g = lay.rng, lay.cfg.grid
    start = (int(rng.integers(g)), int(rng.integers(g)))
    chosen = [start]
    while len(chosen) < n:
        frontier = sorted({(x + dx, y + dy) for x, y in chosen for dx, dy in DIRS.values()
                           if 0 <= x + dx < g and 0 <= y + dy < g} - set(chosen))  # fmt: skip
        chosen.append(frontier[int(rng.integers(len(frontier)))])
    sizes = lay.cfg.sizes
    lay.xs = [0, *np.cumsum([int(rng.choice(sizes)) for _ in range(g)]).tolist()]
    lay.ys = [0, *np.cumsum([int(rng.choice(sizes)) for _ in range(g)]).tolist()]
    lay.cells = {f"r{x}{y}": (x, y) for x, y in chosen}
    lay.spawn = f"r{start[0]}{start[1]}"


def _tree(lay: _Layout) -> None:
    """Random spanning tree by randomised BFS from spawn."""
    rng = lay.rng
    lay.parent = {lay.spawn: None}
    todo = [lay.spawn]
    while todo:
        r = todo.pop(int(rng.integers(len(todo))))
        x, y = lay.cells[r]
        nbrs = [lay.at_cell((x + dx, y + dy)) for dx, dy in DIRS.values()]
        for nb in [n for n in nbrs if n is not None and n not in lay.parent]:
            if nb not in lay.parent:
                lay.parent[nb] = r
                todo.append(nb)


def _children(lay: _Layout, r: str) -> list[str]:
    return [c for c, p in lay.parent.items() if p == r]


def _subtree(lay: _Layout, r: str) -> set[str]:
    out, todo = {r}, [r]
    while todo:
        for c in _children(lay, todo.pop()):
            out.add(c)
            todo.append(c)
    return out


def _depths(lay: _Layout) -> dict[str, int]:
    depth = {lay.spawn: 0}
    order = deque([lay.spawn])
    while order:
        r = order.popleft()
        for c in _children(lay, r):
            depth[c] = depth[r] + 1
            order.append(c)
    return depth


def _levels(lay: _Layout, kinds: set[str]) -> None:
    rng, cfg = lay.rng, lay.cfg
    for r in lay.cells:
        lay.floor[r], lay.ceiling[r] = 0, int(rng.choice([128, 128, 160]))
    leaves = [r for r in lay.cells if r != lay.spawn and not _children(lay, r)]
    depth = _depths(lay)
    deepest = max(leaves, key=lambda r: (depth[r], r)) if leaves else None
    # The deepest leaf stays at floor 0 for the exit; so does one dead end reserved for
    # geometry_trap. The other leaves may change level.
    candidates = [r for r in leaves if r != deepest]
    lay.exit_room = deepest or lay.spawn
    if "geometry_trap" in kinds:
        if not candidates:
            raise ProcgenError("no spare dead end for a trap")
        lay.trap_room = candidates.pop(int(rng.integers(len(candidates))))
    pits: list[str] = []
    for r in candidates:
        if rng.random() >= cfg.p_off_level:
            continue
        if rng.random() < 0.6:
            pits.append(r)
        else:
            lay.floor[r], lay.ceiling[r] = 64, 192  # balcony
    if "one_way_lift" in kinds and not pits:
        if not candidates:
            raise ProcgenError("no leaf for a pit")
        pits.append(candidates[int(rng.integers(len(candidates)))])
    for r in pits:
        lay.floor[r], lay.ceiling[r] = -64, 64


def _edges(lay: _Layout, kinds: set[str]) -> None:
    rng, cfg = lay.rng, lay.cfg
    names = list(cfg.kind_weights)
    weights = np.array([cfg.kind_weights[k] for k in names], float)
    weights /= weights.sum()
    for r, p in lay.parent.items():
        if p is None:
            continue
        if lay.floor[r] != lay.floor[p]:
            kind = "lift"
        else:
            kind = str(rng.choice(names, p=weights))
        lay.edges.append(_Edge(p, r, kind, tree=True))
    flat = [e for e in lay.edges if e.kind != "lift"]
    if "wrong_tag" in kinds:
        # A switch door that cuts off a part of the map nothing else needs.
        free = [e for e in flat if e.kind in ("open", "door") and e.b != lay.trap_room
                and lay.exit_room not in _subtree(lay, e.b)]  # fmt: skip
        if not free:
            raise ProcgenError("no edge for a switch door that cuts off a spare part")
        pick = free[int(rng.integers(len(free)))]
        pick.kind = "switch_door"
        lay.spare = _subtree(lay, pick.b)
    if "wrong_special" in kinds and not any(e.kind == "locked_door" for e in flat):
        free = [e for e in flat if e.kind in ("open", "door") and e.b not in lay.spare]
        if not free:
            raise ProcgenError("no edge to make a locked door")
        free[int(rng.integers(len(free)))].kind = "locked_door"
    # The trap room is entered on foot: no remote door in front of it.
    for e in flat:
        if e.b == lay.trap_room and e.kind == "switch_door":
            e.kind = "door"
    # At most three locked doors (one per key colour).
    for e in [e for e in flat if e.kind == "locked_door"][len(KEY_TYPES):]:
        e.kind = "door"
    # Loops between neighbouring floor-0 rooms not already joined.
    joined = {frozenset((e.a, e.b)) for e in lay.edges}
    loops = 0
    for r in sorted(lay.cells):
        x, y = lay.cells[r]
        for dx, dy in ((1, 0), (0, 1)):
            nb = lay.at_cell((x + dx, y + dy))
            if (nb is None or frozenset((r, nb)) in joined or loops >= 2
                    or lay.floor[r] != 0 or lay.floor[nb] != 0
                    or (r in lay.spare) != (nb in lay.spare)
                    or lay.trap_room in (r, nb)):  # fmt: skip
                continue
            if rng.random() < cfg.p_loop:
                lay.edges.append(_Edge(r, nb, str(rng.choice(["open", "door"])), tree=False))
                joined.add(frozenset((r, nb)))
                loops += 1


def _pad_spot(lay: _Layout, r: str) -> tuple[int, int]:
    """A teleport pad centre 112 mu in from a corner: clear of lift platforms (72 mu
    deep), and in a room of at least 384 mu, at least 80 mu off the centre where
    arrivals land."""
    x0, y0, x1, y1 = lay.rect(r)
    corners = [(x0 + 112, y0 + 112), (x1 - 112, y0 + 112), (x0 + 112, y1 - 112),
               (x1 - 112, y1 - 112)]  # fmt: skip
    return corners[int(lay.rng.integers(4))]


def _big(lay: _Layout, r: str) -> bool:
    x0, y0, x1, y1 = lay.rect(r)
    return x1 - x0 >= 384 and y1 - y0 >= 384


def _teleports(lay: _Layout, kinds: set[str]) -> None:
    if "teleport_loop" not in kinds and lay.rng.random() >= lay.cfg.p_teleport:
        return
    flat = sorted(r for r in lay.cells if lay.floor[r] == 0 and _big(lay, r)
                  and r not in lay.spare and r != lay.trap_room)  # fmt: skip
    pairs = [(a, b) for i, a in enumerate(flat) for b in flat[i + 1:]]
    if not pairs:
        if "teleport_loop" in kinds:
            raise ProcgenError("no pair of rooms for a teleport")
        return
    a, b = pairs[int(lay.rng.integers(len(pairs)))]
    lay.teleports.append((f"tp_{a}_{b}", a, b, _pad_spot(lay, a)))
    lay.teleports.append((f"tp_{b}_{a}", b, a, _pad_spot(lay, b)))


# --- things ----------------------------------------------------------------------------


def _free_point(lay: _Layout, r: str, taken: list[tuple[int, int]]) -> tuple[int, int]:
    x0, y0, x1, y1 = lay.rect(r)
    pads = [at for _, src, _, at in lay.teleports if src == r]
    for _ in range(200):
        x = int(lay.rng.integers(x0 + 96, x1 - 96 + 1))
        y = int(lay.rng.integers(y0 + 96, y1 - 96 + 1))
        if all(abs(x - px) > 80 or abs(y - py) > 80 for px, py in pads + taken):
            return x, y
    raise ProcgenError(f"no free spot in {r}")


def _wall_point(lay: _Layout, r: str) -> tuple[int, int]:
    """A point 40 mu inside a wall of `r` with no portal, preferring one no other mark
    or defect uses. The compiler puts the switch linedef on the nearest wall run."""
    used = lay.reserved_sides.setdefault(r, set())
    sides = [s for s in DIRS if s not in lay.portal_sides(r)]
    if not sides:
        raise ProcgenError(f"{r} has a portal on every side")
    fresh = [s for s in sides if s not in used]
    order = ([fresh[int(lay.rng.integers(len(fresh)))]] if fresh else []) + sorted(used & set(sides))
    x0, y0, x1, y1 = lay.rect(r)
    for side in order:
        lo, hi = (x0, x1) if side in ("north", "south") else (y0, y1)
        mid = (lo + hi) // 2
        taken = lay.marks.setdefault((r, side), [])
        # Along the wall: the middle, else 128 mu either side, 96 mu clear of corners.
        for along in (mid, mid - 128, mid + 128):
            if lo + 96 <= along <= hi - 96 and all(abs(along - t) >= 128 for t in taken):
                taken.append(along)
                used.add(side)
                return {"north": (along, y1 - 40), "south": (along, y0 + 40),
                        "east": (x1 - 40, along), "west": (x0 + 40, along)}[side]  # fmt: skip
    raise ProcgenError(f"no free wall spot for a switch in {r}")


def _reachable_without(lay: _Layout, cut: Iterable[str], start: str) -> set[str]:
    """Rooms reachable from `start` through edges and teleports, never using `cut`."""
    skip = set(cut)
    seen, todo = {start}, [start]
    while todo:
        r = todo.pop()
        nbrs = [e.b if e.a == r else e.a for e in lay.edges
                if e.id not in skip and r in (e.a, e.b)]  # fmt: skip
        nbrs += [b for _, a, b, _ in lay.teleports if a == r]
        for nb in nbrs:
            if nb not in seen:
                seen.add(nb)
                todo.append(nb)
    return seen


def _place_things(lay: _Layout, kinds: set[str]) -> None:
    rng = lay.rng
    depth = _depths(lay)
    lay.goal = lay.exit_room
    taken: dict[str, list[tuple[int, int]]] = {r: [] for r in lay.cells}

    def add(tid: str, ttype: str, room: str, pos: tuple[int, int], **extra: Any) -> None:
        taken[room].append(pos)
        lay.things.append({"id": tid, "type": ttype, "room": room, "pos": list(pos), **extra})

    cx, cy = lay.centre(lay.spawn)
    add("spawn", "player_start", lay.spawn, (cx, cy))
    add("exitsw", "exit_switch", lay.goal, _wall_point(lay, lay.goal), interactive=True)
    # Keys and switches, shallowest door first: each goes in a floor-0 room reachable
    # without its own door and without any door whose key or switch is not placed yet.
    colours = iter(KEY_TYPES)
    gated = sorted((e for e in lay.edges if e.kind in ("locked_door", "switch_door")),
                   key=lambda e: (depth.get(e.b, 0), e.id))  # fmt: skip
    for n, e in enumerate(gated):
        cut = [g.id for g in gated[n:]]
        near = sorted(r for r in _reachable_without(lay, cut, lay.spawn) if lay.floor[r] == 0)
        near = [r for r in near if r not in lay.spare and r != lay.trap_room] or near
        room = near[int(rng.integers(len(near)))]
        if e.kind == "locked_door":
            e.requires = next(colours)
            add(f"key_{e.id}", e.requires, room, _free_point(lay, room, taken[room]),
                interactive=True)  # fmt: skip
        else:
            add(f"sw_{e.id}", "switch", room, _wall_point(lay, room), opens=e.id,
                interactive=True)  # fmt: skip
    # Pickups stay out of the trap room, so two defects never share a room.
    rooms = sorted(r for r in lay.cells if r != lay.trap_room)
    n_health = int(rng.integers(1 if "bad_pickup" in kinds else 0, 3))
    for i in range(n_health):
        r = rooms[int(rng.integers(len(rooms)))]
        add(f"health{i}", "health", r, _free_point(lay, r, taken[r]), interactive=True)
    for ttype, hi in (("armour", 2), ("ammo", 3)):
        for i in range(int(rng.integers(0, hi + 1))):
            r = rooms[int(rng.integers(len(rooms)))]
            add(f"{ttype}{i}", ttype, r, _free_point(lay, r, taken[r]), interactive=True)
    if rng.random() < 0.3:
        r = rooms[int(rng.integers(len(rooms)))]
        if r != lay.spawn:
            add("imp0", "monster_imp", r, _free_point(lay, r, taken[r]))


# --- defects ---------------------------------------------------------------------------


def _choose_defects(lay: _Layout, kinds: Sequence[str]) -> list[dict[str, Any]]:
    rng = lay.rng
    needed_rooms = {t["room"] for t in lay.things
                    if t["type"] in ("player_start", "exit_switch", "switch", *KEY_TYPES)}  # fmt: skip
    tele_rooms = {a for _, a, _, _ in lay.teleports} | {b for _, _, b, _ in lay.teleports}
    lift_rooms = {r for e in lay.edges if e.kind == "lift" for r in (e.a, e.b)}
    blocked: set[str] = set()  # rooms another defect makes unreachable
    used_targets: set[str] = set()
    out: list[dict[str, Any]] = []
    # wrong_tag first: it decides which rooms the others must avoid.
    order = sorted(kinds, key=lambda k: k != "wrong_tag")
    for n, kind in enumerate(order):
        d: dict[str, Any] = {"id": f"d{n}_{kind}", "kind": kind,
                             "expected_oracle": EXPECTED_ORACLE[kind],
                             "severity": SEVERITY[kind], "params": {}}  # fmt: skip
        if kind == "wrong_tag":
            options = []
            for t in lay.things:
                if t["type"] != "switch":
                    continue
                lost = set(lay.cells) - _reachable_without(lay, [t["opens"]], lay.spawn)
                if lost and not lost & needed_rooms:
                    options.append((t["id"], lost))
            if not options:
                raise ProcgenError("no switch whose door cuts off a spare part of the map")
            target, lost = options[int(rng.integers(len(options)))]
            blocked |= lost
        elif kind == "wrong_special":
            doors = [e.id for e in lay.edges if e.kind == "locked_door"
                     and e.a not in blocked and e.b not in blocked]  # fmt: skip
            target = _pick(rng, doors, "locked door")
        elif kind == "geometry_trap":
            # The dead end _levels reserved; a loop or teleport may since have joined it.
            r = lay.trap_room
            ok = (r is not None and lay.degree(r) == 1 and r not in needed_rooms
                  and r not in tele_rooms and r not in lift_rooms and r not in blocked)  # fmt: skip
            target = _pick(rng, [r] if ok and r is not None else [], "dead-end room")
            d["params"] = {"drop": int(rng.choice([64, 96]))}
        elif kind == "one_way_lift":
            # Down into a pit (always a leaf with nothing the map needs).
            lifts = [e.id for e in lay.edges if e.kind == "lift" and lay.floor[e.b] < 0
                     and e.a not in blocked]  # fmt: skip
            target = _pick(rng, lifts, "lift down into a pit")
            used_targets.add(target.split("_", 2)[2])
        elif kind == "teleport_loop":
            if not lay.teleports or {lay.teleports[0][1], lay.teleports[0][2]} & blocked:
                raise ProcgenError("no usable teleport pair")
            target = lay.teleports[0][0]
            d["params"] = {"partner": lay.teleports[1][0]}
        elif kind == "bad_pickup":
            healths = [t["id"] for t in lay.things
                       if t["type"] == "health" and t["room"] not in blocked]  # fmt: skip
            target = _pick(rng, healths, "health pickup")
        else:  # missing_blocking
            options2 = []
            for r in sorted(lay.cells):
                if r in blocked or r in used_targets:
                    continue
                for side in lay.outer_sides(r):
                    if side not in lay.reserved_sides.get(r, set()):
                        options2.append((r, side))
            if not options2:
                raise ProcgenError("no outer wall for a leak")
            target, side = options2[int(rng.integers(len(options2)))]
            lay.reserved_sides.setdefault(target, set()).add(side)
            x0, y0, x1, y1 = lay.rect(target)
            mid = (x0 + x1) // 2 if side in ("north", "south") else (y0 + y1) // 2
            d["params"] = {"wall": side, "segment": [mid - 64, mid + 64]}
        d["target"] = target
        if kind in ("geometry_trap", "missing_blocking"):
            used_targets.add(target)
        d["rooms"] = _defect_rooms(lay, d, blocked if kind == "wrong_tag" else set())
        out.append(d)
    for i, a in enumerate(out):
        for b in out[i + 1:]:
            if a["rooms"] & b["rooms"]:
                raise ProcgenError(f"{a['id']} and {b['id']} share a room")
    for d in out:
        del d["rooms"]
    return out


def _defect_rooms(lay: _Layout, d: dict[str, Any], lost: set[str]) -> set[str]:
    """The rooms a defect touches; two defects may not share one."""
    t = d["target"]
    if d["kind"] in ("missing_blocking", "geometry_trap"):
        return {t}
    if d["kind"] in ("one_way_lift", "wrong_special"):
        e = next(e for e in lay.edges if e.id == t)
        return {e.b} if d["kind"] == "one_way_lift" else {e.a, e.b}
    if d["kind"] == "teleport_loop":
        return {r for pid, a, b, _ in lay.teleports for r in (a, b) if pid == t}
    thing = next(x for x in lay.things if x["id"] == t)  # wrong_tag, bad_pickup
    return {thing["room"]} | lost


def _pick(rng: np.random.Generator, options: Sequence[str], what: str) -> str:
    if not options:
        raise ProcgenError(f"no {what} to target")
    return options[int(rng.integers(len(options)))]


# --- assembly --------------------------------------------------------------------------


def _spec_dict(lay: _Layout, seed: int, slot: int, defects: list[dict[str, Any]]) -> dict[str, Any]:
    rooms = []
    for r in sorted(lay.cells):
        x0, y0, x1, y1 = lay.rect(r)
        rooms.append({"id": r, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                      "floor": lay.floor[r], "ceiling": lay.ceiling[r]})  # fmt: skip
    portals: list[dict[str, Any]] = []
    for e in lay.edges:
        kind = "door" if e.kind == "switch_door" else e.kind
        p: dict[str, Any] = {"id": e.id, "a": e.a, "b": e.b, "kind": kind,
                             "width": 128 if kind == "open" else 96}  # fmt: skip
        if kind == "locked_door":
            p["requires"] = e.requires
        portals.append(p)
    for pid, a, b, at in lay.teleports:
        portals.append({"id": pid, "a": a, "b": b, "kind": "teleport", "at": list(at)})
    # Keys the goal needs: those of locked doors no route to the exit avoids.
    required = sorted(
        str(e.requires) for e in lay.edges
        if e.kind == "locked_door" and lay.goal not in _reachable_without(lay, [e.id], lay.spawn)
    )
    n = len(lay.cells)
    return {
        "name": f"GEN{seed}", "slot": slot, "cell_size": 64, "seed": seed,
        "rooms": rooms, "portals": portals, "things": lay.things,
        "goal": {"kind": "reach_exit", "target": "exitsw", "requires": required,
                 "time_limit_tics": int(min(21000, max(10500, 1750 * n)))},
        "defects": defects,
    }  # fmt: skip


def _attempt(rng: np.random.Generator, cfg: ProcgenConfig, seed: int, slot: int,
             kinds: list[str]) -> MapSpec:  # fmt: skip
    lay = _Layout(rng, cfg)
    _grow(lay, int(rng.integers(cfg.min_rooms, cfg.max_rooms + 1)))
    _tree(lay)
    _levels(lay, set(kinds))
    _edges(lay, set(kinds))
    _teleports(lay, set(kinds))
    _place_things(lay, set(kinds))
    defects = _choose_defects(lay, kinds)
    spec = MapSpec.model_validate(_spec_dict(lay, seed, slot, defects))
    # Exactly what `cli.build` does with it, for both variants.
    intended = compile_checked(spec.without_defects())
    built = compile_checked(spec) if spec.defects else intended
    ground_truth(spec, intended, built, "buggy")
    return spec


def generate(seed: int, cfg: ProcgenConfig | None = None, slot: int = 1) -> MapSpec:
    """The map for `seed`. Deterministic: the same seed always gives the same spec."""
    cfg = cfg or ProcgenConfig()
    rng = np.random.default_rng(seed)
    k = int(rng.integers(0, cfg.max_defects + 1))
    kinds = [str(x) for x in rng.choice(DEFECT_KINDS, size=k, replace=False)]
    errors = []
    for attempt in range(MAX_ATTEMPTS):
        try:
            return _attempt(np.random.default_rng([seed, attempt]), cfg, seed, slot, kinds)
        except (ProcgenError, SpecError, GeometryError, ValueError) as e:
            errors.append(f"{type(e).__name__}: {e}")
    raise ProcgenError(f"seed {seed}: no valid map in {MAX_ATTEMPTS} attempts; last: "
                       f"{errors[-3:]}")  # fmt: skip


def spec_yaml(spec: MapSpec) -> str:
    data = spec.model_dump(mode="json", exclude_defaults=False)
    header = (f"# Generated by qatester.mapgen.procgen from seed {spec.seed}. "
              "Do not hand-edit; regenerate instead.\n")  # fmt: skip
    return header + yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=100)


def write_pack(out_dir: Path, seed_list: Iterable[int],
               cfg: ProcgenConfig | None = None) -> list[Path]:  # fmt: skip
    """Write one spec per seed, slots 1..n, ready for `qatester.mapgen.cli build`. A WAD
    holds at most 32 maps, so a pack is at most 32 seeds."""
    seed_list = list(seed_list)
    if len(seed_list) > 32:
        raise ValueError("a pack holds at most 32 maps (MAP01-MAP32)")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for slot, seed in enumerate(seed_list, start=1):
        path = out_dir / f"gen_{seed:07d}.yaml"
        path.write_text(spec_yaml(generate(seed, cfg, slot=slot)), encoding="utf-8")
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Write a pack of generated MapSpecs.")
    ap.add_argument("--split", choices=sorted(SPLIT_BASE), required=True)
    ap.add_argument("--count", type=int, default=32)
    ap.add_argument("--start", type=int, default=0, help="offset into the split's seeds")
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args(argv)
    paths = write_pack(args.out, seeds(args.split, args.count, args.start))
    print(f"wrote {len(paths)} specs to {args.out}")


if __name__ == "__main__":
    main()
