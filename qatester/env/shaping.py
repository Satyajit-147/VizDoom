"""Potential-based shaping toward the goal (docs/decisions.md D35).

Phi(s) = -(the length of the remaining route to finish the level): to each goal key
not yet held, nearest first, then to the exit. Reward gets w * (Phi(s') - Phi(s)) per
step, so walking toward the next goal pays and walking away costs the same. Over an
episode the bonus telescopes to Phi(end) - Phi(start): it cannot be farmed, and (Ng,
Harada and Russell, 1999) it does not change which policy is best.

Route lengths come from the intended level only, through the env; the agent never sees
them. Within a room the distance is a straight line. Between rooms it runs through
portal openings: a graph whose nodes are the openings, joined by straight lines within
a room, with locked doors passable only while their key is held. Teleports join their
pad to their arrival point, one way. Picking up a key leaves Phi continuous: the route
from the key's spot onward was already part of the old route.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from dataclasses import dataclass

from qatester.mapgen.groundtruth import GroundTruth
from qatester.oracles.level import Level, rect_distance

Point = tuple[float, float]
UNREACHABLE = 1e5  # mu: a target no route reaches (should not happen on a valid level)


@dataclass(frozen=True)
class _Node:
    point: Point
    sectors: frozenset[int]  # room sectors this node can be walked to from


class GoalDistance:
    def __init__(self, truth: GroundTruth) -> None:
        self.level = lv = Level(truth.intent_only() if "defects" in truth.data else truth)
        d = truth.data
        self.room_rects = {r["id"]: tuple(r["rect"]) for r in d["rooms"]}
        nodes: list[_Node] = []
        self.locks: list[str | None] = []  # per node: key needed to use it
        self.jumps: list[tuple[int, int]] = []  # teleport pad node -> arrival node
        for p in lv.portals.values():
            a, b = lv.room_sector[p.a], lv.room_sector[p.b]
            centre = ((p.area[0] + p.area[2]) / 2, (p.area[1] + p.area[3]) / 2)
            if p.kind == "teleport":
                assert p.dest is not None
                nodes.append(_Node(centre, frozenset({a})))
                self.locks.append(None)
                nodes.append(_Node(p.dest, frozenset({b})))
                self.locks.append(None)
                self.jumps.append((len(nodes) - 2, len(nodes) - 1))
            else:
                nodes.append(_Node(centre, frozenset({a, b})))
                self.locks.append(p.requires)
        self.nodes = nodes
        self.keys: dict[str, list[Point]] = {
            t["type"]: [] for t in d["things"] if t["type"].startswith("key_")
        }
        for t in d["things"]:
            if t["type"].startswith("key_"):
                self.keys[t["type"]].append((float(t["pos"][0]), float(t["pos"][1])))
        self.required = tuple(k for k in d["goal"].get("requires") or () if k in self.keys)
        goal = d["goal"]
        line = d["switch_lines"].get(goal.get("target") or "")
        if line is not None:
            self.exit: Point | None = ((line[0] + line[2]) / 2, (line[1] + line[3]) / 2)
            self.exit_sector = lv.room_sector[goal["room"]]
        else:
            self.exit, self.exit_sector = None, -1
        self._cache: dict[tuple[Point, int, frozenset[str]], list[float]] = {}

    # --- geometry ------------------------------------------------------------------
    def room_sector(self, x: float, y: float) -> int:
        """The room sector of a point: the room containing it, else the nearest room
        (doorways and wall strips belong to no room)."""
        room = self.level.truth.room_at(x, y)
        if room is None:
            room = min(self.room_rects, key=lambda r: rect_distance(x, y, self.room_rects[r]))
        return self.level.room_sector[room]

    def _to_target(self, target: Point, t_sector: int, held: frozenset[str]) -> list[float]:
        """For every node, the route length from it to `target` (Dijkstra, backwards)."""
        key = (target, t_sector, held)
        if key in self._cache:
            return self._cache[key]
        n = len(self.nodes)
        open_ = [i for i in range(n) if self.locks[i] is None or self.locks[i] in held]
        dist = [math.inf] * n
        heap: list[tuple[float, int]] = []
        for i in open_:
            if t_sector in self.nodes[i].sectors:
                dist[i] = math.dist(self.nodes[i].point, target)
                heapq.heappush(heap, (dist[i], i))
        rev_jump = {b: a for a, b in self.jumps}
        while heap:
            du, u = heapq.heappop(heap)
            if du > dist[u]:
                continue
            nbrs = [(v, math.dist(self.nodes[u].point, self.nodes[v].point)) for v in open_
                    if v != u and self.nodes[u].sectors & self.nodes[v].sectors]  # fmt: skip
            if u in rev_jump:
                nbrs.append((rev_jump[u], 0.0))  # stepping on the pad arrives here
            for v, w in nbrs:
                if du + w < dist[v]:
                    dist[v] = du + w
                    heapq.heappush(heap, (dist[v], v))
        self._cache[key] = dist
        return dist

    def distance(self, x: float, y: float, target: Point, t_sector: int,
                 held: frozenset[str], sector: int | None = None) -> float:  # fmt: skip
        s = self.room_sector(x, y) if sector is None else sector
        if s == t_sector:
            return math.dist((x, y), target)
        dist = self._to_target(target, t_sector, held)
        best = min((math.dist((x, y), self.nodes[i].point) + dist[i]
                    for i in range(len(self.nodes)) if s in self.nodes[i].sectors),
                   default=math.inf)  # fmt: skip
        return min(best, UNREACHABLE)

    # --- the potential ---------------------------------------------------------------
    def remaining(self, x: float, y: float, keys: Iterable[str]) -> float:
        """Route length left: to each goal key not held (nearest first), then the exit."""
        if self.exit is None:
            return 0.0
        held = frozenset(keys)
        here, sector, total = (x, y), self.room_sector(x, y), 0.0
        todo = [k for k in self.required if k not in held]
        while todo:
            options = [(self.distance(*here, p, self.room_sector(*p), held, sector), k, p)
                       for k in todo for p in self.keys[k]]  # fmt: skip
            d, k, p = min(options)
            total += d
            here, sector, held = p, self.room_sector(*p), held | {k}
            todo.remove(k)
        return total + self.distance(*here, self.exit, self.exit_sector, held, sector)

    def potential(self, x: float, y: float, keys: Iterable[str]) -> float:
        return -self.remaining(x, y, keys)
