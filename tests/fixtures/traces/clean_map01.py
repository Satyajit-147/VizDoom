"""MAP01 CLEAN trace corpus: legitimate play only. Every oracle must stay silent on every
trace, alone and as one run (PLAN M4 acceptance: zero findings on CLEAN).

MAP01: start (0..512, 0..448), hall east of it through an open portal, loft north of
start through a door, vault north of hall through a red-key door, exit on the vault's
east wall.
"""

from __future__ import annotations

import numpy as np
from tracebuilder import Trace, room_rect

from qatester.mapgen.groundtruth import GroundTruth
from qatester.oracles.base import StepRecord

EPISODE_TICS = 10_500


def playthrough(t: GroundTruth) -> list[StepRecord]:
    """Full legitimate run: hall, loft door, red key, locked door, exit."""
    tr = Trace(t)
    tr.path((450, 224), (700, 224), (850, 230), (700, 224), (450, 224), (256, 300), (256, 400))
    tr.face(90).use().open_door("p_start_loft").wait(40)
    tr.path((256, 520), (256, 690)).pick_up(key="key_red")
    tr.path((256, 520), (256, 400), (256, 300), (450, 224), (700, 224), (856, 300), (856, 400))
    tr.face(90).use().open_door("p_hall_vault").wait(40)
    tr.path((856, 520), (1150, 700)).face(0).use()
    return tr.end(truncated=False)


def idle(t: GroundTruth) -> list[StepRecord]:
    """Stands at spawn by choice until time runs out."""
    return Trace(t).until(EPISODE_TICS).end()


def wander(t: GroundTruth, seed: int = 7) -> list[StepRecord]:
    """Random walk over start and hall, crossing the open portal now and then."""
    rng = np.random.default_rng(seed)
    tr = Trace(t)
    room = "start"
    while tr.tic < EPISODE_TICS:
        if rng.random() < 0.3:
            tr.path((480 if room == "start" else 544, 224), (544 if room == "start" else 480, 224))
            room = "hall" if room == "start" else "start"
        x0, y0, x1, y1 = room_rect(t, room)
        tr.walk_to(float(rng.uniform(x0 + 40, x1 - 40)), float(rng.uniform(y0 + 40, y1 - 40)))
        if rng.random() < 0.2:
            tr.wait(int(rng.integers(20, 200)))
    return tr.end()


def hammer_locked_door(t: GroundTruth) -> list[StepRecord]:
    """Tries the red door ten times without the key and leans on it; never gets in."""
    tr = Trace(t).path((450, 224), (700, 224), (856, 300), (856, 400)).face(90)
    for _ in range(10):
        tr.use().push(40)
    return tr.push(400).until(EPISODE_TICS).end()


def corner_two_ways(t: GroundTruth) -> list[StepRecord]:
    """Pushes into start's south-east corner forward and right: two directions blocked."""
    tr = Trace(t).walk_to(496, 16).face(330)
    return tr.push(600, ("forward", "right")).until(2000).end()


def medkit_to_full(t: GroundTruth) -> list[StepRecord]:
    """Hurt, then healed by the hall medikit to exactly 100."""
    tr = Trace(t, health=80).path((450, 224), (700, 224), (850, 224)).pick_up(health=100)
    return tr.wait(200).end()


def wall_hug(t: GroundTruth) -> list[StepRecord]:
    """Walks along every wall of start at one player radius."""
    tr = Trace(t).path((16, 224), (16, 16), (496, 16), (496, 150), (496, 16), (16, 16))
    return tr.path((16, 432), (240, 432), (16, 432)).end()


CORPUS = [playthrough, idle, wander, hammer_locked_door, corner_two_ways, medkit_to_full,
          wall_hug]  # fmt: skip
