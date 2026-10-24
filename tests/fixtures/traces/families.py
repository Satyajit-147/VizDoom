"""One positive and at least two near-miss negatives per oracle family (PLAN M4).

Each fixture takes the corpus ground truth, keyed by (map, variant), and returns the
episodes of one run. The near-misses matter most: an agent standing still by choice, a
player hugging a wall, a legitimate teleport, a pickup that lands on exactly 100.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from tracebuilder import Trace, episodes

from qatester.mapgen.groundtruth import GroundTruth
from qatester.oracles.base import EpisodeContext, StepRecord

Truths = dict[tuple[str, str], GroundTruth]
Run = list[tuple[EpisodeContext, list[StepRecord]]]
EPISODE_TICS = 10_500


def _sector(t: GroundTruth, room: str) -> int:
    return int(next(r["sector"] for r in t.data["rooms"] if r["id"] == room))


# --- Family 1: crash and hang --------------------------------------------------------

def crash_positive(t: Truths) -> Run:
    m = t["MAP01", "fixed"]
    tr = Trace(m).walk_to(400, 224)
    tr.emit(error="ViZDoomErrorException: ViZDoom process died")
    return episodes(m, tr.end(truncated=False))


def hang_positive(t: Truths) -> Run:
    m = t["MAP01", "fixed"]
    return episodes(m, Trace(m).walk_to(400, 224).emit(step_wall_s=31.0).end())


def stall_positive(t: Truths) -> Run:
    m = t["MAP01", "fixed"]
    rec = Trace(m).walk_to(400, 224).records
    frozen = [replace(rec[-1]) for _ in range(4)]  # game time stops advancing
    return episodes(m, [*rec, *frozen, replace(frozen[-1], done=True)])


def crash_near_slow_step(t: Truths) -> Run:
    m = t["MAP01", "fixed"]
    return episodes(m, Trace(m).walk_to(400, 224).emit(step_wall_s=25.0).end(False))


def crash_near_standing_still(t: Truths) -> Run:
    m = t["MAP01", "fixed"]
    return episodes(m, Trace(m).wait(2000).end(False))


# --- Family 2: stuck -----------------------------------------------------------------

def stuck_vertical_positive(t: Truths) -> Run:
    """Drops into MAP02's alcove, whose live floor is 64 below the gallery."""
    m = t["MAP02", "buggy"]
    tr = Trace(m, start=(724, 120), angle=270).set_floor(_sector(m, "alcove"), -64)
    return episodes(m, tr.walk_to(724, -100).wait(100).end())


def stuck_wedge_positive(t: Truths) -> Run:
    """Wedged in MAP01 start's corner: pushes hard in all four directions, cannot move."""
    m = t["MAP01", "fixed"]
    tr = Trace(m).walk_to(496, 16).face(270)
    for _ in range(6):
        for direction in ("forward", "left", "back", "right"):
            tr.push(20, (direction,))
    return episodes(m, tr.wait(40).end(False))


def stuck_near_idle(t: Truths) -> Run:
    m = t["MAP01", "fixed"]
    return episodes(m, Trace(m).wait(700).end(False))


def stuck_near_jitter_in_corner(t: Truths) -> Run:
    """Pushes forward while turning fast in a corner: many facings, none sustained.
    Seen in real random play (tests/test_oracles_engine.py)."""
    m = t["MAP01", "fixed"]
    tr = Trace(m).walk_to(496, 16)
    for i in range(150):
        tr.face(270 + (i * 37) % 180).push(4)
    return episodes(m, tr.end(False))


def stuck_near_corner_two_ways(t: Truths) -> Run:
    m = t["MAP01", "fixed"]
    tr = Trace(m).walk_to(496, 16).face(330)
    return episodes(m, tr.push(600, ("forward", "right")).end(False))


def stuck_near_pit_with_lift(t: Truths) -> Run:
    """MAP06's pit is 64 below everything, but it has a working lift: not a trap."""
    m = t["MAP06", "fixed"]
    return episodes(m, Trace(m, start=(256, 800)).wait(300).end(False))


# --- Family 3: out of bounds ---------------------------------------------------------

def bounds_leak_positive(t: Truths) -> Run:
    """Walks through MAP03's leaking hall wall into the void and back."""
    m = t["MAP03", "buggy"]
    tr = Trace(m, start=(256, -160)).path((256, 16), (256, -160))
    return episodes(m, tr.end(False))


def bounds_wall_traversal_positive(t: Truths) -> Run:
    """Jumps from start into the loft through the wall in one step (no teleport there)."""
    m = t["MAP01", "fixed"]
    tr = Trace(m).walk_to(100, 300).teleport(100, 700)
    return episodes(m, tr.end(False))


def bounds_z_clip_positive(t: Truths) -> Run:
    m = t["MAP01", "fixed"]
    tr = Trace(m).walk_to(300, 224).emit(z=-30.0)
    return episodes(m, tr.end(False))


def bounds_near_wall_hug(t: Truths) -> Run:
    m = t["MAP03", "fixed"]
    return episodes(m, Trace(m, start=(40, -112)).path((600, -112), (40, -112)).end(False))


def bounds_near_teleport(t: Truths) -> Run:
    """A legitimate teleport: a huge jump across walls, but it starts on the pad."""
    m = t["MAP06", "fixed"]
    tr = Trace(m, start=(1240, 256), angle=180).walk_to(1150, 256).teleport(1216, 960, 0)
    return episodes(m, tr.wait(100).end(False))


def bounds_near_lift_ride(t: Truths) -> Run:
    """Rides MAP03's lift: the floor moves under the player, z follows it."""
    m = t["MAP03", "fixed"]
    lift = next(s["index"] for s in m.data["sectors"] if s["kind"] == "lift")
    tr = Trace(m, start=(-160, -150), angle=90).set_floor(lift, 0.0).walk_to(-160, -40)
    for h in range(0, 65, 8):
        tr.set_floor(lift, float(h)).emit()
    return episodes(m, tr.walk_to(-160, 120).end(False))


# --- Family 4: unreachable -----------------------------------------------------------

def _orphan_episode(m: GroundTruth, switch_works: bool) -> list[StepRecord]:
    _sx0, sy0, _sx1, sy1 = m.data["switch_lines"]["east_sw"]
    sy = (sy0 + sy1) / 2
    tr = Trace(m).walk_to(256, 440).face(90).use().open_door("p_a_c").wait(40)
    tr.path((256, 600), (48, sy)).face(180).use()
    if switch_works:
        tr.open_door("p_b_f")
    tr.wait(60).path((256, 600), (256, 440), (256, 300), (450, 128), (700, 128), (960, 128))
    if switch_works:
        tr.path((1100, 128), (1400, 224), (1600, 224), (1400, 224), (1248, 400))
        tr.face(90).use().open_door("p_f_h")
        tr.wait(40).walk_to(1248, 672)
        return tr.end(False)
    tr.face(0).use().push(100)
    return tr.until(EPISODE_TICS).end()


def reach_positive(t: Truths) -> Run:
    """MAP04 buggy: the east-wing switch does nothing, six episodes in a row."""
    m = t["MAP04", "buggy"]
    return episodes(m, *[_orphan_episode(m, switch_works=False) for _ in range(6)])


def reach_near_switch_works(t: Truths) -> Run:
    m = t["MAP04", "fixed"]
    return episodes(m, *[_orphan_episode(m, switch_works=True) for _ in range(6)])


def reach_near_never_tried(t: Truths) -> Run:
    """Nobody found the switch: the east wing is a coverage gap, not a bug."""
    m = t["MAP04", "fixed"]
    tr = Trace(m).path((450, 128), (700, 128), (960, 128)).face(0).use().push(100)
    return episodes(m, *[tr.until(EPISODE_TICS).end() for _ in range(1)])


def reach_near_no_key(t: Truths) -> Run:
    """Hammers MAP01's red door without the key, across six episodes."""
    m = t["MAP01", "fixed"]
    runs = []
    for _ in range(6):
        tr = Trace(m).path((450, 224), (700, 224), (856, 300), (856, 400)).face(90)
        for _ in range(8):
            tr.use().push(40)
        runs.append(tr.until(EPISODE_TICS).end())
    return episodes(m, *runs)


# --- Family 5: invariants ------------------------------------------------------------

def invariant_positive(t: Truths) -> Run:
    """MAP05 buggy: the medikit overheals to 200, the blue door opens without the key."""
    m = t["MAP05", "buggy"]
    tr = Trace(m).path((450, 256), (736, 256)).pick_up(health=200)
    tr.walk_to(736, 440).face(90).use().open_door("p_armory_vault").wait(40)
    return episodes(m, tr.path((736, 600), (736, 700)).end(False))


def invariant_near_heal_to_100(t: Truths) -> Run:
    m = t["MAP05", "fixed"]
    tr = Trace(m, health=80).path((450, 256), (736, 256)).pick_up(health=100)
    return episodes(m, tr.wait(100).end(False))


def invariant_near_door_with_key(t: Truths) -> Run:
    m = t["MAP05", "fixed"]
    tr = Trace(m).walk_to(256, 440).face(90).use().open_door("p_start_keyroom").wait(40)
    tr.path((256, 600), (256, 690)).pick_up(key="key_blue")
    tr.path((256, 600), (256, 440), (256, 300), (450, 256), (736, 256), (736, 440))
    tr.face(90).use().open_door("p_armory_vault").wait(40)
    return episodes(m, tr.path((736, 600), (736, 700)).end(False))


def invariant_near_damage(t: Truths) -> Run:
    m = t["MAP05", "fixed"]
    return episodes(m, Trace(m).walk_to(400, 256).damage(60).damage(35).end(False))


# --- Family 6: softlock --------------------------------------------------------------

def _pit_episode(m: GroundTruth) -> list[StepRecord]:
    tr = Trace(m, angle=90).path((256, 450), (256, 560), (256, 700), (100, 800), (400, 800))
    tr.walk_to(256, 600).until(7000).face(270).push(100).use().wait(60)
    return tr.push(200).use().wait(60).until(EPISODE_TICS).end()


def softlock_positive(t: Truths) -> Run:
    """MAP06 buggy: two episodes end in the pit, pushing at a lift that never comes."""
    m = t["MAP06", "buggy"]
    return episodes(m, _pit_episode(m), _pit_episode(m))


def softlock_loop_positive(t: Truths) -> Run:
    """MAP06 buggy: each teleport lands in front of the other's pad."""
    m = t["MAP06", "buggy"]
    tr = Trace(m, start=(1240, 256), angle=180).walk_to(1156, 256)
    for _ in range(3):
        tr.teleport(1176, 960, 180).walk_to(1156, 960)
        tr.teleport(1176, 256, 180).walk_to(1156, 256)
    return episodes(m, tr.end(False))


def softlock_near_idle(t: Truths) -> Run:
    m = t["MAP01", "fixed"]
    return episodes(m, Trace(m).until(EPISODE_TICS).end(), Trace(m).until(EPISODE_TICS).end())


def softlock_near_legit_teleports(t: Truths) -> Run:
    m = t["MAP06", "fixed"]
    tr = Trace(m, start=(1240, 256), angle=180).walk_to(1150, 256).teleport(1216, 960, 0)
    tr.wait(300).walk_to(1150, 960).teleport(1216, 256, 0).wait(300)
    return episodes(m, tr.end(False))


def softlock_near_pit_with_lift(t: Truths) -> Run:
    """MAP06 fixed: drops into the pit, calls the lift, rides out."""
    m = t["MAP06", "fixed"]
    lift = next(s["index"] for s in m.data["sectors"] if s["kind"] == "lift")
    tr = Trace(m, angle=90).path((256, 450), (256, 560), (256, 700), (256, 640)).face(270)
    tr.use().set_floor(lift, -64.0).wait(20).walk_to(256, 550).set_floor(lift, 0.0)
    return episodes(m, tr.wait(20).path((256, 450), (256, 256)).until(EPISODE_TICS).end())


Fixture = Callable[[Truths], Run]
POSITIVES: dict[str, list[tuple[Fixture, set[str]]]] = {
    "crash": [(crash_positive, {"crash"}), (hang_positive, {"hang"}),
              (stall_positive, {"tic_stall"})],
    "stuck": [(stuck_vertical_positive, {"vertical_trap"}), (stuck_wedge_positive, {"wedge"})],
    "out_of_bounds": [(bounds_leak_positive, {"outside_level"}),
                      (bounds_wall_traversal_positive, {"wall_traversal"}),
                      (bounds_z_clip_positive, {"z_clip"})],
    "unreachable": [(reach_positive, {"broken_trigger"})],
    "invariant": [(invariant_positive, {"health_cap", "locked_needs_key"})],
    "softlock": [(softlock_positive, {"confined"}), (softlock_loop_positive, {"teleport_loop"})],
}  # fmt: skip
NEAR_MISSES: dict[str, list[Fixture]] = {
    "crash": [crash_near_slow_step, crash_near_standing_still],
    "stuck": [stuck_near_idle, stuck_near_corner_two_ways, stuck_near_pit_with_lift,
              stuck_near_jitter_in_corner],
    "out_of_bounds": [bounds_near_wall_hug, bounds_near_teleport, bounds_near_lift_ride],
    "unreachable": [reach_near_switch_works, reach_near_never_tried, reach_near_no_key],
    "invariant": [invariant_near_heal_to_100, invariant_near_door_with_key,
                  invariant_near_damage],
    "softlock": [softlock_near_idle, softlock_near_legit_teleports,
                 softlock_near_pit_with_lift],
}  # fmt: skip
