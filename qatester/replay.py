"""Replay a recorded episode (.lmp) through the oracles.

Used to check a finding after the fact (is it real, does it still reproduce with the
current oracles) and by the regression experiment (E7: replay the buggy build's Bug
Book on the fixed build). The demo is replayed tic by tic and a StepRecord is read from
the engine at every `frame_skip`-th tic, so the oracles see what they saw in training
(frame skip 4) or every tic (frame skip 1).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import vizdoom as vzd

from qatester.env.actions import BUTTONS
from qatester.env.vizdoom_env import _I, GAME_VARIABLES, TURN_DELTA
from qatester.env.world import World
from qatester.mapgen.groundtruth import GroundTruth
from qatester.oracles.base import EpisodeContext, Finding, OracleConfig, StepRecord
from qatester.oracles.registry import make_oracles

GV = vzd.GameVariable
MOVES = (("MOVE_FORWARD", "forward"), ("MOVE_BACKWARD", "back"), ("MOVE_LEFT", "left"),
         ("MOVE_RIGHT", "right"))  # fmt: skip


def replay_records(demo: Path, wad: Path, doom_map: str, truth: GroundTruth,
                   frame_skip: int = 4, time_limit: int | None = None) -> list[StepRecord]:  # fmt: skip
    """The episode in `demo` as the env would have recorded it."""
    intent = truth.intent_only() if "defects" in truth.data else truth
    world = World(intent)
    limit = time_limit or int(intent.data["goal"]["time_limit_tics"])
    g = vzd.DoomGame()
    g.set_doom_scenario_path(str(wad))
    g.set_doom_map(doom_map.lower())
    g.set_doom_skill(3)
    g.set_episode_start_time(14)
    g.set_episode_timeout(limit)
    g.set_window_visible(False)
    g.set_sound_enabled(False)
    g.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    g.set_objects_info_enabled(True)
    g.set_sectors_info_enabled(True)
    g.set_available_buttons([getattr(vzd.Button, b) for b in BUTTONS] + [TURN_DELTA])
    g.set_available_game_variables(list(GAME_VARIABLES))
    g.init()
    short = Path(tempfile.mkdtemp(prefix="qa"))  # ViZDoom hangs on long paths (D34)
    records: list[StepRecord] = []
    try:
        shutil.copyfile(demo, short / "d.lmp")
        g.replay_episode(str(short / "d.lmp"))
        taken: set[str] = set()
        floors: tuple[float, ...] = ()
        ceilings: tuple[float, ...] = ()
        names = {b: i for i, b in enumerate(BUTTONS)}
        t0: int | None = None
        while not g.is_episode_finished():
            g.advance_action(1)
            finished = g.is_episode_finished()
            tic = g.get_episode_time()
            t0 = tic if t0 is None else t0
            state = None if finished else g.get_state()
            if state is not None:
                taken |= world.taken(state.objects)
                floors = tuple(s.floor_height for s in state.sectors)
                ceilings = tuple(s.ceiling_height for s in state.sectors)
            if not finished and (tic - t0) % frame_skip != frame_skip - 1:
                continue
            v: list[float] = (
                list(state.game_variables) if state is not None and state.game_variables is not None
                else [g.get_game_variable(x) for x in GAME_VARIABLES]
            )  # fmt: skip
            a = g.get_last_action()
            dead = bool(v[_I[GV.DEAD]])
            records.append(StepRecord(
                tic=tic, x=float(v[_I[GV.POSITION_X]]), y=float(v[_I[GV.POSITION_Y]]),
                z=float(v[_I[GV.POSITION_Z]]), vx=float(v[_I[GV.VELOCITY_X]]),
                vy=float(v[_I[GV.VELOCITY_Y]]), vz=float(v[_I[GV.VELOCITY_Z]]),
                angle=float(v[_I[GV.ANGLE]]), on_ground=bool(v[_I[GV.ON_GROUND]]),
                dead=dead, health=float(v[_I[GV.HEALTH]]), armour=float(v[_I[GV.ARMOR]]),
                ammo=float(v[_I[GV.AMMO2]]), keys=world.keys_held(taken),
                killcount=int(v[_I[GV.KILLCOUNT]]),
                move_dirs=frozenset(d for b, d in MOVES if a[names[b]]),
                use=bool(a[names["USE"]]), jump=bool(a[names["JUMP"]]),
                sector_floors=floors, sector_ceilings=ceilings, done=finished,
                truncated=finished and not dead and tic >= limit,
            ))  # fmt: skip
    finally:
        g.close()
        shutil.rmtree(short, ignore_errors=True)
    return records


def oracle_findings(records: list[StepRecord], truth: GroundTruth, map_name: str,
                    frame_skip: int = 4, cfg: OracleConfig | None = None) -> list[Finding]:  # fmt: skip
    """Every oracle's findings on one recorded episode (streamed + finalized)."""
    intent = truth.intent_only() if "defects" in truth.data else truth
    oracles = make_oracles(cfg)
    ctx = EpisodeContext(map_name, 0, 0, intent, frame_skip)
    for o in oracles:
        o.reset(ctx)
    found = [f for r in records for o in oracles for f in o.observe(r)]
    return found + [f for o in oracles for f in o.finalize()]
