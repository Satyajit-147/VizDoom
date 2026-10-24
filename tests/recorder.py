"""A Driver that records a StepRecord after every action, from the engine's own state,
to feed real gameplay to the oracles. A warp (cheat teleport) or reset closes the current
episode: a cheat jump is not gameplay."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import vizdoom as vzd
from doomdriver import Driver

from qatester.mapgen.groundtruth import GroundTruth
from qatester.oracles.base import EpisodeContext, StepRecord

GV = vzd.GameVariable


class RecordingDriver(Driver):
    def __init__(self, cfg: Path) -> None:
        super().__init__(cfg)
        self.episodes: list[list[StepRecord]] = []
        self.current: list[StepRecord] = []

    def _act(self, buttons: list[float], tics: int = 1) -> None:
        super()._act(buttons, tics)
        g = self.game
        finished = g.is_episode_finished()
        state = None if finished else g.get_state()
        sectors = state.sectors if state is not None else []
        last = self.current[-1] if self.current else None
        rec = StepRecord(
            tic=g.get_episode_time(),
            x=g.get_game_variable(GV.POSITION_X), y=g.get_game_variable(GV.POSITION_Y),
            z=g.get_game_variable(GV.POSITION_Z), angle=g.get_game_variable(GV.ANGLE),
            on_ground=bool(g.get_game_variable(GV.ON_GROUND)),
            dead=bool(g.get_game_variable(GV.DEAD)),
            health=g.get_game_variable(GV.HEALTH), armour=g.get_game_variable(GV.ARMOR),
            ammo=g.get_game_variable(GV.AMMO2),
            killcount=int(g.get_game_variable(GV.KILLCOUNT)),
            keys=frozenset(self.keys),
            move_dirs=frozenset({"forward"}) if buttons[0] else frozenset(),
            use=bool(buttons[1]), jump=bool(buttons[3]),
            sector_floors=tuple(s.floor_height for s in sectors)
            or (last.sector_floors if last else ()),
            sector_ceilings=tuple(s.ceiling_height for s in sectors)
            or (last.sector_ceilings if last else ()),
            done=finished,
        )  # fmt: skip
        self.current.append(rec)

    def close_episode(self, truncated: bool = False) -> None:
        if self.current:
            self.current[-1] = replace(self.current[-1], done=True, truncated=truncated)
            self.episodes.append(self.current)
        self.current = []

    def warp(self, x: float, y: float) -> None:
        self.close_episode()
        self.game.send_game_command(f"warp {x} {y}")
        Driver.idle(self, 2)
        self.current = []  # the warp's own frames are not gameplay

    def reset(self) -> None:
        self.close_episode()
        super().reset()

    def runs(self, truth: GroundTruth) -> list[tuple[EpisodeContext, list[StepRecord]]]:
        self.close_episode()
        view = truth.intent_only()
        return [(EpisodeContext(truth.data["map"], i, i, view, 1), ep)
                for i, ep in enumerate(self.episodes) if len(ep) > 1]  # fmt: skip
