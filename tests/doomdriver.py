"""Scripted ViZDoom driver for engine-level map tests: warp, face, walk, use.

Test-only. Uses GZDoom cheats (warp, give), which the agent never gets.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import vizdoom as vzd

GV = vzd.GameVariable
FORWARD, USE, TURN = 0, 1, 2
KEY_ITEMS = {"key_red": "RedCard", "key_blue": "BlueCard", "key_yellow": "YellowCard"}


class Driver:
    def __init__(self, cfg: Path) -> None:
        g = vzd.DoomGame()
        g.load_config(str(cfg))
        g.add_game_args("+sv_cheats 1")
        g.set_available_buttons(
            [vzd.Button.MOVE_FORWARD, vzd.Button.USE, vzd.Button.TURN_LEFT_RIGHT_DELTA]
        )
        g.set_seed(0)
        g.init()
        self.game = g

    def close(self) -> None:
        self.game.close()

    def reset(self) -> None:
        self.game.new_episode()

    def pos(self) -> tuple[float, float, float]:
        g = self.game
        return (
            g.get_game_variable(GV.POSITION_X),
            g.get_game_variable(GV.POSITION_Y),
            g.get_game_variable(GV.POSITION_Z),
        )

    def warp(self, x: float, y: float) -> None:
        self.game.send_game_command(f"warp {x} {y}")
        self.idle(2)

    def give(self, key_type: str) -> None:
        self.game.send_game_command(f"give {KEY_ITEMS[key_type]}")
        self.idle(2)

    def face(self, angle: float) -> None:
        """Turn to an absolute angle (degrees, 0 = east, counter-clockwise).
        A positive TURN_LEFT_RIGHT_DELTA turns clockwise."""
        current = self.game.get_game_variable(GV.ANGLE)
        delta = (current - angle + 180.0) % 360.0 - 180.0
        self.game.make_action([0, 0, delta], 1)

    def face_point(self, x: float, y: float) -> None:
        px, py, _ = self.pos()
        self.face(math.degrees(math.atan2(y - py, x - px)))

    def forward(self, tics: int) -> None:
        for _ in range(tics):
            if self.game.is_episode_finished():
                return
            self.game.make_action([1, 0, 0], 1)

    def use(self) -> None:
        self.game.make_action([0, 1, 0], 1)
        self.game.make_action([0, 0, 0], 1)

    def idle(self, tics: int) -> None:
        self.game.make_action([0, 0, 0], tics)

    def walk_until(self, inside: Callable[[float, float], bool], tics: int) -> bool:
        """Walk forward until `inside(x, y)` is true, for at most `tics` tics."""
        for _ in range(tics):
            if self.game.is_episode_finished():
                return False
            x, y, _ = self.pos()
            if inside(x, y):
                return True
            self.game.make_action([1, 0, 0], 1)
        x, y, _ = self.pos()
        return inside(x, y)
