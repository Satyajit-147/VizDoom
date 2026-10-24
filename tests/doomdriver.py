"""Scripted ViZDoom driver for engine-level map tests: warp, face, walk, jump, use.

Test-only. Uses GZDoom cheats (warp, give), which the agent never gets.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import vizdoom as vzd

from qatester.mapgen.spec import Room

GV = vzd.GameVariable
KEY_ITEMS = {"key_red": "RedCard", "key_blue": "BlueCard", "key_yellow": "YellowCard"}
WALL = 16
NOOP = [0, 0, 0, 0]  # MOVE_FORWARD, USE, TURN_LEFT_RIGHT_DELTA, JUMP


class Driver:
    def __init__(self, cfg: Path) -> None:
        g = vzd.DoomGame()
        g.load_config(str(cfg))
        g.add_game_args("+sv_cheats 1")
        g.set_available_buttons(
            [vzd.Button.MOVE_FORWARD, vzd.Button.USE, vzd.Button.TURN_LEFT_RIGHT_DELTA,
             vzd.Button.JUMP]
        )  # fmt: skip
        g.set_seed(0)
        g.init()
        self.game = g
        self.keys: set[str] = set()

    def _act(self, buttons: list[float], tics: int = 1) -> None:
        """Every action goes through here, so a subclass can record it."""
        self.game.make_action(buttons, tics)

    def close(self) -> None:
        self.game.close()

    def reset(self) -> None:
        self.game.new_episode()
        self.keys = set()

    def pos(self) -> tuple[float, float, float]:
        g = self.game
        return (
            g.get_game_variable(GV.POSITION_X),
            g.get_game_variable(GV.POSITION_Y),
            g.get_game_variable(GV.POSITION_Z),
        )

    def health(self) -> float:
        return self.game.get_game_variable(GV.HEALTH)

    def warp(self, x: float, y: float) -> None:
        self.game.send_game_command(f"warp {x} {y}")
        self.idle(2)

    def give(self, key_type: str) -> None:
        self.game.send_game_command(f"give {KEY_ITEMS[key_type]}")
        self.keys.add(key_type)
        self.idle(2)

    def face(self, angle: float) -> None:
        """Turn to an absolute angle (degrees, 0 = east, counter-clockwise).
        A positive TURN_LEFT_RIGHT_DELTA turns clockwise."""
        current = self.game.get_game_variable(GV.ANGLE)
        delta = (current - angle + 180.0) % 360.0 - 180.0
        self._act([0, 0, delta, 0], 1)

    def face_point(self, x: float, y: float) -> None:
        px, py, _ = self.pos()
        self.face(math.degrees(math.atan2(y - py, x - px)))

    def forward(self, tics: int, jump: bool = False) -> None:
        for _ in range(tics):
            if self.game.is_episode_finished():
                return
            self._act([1, 0, 0, int(jump)], 1)

    def use(self) -> None:
        self._act([0, 1, 0, 0], 1)
        self._act(NOOP, 1)

    def idle(self, tics: int) -> None:
        self._act(NOOP, tics)

    def walk_until(self, inside: Callable[[float, float], bool], tics: int) -> bool:
        """Walk forward until `inside(x, y)` is true, for at most `tics` tics."""
        for _ in range(tics):
            if self.game.is_episode_finished():
                return False
            x, y, _ = self.pos()
            if inside(x, y):
                return True
            self._act([1, 0, 0, 0], 1)
        x, y, _ = self.pos()
        return inside(x, y)


def inside_room(room: Room, margin: float = WALL) -> Callable[[float, float], bool]:
    return lambda x, y: (
        room.x0 + margin <= x <= room.x1 - margin and room.y0 + margin <= y <= room.y1 - margin
    )


def approach(room: Room, wall_axis: str, wall_at: float, along: float,
             dist: float) -> tuple[float, float, float]:  # fmt: skip
    """A point `dist` from the wall inside `room`, and the angle facing the wall."""
    if wall_axis == "x":
        sign = -1 if room.x1 == wall_at else 1  # room lies on the low-x side
        return wall_at + sign * dist, along, 0.0 if sign < 0 else 180.0
    sign = -1 if room.y1 == wall_at else 1
    return along, wall_at + sign * dist, 90.0 if sign < 0 else 270.0
