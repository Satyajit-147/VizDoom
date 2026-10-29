"""The discrete actions (PLAN: Action space), as combinations of 9 binary buttons.

The plan's 12 combinations, plus turning in place (12, 13; docs/decisions.md D35). The
plan only turns while moving forward, so lining up on a door or switch meant walking in
circles. With frame skip 4 a held turn button turns 7 degrees on the first step, then
about 14 degrees per step once Doom's turn acceleration kicks in.
"""

from __future__ import annotations

from dataclasses import dataclass

BUTTONS = (
    "MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT", "TURN_LEFT", "TURN_RIGHT",
    "USE", "ATTACK", "JUMP",
)  # fmt: skip


@dataclass(frozen=True)
class Action:
    name: str
    pressed: frozenset[str]

    @property
    def buttons(self) -> list[float]:
        return [1.0 if b in self.pressed else 0.0 for b in BUTTONS]

    @property
    def move_dirs(self) -> frozenset[str]:
        """Commanded movement, in StepRecord's vocabulary."""
        names = {"MOVE_FORWARD": "forward", "MOVE_BACKWARD": "back", "MOVE_LEFT": "left",
                 "MOVE_RIGHT": "right"}  # fmt: skip
        return frozenset(names[b] for b in self.pressed if b in names)

    @property
    def use(self) -> bool:
        return "USE" in self.pressed

    @property
    def jump(self) -> bool:
        return "JUMP" in self.pressed


def _a(name: str, *pressed: str) -> Action:
    return Action(name, frozenset(pressed))


ACTIONS: tuple[Action, ...] = (
    _a("noop"),
    _a("forward", "MOVE_FORWARD"),
    _a("backward", "MOVE_BACKWARD"),
    _a("strafe_left", "MOVE_LEFT"),
    _a("strafe_right", "MOVE_RIGHT"),
    _a("forward_turn_left", "MOVE_FORWARD", "TURN_LEFT"),
    _a("forward_turn_right", "MOVE_FORWARD", "TURN_RIGHT"),
    _a("use", "USE"),
    _a("forward_use", "MOVE_FORWARD", "USE"),
    _a("attack", "ATTACK"),
    _a("jump", "JUMP"),
    _a("forward_jump", "MOVE_FORWARD", "JUMP"),
    _a("turn_left", "TURN_LEFT"),
    _a("turn_right", "TURN_RIGHT"),
)
N_ACTIONS = len(ACTIONS)
INDEX = {a.name: i for i, a in enumerate(ACTIONS)}
