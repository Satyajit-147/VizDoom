"""T2: scripted patrol, wall-following plus USE on contact (PLAN: the "scripted testing"
straw man). It reads only the observation: the depth frame and the scalar vector.

Right-hand rule on the newest depth frame: turn left when the way ahead is blocked, turn
right into openings on the right, otherwise go forward. Anything usable in front
(scalar 17), or a wall at arm's length, gets a USE. When the speed ratio (scalar 6)
says it is pushing without moving, it backs off and strafes for a few steps. A small
chance of a random action breaks the loops a pure wall-follower falls into.
"""

from __future__ import annotations

import numpy as np

from qatester.agents.base import Obs
from qatester.env.actions import INDEX, N_ACTIONS

MU_PER_DEPTH = 7.3  # measured: a wall 64 mu away reads 8-9 in the depth buffer
FRONT_BLOCKED_MU = 96.0
CONTACT_MU = 48.0
OPENING_MU = 192.0
STUCK_STEPS = 6


class WallFollower:
    def __init__(self, rng: np.random.Generator, epsilon: float = 0.05) -> None:
        self.rng = rng
        self.epsilon = epsilon
        self.reset()

    def reset(self) -> None:
        self.stuck = 0
        self.escape: list[int] = []

    def act(self, obs: Obs) -> int:
        if self.escape:
            return self.escape.pop()
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(N_ACTIONS))
        frame = obs["depth"][-1].astype(np.float32) * MU_PER_DEPTH
        band = frame[24:40]
        front = float(band[:, 24:40].min())
        right = float(band[:, 52:].mean())
        s = obs["scalars"]
        usable, speed = bool(s[17] > 0.5), float(s[6])
        self.stuck = self.stuck + 1 if speed < 0.2 else 0
        if self.stuck >= STUCK_STEPS:
            self.stuck = 0
            side = INDEX["strafe_left"] if self.rng.random() < 0.5 else INDEX["strafe_right"]
            self.escape = [side] * 4 + [INDEX["backward"]] * 4  # popped from the end
            return INDEX["use"]
        if usable or front < CONTACT_MU:
            return INDEX["forward_use"]
        if front < FRONT_BLOCKED_MU:
            return INDEX["forward_turn_left"]
        if right > OPENING_MU:
            return INDEX["forward_turn_right"]
        return INDEX["forward"]
