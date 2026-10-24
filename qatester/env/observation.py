"""Observation assembly (PLAN: Observation space).

`depth`: the 160x120 depth buffer, centre-cropped to 120x120, resized to 64x64, stacked 4
deep. `scalars`: 18 egocentric, normalised values. Absolute position is never part of the
observation; it goes to the trace, the oracles and the coverage counts only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

DEPTH_SIZE = 64
STACK = 4
N_SCALARS = 18
WALK_SPEED_MU_PER_TIC = 7.5  # Doom's terminal walking speed (thrust 25, friction 0.906)
KEY_COLOURS = ("key_red", "key_blue", "key_yellow")


def observation_space() -> spaces.Dict:
    return spaces.Dict({
        "depth": spaces.Box(0, 255, (STACK, DEPTH_SIZE, DEPTH_SIZE), np.uint8),
        "scalars": spaces.Box(-1.0, 1.0, (N_SCALARS,), np.float32),
    })  # fmt: skip


def _sample_index(src: int, dst: int) -> np.ndarray:
    return ((np.arange(dst) + 0.5) * src / dst).astype(np.intp)


class DepthProcessor:
    """Crop the centre square of the depth buffer and resize it by nearest sampling."""

    def __init__(self, height: int = 120, width: int = 160, size: int = DEPTH_SIZE) -> None:
        side = min(height, width)
        y0, x0 = (height - side) // 2, (width - side) // 2
        self.rows = (y0 + _sample_index(side, size))[:, None]
        self.cols = (x0 + _sample_index(side, size))[None, :]
        self.shape = (height, width)

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        if depth.shape != self.shape:
            raise ValueError(f"depth buffer {depth.shape}, expected {self.shape}")
        return np.ascontiguousarray(depth[self.rows, self.cols], dtype=np.uint8)


class FrameStack:
    def __init__(self, depth: int = STACK, size: int = DEPTH_SIZE) -> None:
        self.frames = np.zeros((depth, size, size), np.uint8)

    def fill(self, frame: np.ndarray) -> np.ndarray:
        self.frames[:] = frame
        return self.frames.copy()

    def push(self, frame: np.ndarray) -> np.ndarray:
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = frame
        return self.frames.copy()


@dataclass(frozen=True)
class ScalarInputs:
    angle: float  # degrees, 0 = east, counter-clockwise
    vx: float  # world-frame velocity, mu/tic
    vy: float
    vz: float
    on_ground: bool
    moved_mu: float  # displacement over the step
    commanded: bool  # whether the action commanded movement
    tics: int  # tics in the step
    health: float
    armour: float
    ammo: float
    keys: frozenset[str]
    elapsed: float  # fraction of the episode's time limit
    novelty: float  # 1/sqrt(1 + n) for the current cell
    neighbour_novelty: float  # mean of the same over the walkable 8-neighbourhood
    interactive_in_view: int  # pickups in the labels buffer
    use_target: bool  # a usable door, lift or switch within USE reach, in front


def scalars(s: ScalarInputs) -> np.ndarray:
    a = math.radians(s.angle)
    ca, sa = math.cos(a), math.sin(a)
    forward = s.vx * ca + s.vy * sa
    left = -s.vx * sa + s.vy * ca
    speed_ratio = 1.0
    if s.commanded:
        speed_ratio = s.moved_mu / (WALK_SPEED_MU_PER_TIC * max(1, s.tics))
    v = [
        sa, ca,
        forward / 32.0, left / 32.0, s.vz / 32.0,
        float(s.on_ground),
        speed_ratio,
        s.health / 100.0, s.armour / 200.0, s.ammo / 50.0,
        *(float(k in s.keys) for k in KEY_COLOURS),
        s.elapsed,
        s.novelty, s.neighbour_novelty,
        s.interactive_in_view / 8.0,
        float(s.use_target),
    ]  # fmt: skip
    return np.clip(np.asarray(v, np.float32), -1.0, 1.0)
