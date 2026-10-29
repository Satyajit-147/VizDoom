"""What every tester tier shares. Agents see observations only: nothing under
qatester/agents/ may import ground truth (tests/test_no_groundtruth_leak.py)."""

from __future__ import annotations

from typing import Protocol

import numpy as np

Obs = dict[str, np.ndarray]


class Agent(Protocol):
    def reset(self) -> None:
        """A new episode starts."""

    def act(self, obs: Obs) -> int:
        """An action index (qatester.env.actions)."""
