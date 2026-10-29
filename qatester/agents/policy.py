"""The two-branch encoder of the plan's network (PLAN: Network architecture), as a
Stable-Baselines3 features extractor for sb3-contrib's MultiInputLstmPolicy.

    depth (4,64,64) -> Conv(4,32,8,s4) -> Conv(32,64,4,s2) -> Conv(64,64,3,s1)
                    -> Linear(1024, 256)                      (ReLU throughout)
    scalars (18,)   -> Linear(18, 64) -> Linear(64, 64)
    concat -> 320 features -> one shared LSTM(256) -> policy head (14) + value head (1)

SB3 scales the uint8 depth to [0, 1] itself (it is an image space). The plan's GRU and
second value head are not available in RecurrentPPO: an LSTM and one value head are
used instead (docs/decisions.md, open issue 1, decided as option A).
"""

from __future__ import annotations

from typing import Any

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class QAFeatures(BaseFeaturesExtractor):  # type: ignore[misc]
    def __init__(self, observation_space: spaces.Dict, depth_dim: int = 256,
                 scalar_dim: int = 64) -> None:  # fmt: skip
        super().__init__(observation_space, features_dim=depth_dim + scalar_dim)
        depth = observation_space["depth"]
        scal = observation_space["scalars"]
        assert depth.shape is not None and scal.shape is not None
        self.cnn = nn.Sequential(
            nn.Conv2d(depth.shape[0], 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )  # fmt: skip
        with th.no_grad():
            n = int(self.cnn(th.zeros(1, *depth.shape)).shape[1])
        self.depth_fc = nn.Sequential(nn.Linear(n, depth_dim), nn.ReLU())
        self.scalar_mlp = nn.Sequential(nn.Linear(scal.shape[0], 64), nn.ReLU(),
                                        nn.Linear(64, scalar_dim), nn.ReLU())  # fmt: skip

    def forward(self, obs: dict[str, th.Tensor]) -> th.Tensor:
        return th.cat([self.depth_fc(self.cnn(obs["depth"])), self.scalar_mlp(obs["scalars"])],
                      dim=1)  # fmt: skip


def policy_kwargs(lstm_hidden_size: int = 256) -> dict[str, Any]:
    return {
        "features_extractor_class": QAFeatures,
        "lstm_hidden_size": lstm_hidden_size,
        "n_lstm_layers": 1,
        "shared_lstm": True,  # one recurrent trunk for actor and critic, as in the plan
        "enable_critic_lstm": False,
        "net_arch": {"pi": [], "vf": []},  # heads straight off the LSTM
        "ortho_init": True,  # gain sqrt(2) hidden, 0.01 policy head, 1 value head
    }
