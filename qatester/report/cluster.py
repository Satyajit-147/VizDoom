"""Step 1-2 of the report pipeline (PLAN: From findings to a usable bug report).

Raw findings of one oracle family on one map are clustered with DBSCAN on (x, y, z/4),
eps = 96 mu, min_samples = 3. Invariant findings cluster on the invariant's name instead.

Deviation (docs/decisions.md D33): DBSCAN calls a point with fewer than 3 neighbours
noise. Dropping noise would hide a bug seen once or twice, such as a crash, so noise
points are grouped among themselves within eps and kept as small clusters. Their low
cluster size already lowers their severity through the frequency factor.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

RawFinding = dict[str, Any]  # a serialised Finding plus map / env_steps / env_seed


@dataclass
class Cluster:
    oracle: str
    key: str  # the invariant's name for invariants, else "spatial"
    members: list[RawFinding]

    @property
    def representative(self) -> RawFinding:
        """Highest confidence; ties go to the shortest reproduction."""
        return min(self.members, key=lambda f: (-f["confidence"], f["tic_range"][1],
                                                f["env_steps"]))  # fmt: skip

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def first_seen_step(self) -> int:
        return min(int(f["env_steps"]) for f in self.members)


def _features(fs: Sequence[RawFinding]) -> np.ndarray:
    return np.array([[f["position"][0], f["position"][1], f["position"][2] / 4.0] for f in fs])


def _spatial(fs: list[RawFinding], eps: float, min_samples: int) -> list[list[RawFinding]]:
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(_features(fs)).labels_
    groups: dict[int, list[RawFinding]] = defaultdict(list)
    for f, lab in zip(fs, labels, strict=True):
        if lab >= 0:
            groups[int(lab)].append(f)
    noise = [f for f, lab in zip(fs, labels, strict=True) if lab < 0]
    out = list(groups.values())
    if noise:
        nl = DBSCAN(eps=eps, min_samples=1).fit(_features(noise)).labels_
        small: dict[int, list[RawFinding]] = defaultdict(list)
        for f, lab in zip(noise, nl, strict=True):
            small[int(lab)].append(f)
        out += list(small.values())
    return out


def cluster_findings(findings: Sequence[RawFinding], eps: float = 96.0,
                     min_samples: int = 3) -> list[Cluster]:  # fmt: skip
    """Findings of one map -> clusters, each a candidate defect."""
    by_family: dict[str, list[RawFinding]] = defaultdict(list)
    for f in findings:
        by_family[f["oracle"]].append(f)
    clusters: list[Cluster] = []
    for oracle, fs in sorted(by_family.items()):
        if oracle == "invariant":
            by_name: dict[str, list[RawFinding]] = defaultdict(list)
            for f in fs:
                by_name[f["kind"]].append(f)
            clusters += [Cluster(oracle, name, m) for name, m in sorted(by_name.items())]
        else:
            clusters += [Cluster(oracle, "spatial", m) for m in _spatial(fs, eps, min_samples)]
    return clusters
