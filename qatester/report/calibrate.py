"""Step 4: temperature calibration of oracle confidences (PLAN: Confidence calibration).

One temperature T on the logit of the raw confidence, fitted by NLL against labels
(true defect or false alarm). M9 fits it on MAP02/04/06 and evaluates on MAP03/05/07;
until then T = 1, which leaves confidences unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

EPS = 1e-4  # raw confidences of exactly 0 or 1 have infinite logits


def _logit(p: np.ndarray) -> np.ndarray:
    q: np.ndarray = np.clip(p, EPS, 1 - EPS)
    out: np.ndarray = np.log(q / (1 - q))
    return out


def apply_temperature(conf: Sequence[float] | np.ndarray, t: float) -> np.ndarray:
    if t == 1.0:
        return np.asarray(conf, float)
    return 1.0 / (1.0 + np.exp(-_logit(np.asarray(conf, float)) / t))


def fit_temperature(conf: Sequence[float], labels: Sequence[int]) -> float:
    z = _logit(np.asarray(conf, float))
    y = np.asarray(labels, float)

    def nll(log_t: float) -> float:
        p = np.clip(1.0 / (1.0 + np.exp(-z / np.exp(log_t))), 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    res = minimize_scalar(nll, bounds=(-5.0, 5.0), method="bounded")
    return float(np.exp(res.x))


def ece(conf: Sequence[float], labels: Sequence[int], bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    c, y = np.asarray(conf, float), np.asarray(labels, float)
    if len(c) == 0:
        return 0.0
    idx = np.minimum((c * bins).astype(int), bins - 1)
    return float(sum(abs(c[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum()
                     for b in range(bins) if (idx == b).any()) / len(c))  # fmt: skip


def load_temperature(path: Path | None) -> float:
    if path is None:
        return 1.0
    return float(json.loads(Path(path).read_text(encoding="utf-8"))["temperature"])
