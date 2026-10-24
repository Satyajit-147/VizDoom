import os

# Must be set before vizdoom is imported anywhere (PLAN: Kaggle operating rules, headless notes).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("OMP_NUM_THREADS", "1")


from pathlib import Path

import pytest

from qatester.mapgen.cli import BuildResult, build
from qatester.mapgen.groundtruth import GroundTruth

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, BuildResult]:
    """Both variants of the corpus, built once per test session."""
    out = tmp_path_factory.mktemp("corpus")
    return {v: build(REPO / "maps", out, v) for v in ("buggy", "fixed")}  # type: ignore[arg-type]


@pytest.fixture(scope="session")
def truth(built: dict[str, BuildResult]) -> dict[tuple[str, str], GroundTruth]:
    """Full ground truth (with defects) by (map, variant)."""
    return {
        (lump, variant): GroundTruth.load(path)
        for variant, result in built.items()
        for lump, path in result.groundtruth_paths.items()
    }
