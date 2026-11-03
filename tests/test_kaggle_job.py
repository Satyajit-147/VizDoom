"""scripts/kaggle_job.py: the notebook cells it pushes must *run*, not just compile.
Regression: a gate-less job wrote JSON's `null` into a Python cell; it compiled, then
failed on Kaggle with a NameError once the pre-flight had passed."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]

FAKE_TRAIN = """
import argparse, json, pathlib
ap = argparse.ArgumentParser()
for a in ("--config", "--runs-dir", "--results-dir", "--total-steps"):
    ap.add_argument(a)
ap.add_argument("--seed", type=int)
a = ap.parse_args()
d = pathlib.Path(a.runs_dir) / (pathlib.Path(a.config).stem + f"_seed{a.seed}")
d.mkdir(parents=True, exist_ok=True)
v = float(pathlib.Path(a.config).read_text())  # the fake config holds the validity to log
d.joinpath("metrics.jsonl").write_text("".join(json.dumps({"validity": v}) + "\\n" for _ in range(12)))
print("RUN COMPLETE", flush=True)
"""


@pytest.fixture(scope="module")
def kj() -> ModuleType:
    spec = importlib.util.spec_from_file_location("kaggle_job", REPO / "scripts" / "kaggle_job.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("gate", [None, [1, -0.5]])
def test_launch_cell_dry_runs(kj: ModuleType, tmp_path: Path, gate: list[float] | None) -> None:
    src = kj.launch_cell([[("configs/t4_goexplore_map07.yaml", 0)]], [], gate)
    assert "null" not in src
    kj.dry_run(src, tmp_path)


def test_preflight_cell_names_real_tests(kj: ModuleType) -> None:
    for cpu in (True, False):
        src = kj.preflight_cell(cpu)
        compile(src, "preflight", "exec")
        tests = [t for t in src.split("'") if t.startswith("tests/")]
        assert tests and all((REPO / t).exists() for t in tests)


def _launch(kj: ModuleType, tmp: Path, first_wave_validities: tuple[str, str]) -> Path:
    (tmp / "fake_train.py").write_text(FAKE_TRAIN)
    for name, v in (("a", first_wave_validities[0]), ("b", first_wave_validities[1]),
                    ("good", "0.4")):  # fmt: skip
        (tmp / f"{name}.yaml").write_text(v)
    waves = [[(str(tmp / "a.yaml"), 0), (str(tmp / "b.yaml"), 1)], [(str(tmp / "good.yaml"), 5)]]
    src = kj.launch_cell(waves, [], [1, -0.5])
    src = (src.replace("/kaggle/working", str(tmp)).replace('"scripts/train.py"',
           repr(str(tmp / "fake_train.py"))).replace("time.sleep(20)", "time.sleep(0)")
           .replace("time.sleep(30)", "time.sleep(0.2)"))  # fmt: skip
    exec(compile(src, "launch", "exec"), {"__name__": "launch"})  # noqa: S102 - our own cell
    return tmp / "runs" / "good_seed5"


def test_the_gate_lets_progress_through(kj: ModuleType, tmp_path: Path) -> None:
    assert _launch(kj, tmp_path, ("-2.3", "0.4")).exists()


def test_the_gate_stops_a_job_that_makes_no_progress(kj: ModuleType, tmp_path: Path) -> None:
    assert not _launch(kj, tmp_path, ("-2.3", "-2.0")).exists()


def test_summary_cell_compiles(kj: ModuleType) -> None:
    compile(kj.SUMMARY, "summary", "exec")
    assert json.dumps(None) == "null" and repr(None) == "None"  # why launch_cell uses repr
    assert sys.executable
