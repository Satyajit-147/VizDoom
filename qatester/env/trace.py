"""Episode traces as Parquet, and file names for .lmp demos (PLAN: Determinism and
recording). One Parquet file per episode, one row per StepRecord; the run's metadata
(map, seed, episode, demo) goes in the file's schema metadata."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from qatester.oracles.base import StepRecord

FIELDS = tuple(f.name for f in fields(StepRecord))
SET_FIELDS = ("keys", "move_dirs")
TUPLE_FIELDS = ("sector_floors", "sector_ceilings", "notifications")


def _column(records: Sequence[StepRecord], name: str) -> list[Any]:
    vals = [getattr(r, name) for r in records]
    if name in SET_FIELDS:
        return [sorted(v) for v in vals]
    if name in TUPLE_FIELDS:
        return [list(v) for v in vals]
    return vals


SCHEMA = pa.schema([
    ("tic", pa.int32()), ("x", pa.float32()), ("y", pa.float32()), ("z", pa.float32()),
    ("vx", pa.float32()), ("vy", pa.float32()), ("vz", pa.float32()),
    ("angle", pa.float32()), ("on_ground", pa.bool_()), ("dead", pa.bool_()),
    ("health", pa.float32()), ("armour", pa.float32()), ("ammo", pa.float32()),
    ("keys", pa.list_(pa.string())), ("killcount", pa.int32()), ("action", pa.int8()),
    ("move_dirs", pa.list_(pa.string())), ("use", pa.bool_()), ("jump", pa.bool_()),
    ("sector_floors", pa.list_(pa.float32())), ("sector_ceilings", pa.list_(pa.float32())),
    ("notifications", pa.list_(pa.string())), ("done", pa.bool_()),
    ("truncated", pa.bool_()), ("error", pa.string()), ("step_wall_s", pa.float32()),
])  # fmt: skip
assert tuple(SCHEMA.names) == FIELDS, "trace schema out of step with StepRecord"


def write_trace(path: Path, records: Sequence[StepRecord], meta: dict[str, Any]) -> None:
    table = pa.table({n: _column(records, n) for n in FIELDS}, schema=SCHEMA)
    table = table.replace_schema_metadata({"qatester": json.dumps(meta)})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def read_trace(path: Path) -> tuple[list[StepRecord], dict[str, Any]]:
    table = pq.read_table(path)
    meta = json.loads((table.schema.metadata or {}).get(b"qatester", b"{}"))
    cols = table.to_pydict()
    out = []
    for i in range(table.num_rows):
        row: dict[str, Any] = {n: cols[n][i] for n in FIELDS}
        for n in SET_FIELDS:
            row[n] = frozenset(row[n])
        for n in TUPLE_FIELDS:
            row[n] = tuple(row[n])
        out.append(StepRecord(**row))
    return out, meta


def episode_stem(map_name: str, seed: int, episode: int) -> str:
    return f"{map_name}_s{seed}_ep{episode:06d}"
