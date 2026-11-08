"""The Fixer's edit operations on a compiled level (docs/FIXER_PLAN.md section 3, F0).

Bugs are planted in the compiled level, not the MapSpec (D20), so the Fixer edits the
compiled `udmf.TextMap`, the level's source as a Doom map editor sees it. Six generic
operations, none specific to a bug kind:

| op             | changes                                    | e.g. repairs     |
|----------------|--------------------------------------------|------------------|
| SetFlag        | one linedef flag on or off                 | a walk-through wall |
| SetArg         | one argument of a linedef's special        | a switch's target tag, a door's lock |
| CopySpecial    | special, args and activation flags         | a lift with no trigger on one side |
| SetFloor       | a sector's floor height                    | a pit with no way out |
| SetThingType   | a thing's type, within its class           | an over-cap health pickup |
| MoveThing      | a teleport destination's position, angle   | teleporters that loop |

Deleting walls, sectors or things is deliberately not offered: deleting is the easiest
way to make a bug unobservable rather than fixed. Every op is validated, and `apply`
never mutates its input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

from qatester.mapgen import udmf
from qatester.mapgen.wad import map_lumps, write_wad

PLAYER_HEIGHT = 56  # a floor raised closer than this to the ceiling would crush the player
TELEPORT_DEST = udmf.THING_EDNUMS["teleport_dest"]
# Flags a fix may toggle. `twosided` is structure, not behaviour: flipping it breaks the line.
EDITABLE_FLAGS = udmf.LINEDEF_FLAGS - {"twosided"}
ACTIVATION_FLAGS = frozenset({"playeruse", "playercross", "monstercross", "repeatspecial"})
# A thing may change type only within its class (Doom editor numbers).
THING_CLASSES: dict[str, frozenset[int]] = {
    "health": frozenset({2011, 2012, 2013, 2014}),  # stimpack, medikit, soulsphere, bonus
    "armour": frozenset({2015, 2018, 2019}),
    "ammo": frozenset({2007, 2008, 2010, 2047, 2048, 2049}),
    "key": frozenset({5, 6, 13, 38, 39, 40}),
}


class EditError(ValueError):
    """An edit that does not apply to this level."""


@dataclass(frozen=True)
class SetFlag:
    linedef: int
    flag: str
    on: bool


@dataclass(frozen=True)
class SetArg:
    linedef: int
    index: int
    value: int


@dataclass(frozen=True)
class CopySpecial:
    source: int
    target: int


@dataclass(frozen=True)
class SetFloor:
    sector: int
    height: int


@dataclass(frozen=True)
class SetThingType:
    thing: int
    type: int


@dataclass(frozen=True)
class MoveThing:
    thing: int
    x: float
    y: float
    angle: int


Edit = SetFlag | SetArg | CopySpecial | SetFloor | SetThingType | MoveThing


def thing_class(ednum: int) -> str | None:
    return next((c for c, members in THING_CLASSES.items() if ednum in members), None)


def _index(items: list[object], i: int, what: str) -> None:
    if not 0 <= i < len(items):
        raise EditError(f"no {what} {i} (the level has {len(items)})")


def _apply_one(tm: udmf.TextMap, e: Edit) -> None:
    if isinstance(e, SetFlag):
        _index(tm.linedefs, e.linedef, "linedef")  # type: ignore[arg-type]
        if e.flag not in EDITABLE_FLAGS:
            raise EditError(f"flag {e.flag!r} cannot be edited")
        ld = tm.linedefs[e.linedef]
        flags = ld.flags | {e.flag} if e.on else ld.flags - {e.flag}
        tm.linedefs[e.linedef] = replace(ld, flags=frozenset(flags))
    elif isinstance(e, SetArg):
        _index(tm.linedefs, e.linedef, "linedef")  # type: ignore[arg-type]
        ld = tm.linedefs[e.linedef]
        if ld.special == 0:
            raise EditError(f"linedef {e.linedef} has no special, so no arguments")
        if not 0 <= e.index < 5:
            raise EditError("a special has arguments 0-4")
        args = list(ld.args) + [0] * (e.index + 1 - len(ld.args))
        args[e.index] = int(e.value)
        tm.linedefs[e.linedef] = replace(ld, args=tuple(args))
    elif isinstance(e, CopySpecial):
        _index(tm.linedefs, e.source, "linedef")  # type: ignore[arg-type]
        _index(tm.linedefs, e.target, "linedef")  # type: ignore[arg-type]
        src, dst = tm.linedefs[e.source], tm.linedefs[e.target]
        if src.special == 0 or e.source == e.target:
            raise EditError(f"linedef {e.source} has no special to copy")
        flags = (dst.flags - ACTIVATION_FLAGS) | (src.flags & ACTIVATION_FLAGS)
        tm.linedefs[e.target] = replace(dst, special=src.special, args=src.args,
                                        flags=frozenset(flags))  # fmt: skip
    elif isinstance(e, SetFloor):
        _index(tm.sectors, e.sector, "sector")  # type: ignore[arg-type]
        s = tm.sectors[e.sector]
        if e.height > s.heightceiling - PLAYER_HEIGHT:
            raise EditError(f"floor {e.height} leaves no room under ceiling {s.heightceiling}")
        tm.sectors[e.sector] = replace(s, heightfloor=int(e.height))
    elif isinstance(e, SetThingType):
        _index(tm.things, e.thing, "thing")  # type: ignore[arg-type]
        t = tm.things[e.thing]
        cls = thing_class(t.type)
        if cls is None or e.type not in THING_CLASSES[cls]:
            raise EditError(f"thing {e.thing} (type {t.type}) cannot become type {e.type}")
        tm.things[e.thing] = replace(t, type=int(e.type))
    elif isinstance(e, MoveThing):
        _index(tm.things, e.thing, "thing")  # type: ignore[arg-type]
        t = tm.things[e.thing]
        if t.type != TELEPORT_DEST:
            raise EditError(f"thing {e.thing} is not a teleport destination; only those move")
        tm.things[e.thing] = replace(t, x=float(e.x), y=float(e.y), angle=int(e.angle) % 360)
    else:  # pragma: no cover - exhaustive over Edit
        raise EditError(f"unknown edit {e!r}")


def apply(tm: udmf.TextMap, edits: list[Edit]) -> udmf.TextMap:
    """A new TextMap with `edits` applied in order; `tm` is left untouched. Raises
    EditError on the first edit that does not apply."""
    out = udmf.TextMap(list(tm.vertices), list(tm.linedefs), list(tm.sidedefs),
                       list(tm.sectors), list(tm.things))  # fmt: skip
    for e in edits:
        _apply_one(out, e)
    return out


def describe(e: Edit) -> str:
    """One line for prompts and logs."""
    if isinstance(e, SetFlag):
        return f"{'set' if e.on else 'clear'} flag {e.flag} on linedef {e.linedef}"
    if isinstance(e, SetArg):
        return f"set argument {e.index} of linedef {e.linedef} to {e.value}"
    if isinstance(e, CopySpecial):
        return f"copy the special of linedef {e.source} to linedef {e.target}"
    if isinstance(e, SetFloor):
        return f"set the floor of sector {e.sector} to {e.height}"
    if isinstance(e, SetThingType):
        return f"change thing {e.thing} to type {e.type}"
    return f"move thing {e.thing} to ({e.x:.0f}, {e.y:.0f}) facing {e.angle}"


def write_level(tm: udmf.TextMap, map_lump: str, cfg_text: str, out_dir: Path,
                name: str) -> Path:  # fmt: skip
    """Write the edited level as <out_dir>/<name>.wad (one map) and <name>.cfg, a copy of
    the original ViZDoom config pointed at it. Returns the .cfg path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wad = out_dir / f"{name}.wad"
    write_wad(wad, map_lumps(map_lump, tm.to_text()))
    cfg, n = re.subn(r"(?m)^doom_scenario_path\s*=.*$", f"doom_scenario_path = {wad.name}",
                     cfg_text)  # fmt: skip
    if n != 1:
        raise ValueError("the config has no doom_scenario_path line")
    cfg_path = out_dir / f"{name}.cfg"
    cfg_path.write_text(cfg, encoding="ascii")
    return cfg_path
