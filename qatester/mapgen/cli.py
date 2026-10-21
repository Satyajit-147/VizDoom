"""Build MapSpecs into a WAD plus ViZDoom configs (PLAN: Derived outputs).

M1 scope: single-room maps only. Multi-room geometry, portals, ground truth and
preview SVGs are M2/M3; they raise NotImplementedError here until then.

Usage: python -m qatester.mapgen.cli build <spec dir or file> -o build/ --variant buggy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from qatester.mapgen import udmf
from qatester.mapgen.spec import MapSpec, Room, load_spec
from qatester.mapgen.wad import Lump, map_lumps, write_wad

Variant = Literal["buggy", "fixed"]
WALL_TEXTURE = "BRICK7"


@dataclass(frozen=True)
class BuildResult:
    wad_path: Path
    cfg_paths: dict[str, Path]  # map lump name -> cfg path


def wad_name(variant: Variant) -> str:
    return f"qa_scenarios_{variant}.wad"


def cfg_name(map_lump: str, variant: Variant) -> str:
    # Both variants share build/; slot numbers are identical across them (PLAN: Build matrix).
    return f"{map_lump}.cfg" if variant == "buggy" else f"{map_lump}.fixed.cfg"


def room_vertices(room: Room) -> list[tuple[int, int]]:
    """Corners in clockwise order, so each linedef's front (right) side faces into the
    room. Appendix A's counter-clockwise example renders invisible walls; see
    docs/decisions.md D10."""
    return [(room.x0, room.y0), (room.x0, room.y1), (room.x1, room.y1), (room.x1, room.y0)]


def compile_textmap(spec: MapSpec) -> udmf.TextMap:
    if len(spec.rooms) != 1 or spec.portals:
        raise NotImplementedError("multi-room maps and portals arrive in M2")
    room = spec.rooms[0]
    tm = udmf.TextMap()
    tm.vertices = [udmf.Vertex(x, y) for x, y in room_vertices(room)]
    for i in range(4):
        tm.linedefs.append(udmf.Linedef(i, (i + 1) % 4, i, flags=frozenset({"blocking"})))
        tm.sidedefs.append(udmf.Sidedef(0, texturemiddle=WALL_TEXTURE))
    tm.sectors = [
        udmf.Sector(room.floor, room.ceiling, lightlevel=room.light, id=room.tag or 0)
    ]
    for thing in spec.things:
        if thing.type == "exit_switch":
            raise NotImplementedError("exit switches are linedefs; they arrive in M2")
        tm.things.append(udmf.Thing(thing.pos[0], thing.pos[1], udmf.THING_EDNUMS[thing.type]))
    return tm


def render_cfg(spec: MapSpec, variant: Variant) -> str:
    """ViZDoom config (PLAN Appendix B). QADoomEnv re-applies these settings in code."""
    return f"""\
# Generated from the {spec.name} MapSpec. Do not hand-edit.
doom_scenario_path = {wad_name(variant)}
doom_map = {spec.map_lump.lower()}
doom_skill = 3

mode = PLAYER
screen_resolution = RES_160X120
screen_format = GRAY8
window_visible = false
sound_enabled = false
console_enabled = false

depth_buffer_enabled = true
labels_buffer_enabled = true
objects_info_enabled = true
sectors_info_enabled = true
automap_buffer_enabled = false

render_hud = false
render_minimal_hud = false
render_weapon = false
render_crosshair = false
render_decals = false
render_particles = false
render_effects_sprites = false
render_messages = false
render_corpses = false
render_screen_flashes = false

episode_start_time = 14
episode_timeout = {spec.goal.time_limit_tics}
living_reward = 0

available_buttons = {{
    MOVE_FORWARD MOVE_BACKWARD MOVE_LEFT MOVE_RIGHT
    TURN_LEFT TURN_RIGHT USE ATTACK JUMP
}}

available_game_variables = {{
    POSITION_X POSITION_Y POSITION_Z ANGLE
    VELOCITY_X VELOCITY_Y VELOCITY_Z
    HEALTH ARMOR AMMO2 SELECTED_WEAPON
    KILLCOUNT ITEMCOUNT SECRETCOUNT
    DEAD ON_GROUND ATTACK_READY
    USER1 USER2 USER3 USER4
}}
"""


def _spec_paths(spec_path: Path) -> list[Path]:
    if spec_path.is_dir():
        return sorted(spec_path.glob("*.yaml"))
    return [spec_path]


def build(spec_dir: Path, out_dir: Path, variant: Variant) -> BuildResult:
    """Writes <out>/qa_scenarios_<variant>.wad and one .cfg per map.

    `spec_dir` may also be a single spec file. Ground truth and preview SVGs are M2/M3.
    """
    specs = [load_spec(p) for p in _spec_paths(spec_dir)]
    if not specs:
        raise FileNotFoundError(f"no *.yaml specs under {spec_dir}")
    slots = [s.slot for s in specs]
    if len(set(slots)) != len(slots):
        raise ValueError(f"duplicate map slots: {sorted(slots)}")

    lumps: list[Lump] = []
    cfg_paths: dict[str, Path] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in sorted(specs, key=lambda s: s.slot):
        if variant == "fixed":
            spec = spec.without_defects()
        elif spec.defects:
            raise NotImplementedError("defect injection arrives in M3")
        lumps += map_lumps(spec.map_lump, compile_textmap(spec).to_text())
        cfg_path = out_dir / cfg_name(spec.map_lump, variant)
        cfg_path.write_text(render_cfg(spec, variant), encoding="ascii")
        cfg_paths[spec.map_lump] = cfg_path

    wad_path = out_dir / wad_name(variant)
    write_wad(wad_path, lumps)
    return BuildResult(wad_path, cfg_paths)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m qatester.mapgen.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="compile MapSpecs into a WAD and configs")
    b.add_argument("specs", type=Path, help="spec directory or single .yaml file")
    b.add_argument("-o", "--out", type=Path, required=True)
    b.add_argument("--variant", choices=["buggy", "fixed"], default="buggy")
    args = parser.parse_args(argv)

    result = build(args.specs, args.out, args.variant)
    print(f"wrote {result.wad_path}")
    for lump, cfg in result.cfg_paths.items():
        print(f"  {lump}: {cfg}")


if __name__ == "__main__":
    main()
