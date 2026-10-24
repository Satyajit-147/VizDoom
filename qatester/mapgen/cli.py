"""Build MapSpecs into a WAD plus, per map, a ViZDoom config, ground truth JSON and preview
SVG (PLAN: Derived outputs, Build matrix).

Usage: python -m qatester.mapgen.cli build <spec dir or file> -o build/ --variant buggy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from qatester.mapgen.geometry import CompiledMap, compile_map
from qatester.mapgen.groundtruth import ground_truth, write_ground_truth
from qatester.mapgen.preview import render_svg
from qatester.mapgen.spec import MapSpec, load_spec
from qatester.mapgen.validate import validate
from qatester.mapgen.wad import Lump, map_lumps, write_wad

Variant = Literal["buggy", "fixed"]


@dataclass(frozen=True)
class BuildResult:
    wad_path: Path
    cfg_paths: dict[str, Path]  # map lump name -> cfg path
    groundtruth_paths: dict[str, Path]
    preview_paths: dict[str, Path]
    compiled: dict[str, CompiledMap]  # what went into the WAD


def wad_name(variant: Variant) -> str:
    return f"qa_scenarios_{variant}.wad"


def artefact_name(map_lump: str, variant: Variant, suffix: str) -> str:
    """Both variants share build/ and slot numbers (PLAN: Build matrix), so the fixed
    variant's files carry `.fixed`: MAP03.cfg vs MAP03.fixed.cfg."""
    return f"{map_lump}.{suffix}" if variant == "buggy" else f"{map_lump}.fixed.{suffix}"


def cfg_name(map_lump: str, variant: Variant) -> str:
    return artefact_name(map_lump, variant, "cfg")


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


def compile_checked(spec: MapSpec) -> CompiledMap:
    """Compile (injecting the spec's defects) and assert every geometry invariant;
    raises GeometryError on failure."""
    cm = compile_map(spec)
    validate(cm, cm.textmap())
    return cm


def build(spec_dir: Path, out_dir: Path, variant: Variant) -> BuildResult:
    """Writes <out>/qa_scenarios_<variant>.wad and, per map, a .cfg, a .groundtruth.json
    and a .preview.svg. `spec_dir` may also be a single spec file."""
    specs = [load_spec(p) for p in _spec_paths(spec_dir)]
    if not specs:
        raise FileNotFoundError(f"no *.yaml specs under {spec_dir}")
    slots = [s.slot for s in specs]
    if len(set(slots)) != len(slots):
        raise ValueError(f"duplicate map slots: {sorted(slots)}")

    lumps: list[Lump] = []
    cfg_paths: dict[str, Path] = {}
    gt_paths: dict[str, Path] = {}
    preview_paths: dict[str, Path] = {}
    compiled: dict[str, CompiledMap] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in sorted(specs, key=lambda s: s.slot):
        if variant == "fixed":
            spec = spec.without_defects()
        lump = spec.map_lump
        intended = compile_checked(spec.without_defects())
        built = compile_checked(spec) if spec.defects else intended
        compiled[lump] = built
        lumps += map_lumps(lump, built.textmap().to_text())

        cfg_paths[lump] = out_dir / cfg_name(lump, variant)
        cfg_paths[lump].write_text(render_cfg(spec, variant), encoding="ascii")
        gt_paths[lump] = out_dir / artefact_name(lump, variant, "groundtruth.json")
        write_ground_truth(gt_paths[lump], ground_truth(spec, intended, built, variant))
        preview_paths[lump] = out_dir / artefact_name(lump, variant, "preview.svg")
        preview_paths[lump].write_text(render_svg(built), encoding="utf-8")

    wad_path = out_dir / wad_name(variant)
    write_wad(wad_path, lumps)
    return BuildResult(wad_path, cfg_paths, gt_paths, preview_paths, compiled)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m qatester.mapgen.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="compile MapSpecs into a WAD, configs and previews")
    b.add_argument("specs", type=Path, help="spec directory or single .yaml file")
    b.add_argument("-o", "--out", type=Path, required=True)
    b.add_argument("--variant", choices=["buggy", "fixed"], default="buggy")
    args = parser.parse_args(argv)

    result = build(args.specs, args.out, args.variant)
    print(f"wrote {result.wad_path}")
    for lump, cfg in result.cfg_paths.items():
        cm = result.compiled[lump]
        print(
            f"  {lump} {cm.spec.name:9s} rooms={len(cm.spec.rooms):2d} "
            f"sectors={len(cm.sectors):2d} linedefs={len(cm.lines):3d} "
            f"defects={len(cm.defects)}  {cfg}"
        )


if __name__ == "__main__":
    main()
