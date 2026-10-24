"""Walk a built map by hand in a visible window (PLAN M2: SPECTATOR walk-through).

    python scripts/walk_map.py build/MAP01.fixed.cfg

Controls are GZDoom defaults: arrow keys or WASD to move, mouse to turn, E or Space to
use (doors, lifts, switches). Close the window or press Esc then quit to stop.
Prints your position and room once a second.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import vizdoom as vzd
import yaml

GV = vzd.GameVariable


def room_names(cfg: Path) -> list[tuple[str, int, int, int, int]]:
    """Room rectangles for the map this cfg loads, read from the spec it was built from."""
    lump = cfg.name.split(".")[0]
    maps = Path(__file__).resolve().parents[1] / "maps"
    for spec_path in maps.glob("*.yaml"):
        spec = yaml.safe_load(spec_path.read_text())
        if f"MAP{spec['slot']:02d}" == lump:
            return [(r["id"], r["x0"], r["y0"], r["x1"], r["y1"]) for r in spec["rooms"]]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=Path)
    args = parser.parse_args()

    rooms = room_names(args.cfg)
    game = vzd.DoomGame()
    game.load_config(str(args.cfg))
    game.set_mode(vzd.Mode.SPECTATOR)
    game.set_window_visible(True)
    game.set_screen_resolution(vzd.ScreenResolution.RES_800X600)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_render_hud(True)
    game.set_render_weapon(True)
    game.set_render_crosshair(True)
    game.set_episode_timeout(0)
    game.init()
    game.new_episode()
    tic = 0
    while not game.is_episode_finished():
        game.advance_action()
        tic += 1
        if tic % 35 == 0:
            x, y, z = (game.get_game_variable(v) for v in (GV.POSITION_X, GV.POSITION_Y,
                                                            GV.POSITION_Z))  # fmt: skip
            here = [r[0] for r in rooms if r[1] <= x <= r[3] and r[2] <= y <= r[4]]
            print(f"t={tic // 35:4d}s  x={x:7.1f} y={y:7.1f} z={z:6.1f}  room={here or '-'}")
    print("episode finished (exit reached or window closed)")
    game.close()


if __name__ == "__main__":
    main()
