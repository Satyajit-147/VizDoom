"""Watch a recorded demo (.lmp) from a bug report, in a window.

    python scripts/replay.py results/<run>/MAP03/demos/MAP03-oob-01.lmp \\
        --wad results/<run>/MAP03/qa_scenarios_buggy.wad --map MAP03

Paths given relative to the report folder also work when run from inside it. Close the
window or press Ctrl+C to stop. `--headless` replays without a window and prints where
the player ended, which is how the tests check that a demo works.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import vizdoom as vzd

GV = vzd.GameVariable


def replay(demo: Path, wad: Path, doom_map: str, headless: bool = False) -> tuple[float, float, int]:
    g = vzd.DoomGame()
    g.set_doom_scenario_path(str(wad))
    g.set_doom_map(doom_map.lower())
    g.set_doom_skill(3)
    g.set_episode_start_time(14)
    g.set_window_visible(not headless)
    g.set_sound_enabled(False)
    g.set_screen_resolution(vzd.ScreenResolution.RES_160X120 if headless
                            else vzd.ScreenResolution.RES_640X480)
    g.set_render_hud(not headless)
    g.set_mode(vzd.Mode.PLAYER)
    g.init()
    short = Path(tempfile.mkdtemp(prefix="qa"))  # ViZDoom hangs on long demo paths
    try:
        shutil.copyfile(demo, short / "d.lmp")
        g.replay_episode(str(short / "d.lmp"))
        tics = 0
        while not g.is_episode_finished():
            g.advance_action()
            tics += 1
        return g.get_game_variable(GV.POSITION_X), g.get_game_variable(GV.POSITION_Y), tics
    finally:
        g.close()
        shutil.rmtree(short, ignore_errors=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("demo", type=Path)
    ap.add_argument("--wad", type=Path, required=True)
    ap.add_argument("--map", required=True, help="map lump, e.g. MAP03")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args(argv)
    for p in (args.demo, args.wad):
        if not p.exists():
            sys.exit(f"not found: {p}")
    x, y, tics = replay(args.demo, args.wad, args.map, args.headless)
    print(f"replayed {tics} tics; the player ended at ({x:.0f}, {y:.0f})")


if __name__ == "__main__":
    main()
