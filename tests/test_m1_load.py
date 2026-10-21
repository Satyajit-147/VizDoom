"""M1 acceptance: build maps/m01_clean.yaml, ViZDoom loads MAP01 through the generated
.cfg, the agent spawns, POSITION_X is finite, and 500 random steps run without error.

Also checks two things a map can get silently wrong: walls must render from inside
(depth buffer not empty: catches reversed linedef winding) and must block (player stays
inside the room).
"""

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import vizdoom as vzd

from qatester.mapgen.cli import build

REPO = Path(__file__).resolve().parents[1]
GV = vzd.GameVariable
PLAYER_RADIUS = 16


@pytest.fixture(scope="module")
def game(tmp_path_factory: pytest.TempPathFactory) -> Iterator[vzd.DoomGame]:
    out = tmp_path_factory.mktemp("build")
    result = build(REPO / "maps" / "m01_clean.yaml", out, "buggy")
    g = vzd.DoomGame()
    g.load_config(str(result.cfg_paths["MAP01"]))
    g.set_seed(0)
    g.init()
    yield g
    g.close()


def _pos(g: vzd.DoomGame) -> tuple[float, float]:
    return g.get_game_variable(GV.POSITION_X), g.get_game_variable(GV.POSITION_Y)


def test_spawns_at_player_start(game: vzd.DoomGame) -> None:
    game.new_episode()
    x, y = _pos(game)
    assert math.isfinite(x) and math.isfinite(y)
    assert (x, y) == pytest.approx((-160.0, -160.0))


def test_walls_render_from_inside(game: vzd.DoomGame) -> None:
    game.new_episode()
    depth = game.get_state().depth_buffer
    assert depth is not None
    assert depth.max() > 0, "depth buffer empty: linedef winding is probably reversed"


def test_500_random_steps_stay_inside_room(game: vzd.DoomGame) -> None:
    rng = np.random.default_rng(0)
    n_buttons = game.get_available_buttons_size()
    game.new_episode()
    for _ in range(500):
        if game.is_episode_finished():
            game.new_episode()
        game.make_action(rng.integers(0, 2, size=n_buttons).tolist(), 4)
        x, y = _pos(game)
        assert math.isfinite(x) and math.isfinite(y)
        # Room is (-320,-320)-(0,0); the player centre stays one radius from each wall.
        assert -320 + PLAYER_RADIUS - 1 <= x <= 0 - PLAYER_RADIUS + 1
        assert -320 + PLAYER_RADIUS - 1 <= y <= 0 - PLAYER_RADIUS + 1
