"""M0 gate: ViZDoom runs a stock scenario headless at > 300 steps/s.

Uses the plan's normative throughput settings (160x120, GRAY8, no window, no
sound, frame skip 4) on top of the stock basic.cfg.
"""

import os
import time

import numpy as np
import vizdoom as vzd

N_STEPS = 1000
FRAME_SKIP = 4
MIN_STEPS_PER_S = 300.0


def _make_game() -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_console_enabled(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_seed(0)
    game.init()
    return game


def test_random_steps_headless_throughput() -> None:
    rng = np.random.default_rng(0)
    game = _make_game()
    try:
        n_buttons = game.get_available_buttons_size()
        game.new_episode()
        start = time.perf_counter()
        for _ in range(N_STEPS):
            if game.is_episode_finished():
                game.new_episode()
            action = rng.integers(0, 2, size=n_buttons).tolist()
            game.make_action(action, FRAME_SKIP)
        elapsed = time.perf_counter() - start
    finally:
        game.close()

    steps_per_s = N_STEPS / elapsed
    print(f"vizdoom {vzd.__version__}: {steps_per_s:.0f} steps/s (frame skip {FRAME_SKIP})")
    assert steps_per_s > MIN_STEPS_PER_S


def test_notifications_buffer_available() -> None:
    # PLAN R9: feature-detect; the invariant oracle degrades gracefully without it.
    game = vzd.DoomGame()
    assert hasattr(game, "set_notifications_buffer_enabled"), (
        f"vizdoom {vzd.__version__} lacks the notifications buffer; use the R9 fallback"
    )
