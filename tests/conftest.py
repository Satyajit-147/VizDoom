import os

# Must be set before vizdoom is imported anywhere (PLAN: Kaggle operating rules, headless notes).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("OMP_NUM_THREADS", "1")
