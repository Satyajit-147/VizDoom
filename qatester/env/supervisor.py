"""Crash and hang supervision for one ViZDoom instance (PLAN: Robustness and throughput).

- Every engine call goes through `call()`. An engine exception is returned as an error
  string rather than raised; the env records it on the trace (Family 1's `crash`) and
  then calls `restart()`.
- A watchdog thread kills the engine's child process when a call runs past
  `step_timeout_s`; the blocked call then fails and is reported as a hang (Family 1's
  `hang`, from the step's wall time), not as a crash.
- `start()` / `restart()` retry init up to `retries` times with exponential backoff.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def engine_errors() -> tuple[type[BaseException], ...]:
    import vizdoom as vzd

    names = ("ViZDoomErrorException", "ViZDoomIsNotRunningException",
             "ViZDoomUnexpectedExitException", "SharedMemoryException",
             "MessageQueueException")  # fmt: skip
    return tuple(getattr(vzd, n) for n in names if hasattr(vzd, n))


def exit_on_sigterm() -> None:
    """Turn SIGTERM into a normal exit, so `finally` blocks and atexit handlers run and
    the ViZDoom engines (separate processes that outlive a killed parent) are closed."""
    def handler(*_: object) -> None:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)  # a second SIGTERM just ends it
        sys.exit(143)

    signal.signal(signal.SIGTERM, handler)


def child_pids(pid: int | None = None) -> set[int]:
    """Direct children of a process, from /proc (Linux)."""
    pid = os.getpid() if pid is None else pid
    out: set[int] = set()
    for task in Path(f"/proc/{pid}/task").glob("*"):
        try:
            out |= {int(c) for c in (task / "children").read_text().split()}
        except OSError:
            continue
    return out


@dataclass(frozen=True)
class CallResult:
    value: Any
    error: str | None  # engine exception, if any
    hung: bool  # the watchdog killed the engine during this call
    wall_s: float


class Supervisor:
    def __init__(
        self,
        factory: Callable[[], Any],
        *,
        step_timeout_s: float = 30.0,
        retries: int = 5,
        backoff_s: float = 2.0,
        errors: tuple[type[BaseException], ...] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.factory = factory
        self.step_timeout_s = step_timeout_s
        self.retries = retries
        self.backoff_s = backoff_s
        self.errors = errors if errors is not None else engine_errors()
        self.sleep = sleep
        self.game: Any = None
        self.engine_pids: set[int] = set()
        self.restarts = 0
        self._deadline: float | None = None
        self._hung = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._watchdog: threading.Thread | None = None

    # --- lifecycle --------------------------------------------------------------------
    def start(self) -> None:
        delay = self.backoff_s
        for attempt in range(self.retries):
            before = child_pids()
            try:
                self.game = self.factory()
                self.engine_pids = child_pids() - before
                break
            except self.errors:
                if attempt == self.retries - 1:
                    raise
                self.sleep(delay)
                delay *= 2
        if self._watchdog is None:
            atexit.register(self.kill_engine)  # a dying worker must not orphan its engine
            self._watchdog = threading.Thread(target=self._watch, daemon=True,
                                              name="vizdoom-watchdog")  # fmt: skip
            self._watchdog.start()

    def restart(self) -> None:
        self._close_game()
        self.restarts += 1
        self.start()

    def close(self) -> None:
        self._stop.set()
        self._close_game()

    def _close_game(self) -> None:
        if self.game is not None:
            with contextlib.suppress(Exception):  # the engine may already be gone
                self.game.close()
        self.game = None

    # --- calls ------------------------------------------------------------------------
    def call(self, fn: Callable[..., T], *args: Any) -> CallResult:
        with self._lock:
            self._hung = False
            self._deadline = time.monotonic() + self.step_timeout_s
        t0 = time.perf_counter()
        value: Any = None
        error: str | None = None
        try:
            value = fn(*args)
        except self.errors as e:
            error = f"{type(e).__name__}: {e}"
        wall = time.perf_counter() - t0
        with self._lock:
            self._deadline = None
            hung = self._hung
        return CallResult(value, None if hung else error, hung, wall)

    def _watch(self) -> None:
        tick = min(1.0, self.step_timeout_s / 4)
        while not self._stop.wait(tick):
            with self._lock:
                late = self._deadline is not None and time.monotonic() > self._deadline
                if late:
                    self._hung = True
                    self._deadline = None
            if late:
                self.kill_engine()

    def kill_engine(self) -> None:
        for pid in self.engine_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
