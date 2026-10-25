"""QADoomEnv: ViZDoom as a Gymnasium environment for the QA tester (PLAN: ViZDoom
configuration and the environment wrapper; Observation, action and reward design).

One env = one ViZDoom instance on one map. Each step:
  1. the action's buttons are held for `frame_skip` tics (through the supervisor);
  2. a StepRecord is read from the engine (position, stats, live sector heights, keys);
  3. every oracle observes it, live; findings go to `info["findings"]` and the queue;
  4. the composite reward is computed (qatester.env.reward);
  5. the observation is assembled (qatester.env.observation).

The env sees only the intended level (`GroundTruth.intent_only()`), never the defects.
Families 4 and 6 judge across episodes; their findings come from `finalize_oracles()`,
which `close()` also calls.
"""

from __future__ import annotations

import multiprocessing as mp
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import vizdoom as vzd

from qatester.coverage import CoverageTracker
from qatester.env.actions import ACTIONS, BUTTONS, N_ACTIONS
from qatester.env.observation import (
    DepthProcessor,
    FrameStack,
    ScalarInputs,
    observation_space,
    scalars,
)
from qatester.env.reward import ADV_PARTS, VALID_PARTS, EpisodeReward, RewardConfig
from qatester.env.supervisor import CallResult, Supervisor
from qatester.env.trace import episode_stem, write_trace
from qatester.env.world import World, count_pickups_in_view, keys_mask
from qatester.mapgen.groundtruth import GroundTruth
from qatester.oracles.base import EpisodeContext, Finding, OracleConfig, StepRecord
from qatester.oracles.level import TraversalTracker
from qatester.oracles.registry import make_oracles

GV = vzd.GameVariable
GAME_VARIABLES = (
    GV.POSITION_X, GV.POSITION_Y, GV.POSITION_Z, GV.ANGLE,
    GV.VELOCITY_X, GV.VELOCITY_Y, GV.VELOCITY_Z,
    GV.HEALTH, GV.ARMOR, GV.AMMO2, GV.SELECTED_WEAPON, GV.KILLCOUNT, GV.ITEMCOUNT,
    GV.SECRETCOUNT, GV.DEAD, GV.ON_GROUND, GV.ATTACK_READY,
    GV.USER1, GV.USER2, GV.USER3, GV.USER4,
)  # fmt: skip
_I = {v: i for i, v in enumerate(GAME_VARIABLES)}
# Only the warp test hook turns with this; no action ever sets it.
TURN_DELTA = vzd.Button.TURN_LEFT_RIGHT_DELTA
TracePolicy = Literal["all", "findings", "none"]
Obs = dict[str, np.ndarray]


def _cfg_value(cfg_path: Path, key: str) -> str:
    for line in Path(cfg_path).read_text(encoding="ascii").splitlines():
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip()
    raise KeyError(f"{cfg_path}: no {key}")


def wad_of(cfg_path: Path) -> Path:
    """The WAD a ViZDoom .cfg loads (relative paths resolve against the .cfg's folder)."""
    return (Path(cfg_path).parent / _cfg_value(cfg_path, "doom_scenario_path")).resolve()


def doom_map_of(cfg_path: Path) -> str:
    return _cfg_value(cfg_path, "doom_map")


class QADoomEnv(gym.Env[Obs, np.int64]):
    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012 - as gymnasium.Env

    def __init__(
        self,
        cfg_path: Path,
        groundtruth: GroundTruth,
        coverage: CoverageTracker | None = None,
        oracle_queue: mp.Queue[Any] | None = None,
        seed: int = 0,
        record_dir: Path | None = None,
        *,
        reward_cfg: RewardConfig | None = None,
        oracle_cfg: OracleConfig | None = None,
        frame_skip: int = 4,
        trace_policy: TracePolicy = "findings",
        cheats: bool = False,
        init_retries: int = 5,
        init_backoff_s: float = 2.0,
        map_name: str | None = None,
    ) -> None:
        super().__init__()
        self.queue = oracle_queue
        self.base_seed = seed
        self.record_dir = Path(record_dir) if record_dir is not None else None
        self.reward_cfg = reward_cfg or RewardConfig()
        self.oracle_cfg = oracle_cfg or OracleConfig()
        self.frame_skip = frame_skip
        self.trace_policy: TracePolicy = trace_policy
        self.cheats = cheats

        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = observation_space()  # type: ignore[assignment]
        self.depth = DepthProcessor()
        self.frames = FrameStack()
        self.reward = EpisodeReward(self.reward_cfg)
        self._configure_map(Path(cfg_path), groundtruth, coverage, map_name)
        self.supervisor = Supervisor(
            self._make_game, step_timeout_s=self.oracle_cfg.crash.step_timeout_s,
            retries=init_retries, backoff_s=init_backoff_s,
        )  # fmt: skip
        self.supervisor.start()

        self.episode = -1
        self.total_steps = 0  # env steps over this env's lifetime
        self.game_seed = 0
        self.records: list[StepRecord] = []
        self.episode_findings: list[Finding] = []
        self.demo_path: Path | None = None
        self._demo_tmp: Path | None = None
        self._pending_demos: list[tuple[Path, Path]] = []  # (recording, destination)
        self.in_episode = False
        self.closed = False
        self._reset_episode_state()

    def _configure_map(self, cfg_path: Path, groundtruth: GroundTruth,
                       coverage: CoverageTracker | None, name: str | None) -> None:  # fmt: skip
        """Per-map state: the level's intent, its oracles, its coverage counts."""
        truth = groundtruth.intent_only() if "defects" in groundtruth.data else groundtruth
        world = World(truth)
        if world.goal_kind != "reach_exit":
            raise ValueError(f"goal kind {world.goal_kind!r} is not supported yet")
        self.cfg_path = cfg_path
        self.truth = truth
        self.map_name: str = name or truth.data["map"]
        self.time_limit: int = int(truth.data["goal"]["time_limit_tics"])
        self.world = world
        self.coverage = coverage
        self.oracles = make_oracles(self.oracle_cfg)
        self.traversal = TraversalTracker(world.level)

    def set_map(self, cfg_path: Path, groundtruth: GroundTruth,
                coverage: CoverageTracker | None = None, name: str | None = None) -> None:
        """Switch to another map before the next reset (the training curriculum).

        Findings judged across episodes are finalized for the old map first. A map in the
        same WAD is a 5 ms `set_doom_map`; another WAD restarts the engine."""
        cfg_path = Path(cfg_path)
        if cfg_path == self.cfg_path:
            return
        self.finalize_oracles()
        same_wad = wad_of(cfg_path) == wad_of(self.cfg_path)
        self._configure_map(cfg_path, groundtruth, coverage, name)
        if same_wad:
            self.game.set_doom_map(doom_map_of(cfg_path))
            self.game.set_episode_timeout(self.time_limit)
        else:
            self.supervisor.restart()

    # --- engine -----------------------------------------------------------------------
    def _make_game(self) -> vzd.DoomGame:
        """The plan's normative settings, applied in code over the generated .cfg."""
        g = vzd.DoomGame()
        g.load_config(str(self.cfg_path))
        g.set_mode(vzd.Mode.PLAYER)
        g.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        g.set_screen_format(vzd.ScreenFormat.GRAY8)
        g.set_depth_buffer_enabled(True)
        g.set_labels_buffer_enabled(True)
        g.set_objects_info_enabled(True)
        g.set_sectors_info_enabled(True)
        g.set_automap_buffer_enabled(False)
        g.set_notifications_buffer_enabled(True)
        g.set_notifications_buffer_size(4)
        g.set_render_hud(False)
        g.set_render_minimal_hud(False)
        g.set_render_weapon(False)
        g.set_render_crosshair(False)
        g.set_render_decals(False)
        g.set_render_particles(False)
        g.set_render_effects_sprites(False)
        g.set_render_messages(False)
        g.set_render_corpses(False)
        g.set_render_screen_flashes(False)
        g.set_window_visible(False)
        g.set_sound_enabled(False)
        g.set_console_enabled(False)
        g.set_doom_skill(3)
        g.set_episode_timeout(self.time_limit)
        g.set_episode_start_time(14)
        g.set_living_reward(0)
        g.set_seed(self.base_seed)
        g.set_available_buttons([getattr(vzd.Button, b) for b in BUTTONS] + [TURN_DELTA])
        g.set_available_game_variables(list(GAME_VARIABLES))
        if self.cheats:
            g.add_game_args("+sv_cheats 1")
        g.init()
        return g

    @property
    def game(self) -> vzd.DoomGame:
        game: vzd.DoomGame = self.supervisor.game
        return game

    def _buttons(self, action: int, turn_delta: float = 0.0) -> list[float]:
        return [*ACTIONS[action].buttons, turn_delta]

    # --- episodes ---------------------------------------------------------------------
    def _reset_episode_state(self) -> None:
        self.reward.reset()
        self.traversal.reset()
        self.cell: tuple[int, int] | None = None
        self.abstract: tuple[int, int, int, int] | None = None
        self.taken: set[str] = set()
        self.keys: frozenset[str] = frozenset()
        self.notifications = ""
        self.last: StepRecord | None = None
        self.last_frame = np.zeros((64, 64), np.uint8)
        self.loaded = False
        self.ep_parts: dict[str, float] = {}
        self.ep_return = 0.0
        self.ep_steps = 0

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[Obs, dict[str, Any]]:  # fmt: skip
        if seed is None and self._np_random is None:
            seed = self.base_seed
        super().reset(seed=seed)
        options = options or {}
        if self.in_episode:
            self._end_episode()
        self.episode += 1
        self.game_seed = int(self.np_random.integers(2**31 - 1))
        self.game.set_seed(self.game_seed)
        self.demo_path = None
        rec: list[str] = []
        if self.record_dir is not None and "load_state" not in options:
            self.demo_path = self.record_dir / "demos" / f"{self._stem()}.lmp"
            self.demo_path.parent.mkdir(parents=True, exist_ok=True)
            # ViZDoom hangs on a recording path of ~113+ characters, so record to a short
            # temporary path and move the file once the engine has closed it (D33).
            tmp = self._short() / f"e{self.episode}.lmp"
            self._pending_demos.append((tmp, self.demo_path))
            rec = [str(tmp)]
        res = self.supervisor.call(self.game.new_episode, *rec)
        if res.error is not None or res.hung:
            self.supervisor.restart()  # a failed new_episode is retried once, on a fresh engine
            self.game.set_seed(self.game_seed)
            self.game.new_episode(*rec)
        self._flush_demos(keep_last=bool(rec))
        if "warp" in options:
            self._warp(*options["warp"])
        if "load_state" in options:
            tmp = self._short() / "l.sav"
            shutil.copyfile(options["load_state"], tmp)
            self.game.load(str(tmp))
        return self._begin_episode(loaded="load_state" in options)

    def _warp(self, x: float, y: float, angle: float | None = None) -> None:
        """Test hook: cheat-teleport before the episode's first record."""
        if not self.cheats:
            raise RuntimeError("the warp option needs QADoomEnv(cheats=True)")
        self.game.send_game_command(f"warp {x} {y}")
        self.game.make_action(self._buttons(0), 2)
        if angle is not None:
            current = self.game.get_game_variable(GV.ANGLE)
            delta = (current - angle + 180.0) % 360.0 - 180.0  # positive turns clockwise
            self.game.make_action(self._buttons(0, delta), 1)

    def _begin_episode(self, loaded: bool) -> tuple[Obs, dict[str, Any]]:
        self._reset_episode_state()
        self.loaded = loaded
        self.records = []
        self.episode_findings = []
        self.in_episode = True
        ctx = EpisodeContext(self.map_name, self.episode, self.game_seed, self.truth,
                             self.frame_skip)  # fmt: skip
        for o in self.oracles:
            o.reset(ctx)
        state = self.game.get_state()
        self.taken = self.world.taken(state.objects) if state is not None else set()
        rec = self._record(state, action=0, finished=False, wall_s=0.0)
        self.keys = rec.keys
        self.traversal.update(rec.x, rec.y)
        self.cell = self.truth.cell_of(rec.x, rec.y)
        self.reward.enter_cell(self.cell)
        if self.coverage is not None:
            self.coverage.visit_cell(self.cell)
        floors, ceilings = rec.sector_floors, rec.sector_ceilings
        self.abstract = (*self.cell, keys_mask(rec.keys), self.world.doors_mask(floors, ceilings))
        findings = self._observe(rec)
        self.last = rec
        obs = {"depth": self.frames.fill(self._frame(state)),
               "scalars": self._scalars(rec, state, moved=0.0, commanded=False)}  # fmt: skip
        return obs, self._info(rec, findings, {})

    def _stem(self) -> str:
        return episode_stem(self.map_name, self.base_seed, self.episode)

    def _end_episode(self) -> None:
        """Bookkeeping when an episode ends or is cut short by reset / load_state. A cut
        episode is closed for the oracles with a final `done` copy of its last record,
        so open events (an excursion outside the level, a wedge) are still reported."""
        self.in_episode = False
        if self.last is not None and self.records and not self.records[-1].done:
            self._observe(replace(self.last, done=True, notifications=()))
        if self.record_dir is not None and self.records and (
            self.trace_policy == "all" or (self.trace_policy == "findings" and self.episode_findings)
        ):
            write_trace(self.record_dir / "traces" / f"{self._stem()}.parquet", self.records,
                        {"map": self.map_name, "episode": self.episode, "seed": self.game_seed,
                         "env_seed": self.base_seed, "frame_skip": self.frame_skip,
                         "demo": str(self.demo_path) if self.demo_path else None,
                         "loaded": self.loaded})  # fmt: skip
        self.records = []

    # --- stepping ---------------------------------------------------------------------
    def step(self, action: int | np.integer[Any]) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        if not self.in_episode or self.last is None:
            raise RuntimeError("call reset() before step()")
        a = int(action)
        if not 0 <= a < N_ACTIONS:
            raise ValueError(f"action {a} out of range")
        res = self.supervisor.call(self.game.make_action, self._buttons(a), self.frame_skip)
        if res.error is not None or res.hung:
            return self._engine_failure(a, res)
        g = self.game
        finished = g.is_episode_finished()
        state = None if finished else g.get_state()
        prev = self.last
        rec = self._record(state, a, finished, res.wall_s)
        died = rec.dead
        timeout = finished and not died and g.is_episode_timeout_reached()
        goal = finished and not died and not timeout
        if finished:
            rec = replace(rec, done=True, truncated=timeout)
        findings = self._observe(rec)

        crossed = self.traversal.update(rec.x, rec.y)
        newly_taken = (self.world.taken(state.objects) if state is not None else self.taken) - self.taken
        self.taken |= newly_taken
        entities = set(newly_taken)
        if ACTIONS[a].use:
            entities |= set(self.world.switches_hit(rec.x, rec.y, rec.angle))
        cell_count = state_count = None
        cell = self.truth.cell_of(rec.x, rec.y)
        counted = False
        if cell != self.cell:
            self.cell = cell
            counted = self.reward.enter_cell(cell)
            if counted and self.coverage is not None:
                cell_count = self.coverage.visit_cell(cell)
        abstract = (*cell, keys_mask(rec.keys),
                    self.world.doors_mask(rec.sector_floors, rec.sector_ceilings))
        changed = counted or abstract[2:] != (self.abstract or ())[2:]
        if self.coverage is not None and abstract != self.abstract and changed:
            state_count = self.coverage.visit_state(abstract)
        self.abstract = abstract
        if self.coverage is not None:
            for t in entities - self.reward.entities:
                self.coverage.use_thing(t)
            for p in set(crossed) - self.reward.portals:
                self.coverage.traverse_portal(p)
        reward, parts = self.reward.step(
            progress=self.world.progress(rec.keys, crossed), goal=goal, died=died,
            cell_count=cell_count, state_count=state_count, entities=entities,
            portals=crossed, findings=findings,
        )  # fmt: skip

        self._tally(reward, parts)
        moved = float(np.hypot(rec.x - prev.x, rec.y - prev.y))
        obs = {"depth": self.frames.push(self._frame(state)),
               "scalars": self._scalars(rec, state, moved, bool(ACTIONS[a].move_dirs))}  # fmt: skip
        self.last = rec
        self.keys = rec.keys
        info = self._info(rec, findings, parts)
        if finished:
            info["outcome"] = "death" if died else ("timeout" if timeout else "goal")
            info["episode_stats"] = self._episode_stats(info["outcome"])
            self._end_episode()
        return obs, float(reward), bool(died or goal), bool(timeout), info

    def _engine_failure(self, action: int, res: CallResult) -> tuple[Obs, float, bool, bool,
                                                                       dict[str, Any]]:  # fmt: skip
        """The engine crashed or hung: record it on the trace, restart, truncate."""
        assert self.last is not None
        rec = replace(self.last, action=action, error=res.error, step_wall_s=res.wall_s,
                      done=True, truncated=True, notifications=())  # fmt: skip
        findings = self._observe(rec)
        reward, parts = self.reward.step(findings=findings)
        self._tally(reward, parts)
        self.supervisor.restart()
        obs = {"depth": self.frames.frames.copy(),
               "scalars": self._scalars(rec, None, 0.0, False)}  # fmt: skip
        info = self._info(rec, findings, parts)
        info["outcome"] = "hang" if res.hung else "crash"
        info["engine_failure"] = res.error or f"no reply within {res.wall_s:.1f}s"
        info["episode_stats"] = self._episode_stats(info["outcome"])
        self.demo_path = None  # the engine died before writing it
        if self._pending_demos:
            self._pending_demos.pop()
        self._end_episode()
        return obs, float(reward), False, True, info

    def _tally(self, reward: float, parts: dict[str, float]) -> None:
        self.total_steps += 1
        self.ep_steps += 1
        self.ep_return += reward
        for k, v in parts.items():
            self.ep_parts[k] = self.ep_parts.get(k, 0.0) + v

    def _episode_stats(self, outcome: str) -> dict[str, Any]:
        valid = sum(self.ep_parts.get(k, 0.0) for k in VALID_PARTS)
        adv = sum(self.ep_parts.get(k, 0.0) for k in ADV_PARTS)
        return {"map": self.map_name, "outcome": outcome, "steps": self.ep_steps,
                "return": self.ep_return, "validity": valid, "adversarial": adv,
                "parts": dict(self.ep_parts)}  # fmt: skip

    def _post(self, item: dict[str, Any]) -> None:
        if self.queue is not None:
            self.queue.put({"map": self.map_name, "env_seed": self.base_seed,
                            "env_steps": self.total_steps, **item})  # fmt: skip

    def _observe(self, rec: StepRecord) -> list[Finding]:
        self.records.append(rec)
        findings = [f for o in self.oracles for f in o.observe(rec)]
        if self.demo_path is not None:
            findings = [replace(f, demo_path=self.demo_path) for f in findings]
        self.episode_findings += findings
        for f in findings:
            self._post({"type": "finding", "finding": f})
        return findings

    # --- reading the engine -----------------------------------------------------------
    def _record(self, state: Any, action: int, finished: bool, wall_s: float) -> StepRecord:
        g = self.game
        if state is not None:
            v = state.game_variables
            sectors = state.sectors
            floors = tuple(s.floor_height for s in sectors)
            ceilings = tuple(s.ceiling_height for s in sectors)
            taken = self.taken | self.world.taken(state.objects)
            text = state.notifications_buffer or ""
            fresh = tuple(m for m in text.splitlines() if m and m not in self.notifications)
            self.notifications = text
        else:
            v = [g.get_game_variable(gv) for gv in GAME_VARIABLES]
            last = self.last
            floors = last.sector_floors if last else ()
            ceilings = last.sector_ceilings if last else ()
            taken, fresh = self.taken, ()
        act = ACTIONS[action]
        return StepRecord(
            tic=g.get_episode_time(),
            x=float(v[_I[GV.POSITION_X]]), y=float(v[_I[GV.POSITION_Y]]),
            z=float(v[_I[GV.POSITION_Z]]),
            vx=float(v[_I[GV.VELOCITY_X]]), vy=float(v[_I[GV.VELOCITY_Y]]),
            vz=float(v[_I[GV.VELOCITY_Z]]), angle=float(v[_I[GV.ANGLE]]),
            on_ground=bool(v[_I[GV.ON_GROUND]]), dead=bool(v[_I[GV.DEAD]]),
            health=float(v[_I[GV.HEALTH]]), armour=float(v[_I[GV.ARMOR]]),
            ammo=float(v[_I[GV.AMMO2]]), keys=self.world.keys_held(taken),
            killcount=int(v[_I[GV.KILLCOUNT]]), action=action, move_dirs=act.move_dirs,
            use=act.use, jump=act.jump, sector_floors=floors, sector_ceilings=ceilings,
            notifications=fresh, done=finished, step_wall_s=wall_s,
        )  # fmt: skip

    def _frame(self, state: Any) -> np.ndarray:
        if state is not None and state.depth_buffer is not None:
            self.last_frame = self.depth(state.depth_buffer)
        return self.last_frame

    def _scalars(self, rec: StepRecord, state: Any, moved: float,
                 commanded: bool) -> np.ndarray:  # fmt: skip
        cell = self.truth.cell_of(rec.x, rec.y)
        novelty = neighbours = 1.0
        if self.coverage is not None:
            novelty = 1.0 / float(np.sqrt(1.0 + self.coverage.cell_count(cell)))
            neighbours = self.coverage.neighbour_novelty(cell)
        return scalars(ScalarInputs(
            angle=rec.angle, vx=rec.vx, vy=rec.vy, vz=rec.vz, on_ground=rec.on_ground,
            moved_mu=moved, commanded=commanded, tics=self.frame_skip, health=rec.health,
            armour=rec.armour, ammo=rec.ammo, keys=rec.keys,
            elapsed=min(1.0, rec.tic / self.time_limit), novelty=novelty,
            neighbour_novelty=neighbours,
            interactive_in_view=count_pickups_in_view(state.labels) if state is not None else 0,
            use_target=self.world.use_target(rec.x, rec.y, rec.angle),
        ))  # fmt: skip

    def _info(self, rec: StepRecord, findings: list[Finding],
              parts: dict[str, float]) -> dict[str, Any]:  # fmt: skip
        return {
            "map": self.map_name, "seed": self.game_seed,
            "tic": rec.tic, "pos": (rec.x, rec.y, rec.z), "cell": self.truth.cell_of(rec.x, rec.y),
            "keys": sorted(rec.keys), "findings": findings, "reward_parts": parts,
        }  # fmt: skip

    def _flush_demos(self, keep_last: bool = False) -> None:
        """Move finished recordings to their place in the run folder."""
        done = self._pending_demos[:-1] if keep_last else self._pending_demos
        for tmp, dest in done:
            if tmp.exists():
                shutil.move(tmp, dest)
        self._pending_demos = self._pending_demos[-1:] if keep_last else []

    # --- Go-Explore hooks (M8) ----------------------------------------------------------
    def _short(self) -> Path:
        """A short scratch folder: ViZDoom hangs on file paths of ~113+ characters (D33)."""
        if self._demo_tmp is None:
            self._demo_tmp = Path(tempfile.mkdtemp(prefix="qa"))
        return self._demo_tmp

    def save_state(self, save_path: Path) -> None:
        tmp = self._short() / "s.sav"
        self.game.save(str(tmp))
        shutil.move(tmp, save_path)

    def load_state(self, save_path: Path) -> Obs:
        """Go-Explore reset: restore a saved state and start a new oracle episode there.
        No demo is kept for it: a .lmp only replays from the map start."""
        obs, _ = self.reset(options={"load_state": save_path})
        return obs

    # --- run end ----------------------------------------------------------------------
    def finalize_oracles(self) -> list[Finding]:
        """Findings judged across episodes (Families 4 and 6); starts a fresh oracle set."""
        if self.in_episode:
            self._end_episode()
        findings = [f for o in self.oracles for f in o.finalize()]
        if self.record_dir is not None:
            findings = [
                replace(f, demo_path=self.record_dir / "demos" / (
                    episode_stem(self.map_name, self.base_seed, f.evidence["episode"]) + ".lmp"))
                if f.demo_path is None and "episode" in f.evidence else f
                for f in findings
            ]  # fmt: skip
        for f in findings:
            self._post({"type": "finding", "finding": f})
        reach = next((o for o in self.oracles if o.name == "unreachable"), None)
        for gap in getattr(reach, "coverage_gaps", []):
            self._post({"type": "coverage_gap", "gap": gap})
        self.oracles = make_oracles(self.oracle_cfg)
        return findings

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.finalize_oracles()
        finally:
            self.supervisor.close()
            self._flush_demos()
            if self._demo_tmp is not None:
                shutil.rmtree(self._demo_tmp, ignore_errors=True)
