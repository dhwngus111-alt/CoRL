"""Risky Overcooked env wrapper for HSP algorithms.

The HSP runner expects the same public surface as its original Overcooked env:

    reset() -> obs, share_obs, available_actions
    step(actions) -> obs, share_obs, rewards, dones, info, available_actions

This adapter keeps Risky Overcooked as the source of truth for transitions and
only translates actions, observations, rewards, and event categories at the
boundary. Hidden utility follows HSP's category schema, with Risky multipath
events appended.
"""

from __future__ import annotations

import copy
import os
import warnings
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import cv2
import gym
import imageio
import numpy as np
import pygame

from transplant.bootstrap import ensure_paths


ensure_paths()

from risky_overcooked_py.mdp.actions import Action, Direction  # noqa: E402
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv  # noqa: E402
from risky_overcooked_py.mdp.overcooked_mdp import (  # noqa: E402
    BASE_REW_SHAPING_PARAMS,
    OvercookedGridworld,
)
from risky_overcooked_py.visualization.state_visualizer import StateVisualizer  # noqa: E402
from transplant.hsp_hidden_utility import (  # noqa: E402
    HIDDEN_UTILITY_KEYS,
    RISKY_EVENT_KEYS,
    ordered_zero_category_dict,
    risky_events_to_hsp_category_info,
)
from transplant.common import (  # noqa: E402
    DEFAULT_DISTANCE_SHAPING_REW,
    DEFAULT_EPISODE_LENGTH,
    DEFAULT_LAYOUT,
    DEFAULT_P_SLIP,
    DEFAULT_SUBGOAL_DISABLE_STEPS,
    DEFAULT_SUBGOAL_PRESS_REW,
    DEFAULT_TIME_COST,
)

CATEGORY_KEYS = HIDDEN_UTILITY_KEYS

COUNTER_PICKUP_CATEGORIES = {
    "onion": "pickup_onion_from_X",
    "tomato": "pickup_tomato_from_X",
    "dish": "pickup_dish_from_X",
    "soup": "pickup_soup_from_X",
}

DISPENSER_PICKUP_CATEGORIES = {
    ("O", "onion"): "pickup_onion_from_O",
    ("T", "tomato"): "pickup_tomato_from_T",
    ("D", "dish"): "pickup_dish_from_D",
}


def _safe_path_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _ordered_zero_event_dict() -> OrderedDict[str, float]:
    return ordered_zero_category_dict()


def risky_events_to_category_info(event_infos: dict) -> list[OrderedDict[str, float]]:
    """Map Risky raw event booleans into HSP-style hidden utility categories."""

    return risky_events_to_hsp_category_info(event_infos)


def shaped_info_to_array(shaped_info_by_agent: Iterable[dict]) -> np.ndarray:
    return np.asarray(
        [[agent_info.get(key, 0.0) for key in CATEGORY_KEYS] for agent_info in shaped_info_by_agent],
        dtype=np.float32,
    )


class RiskyOvercooked(gym.Env):
    """Risky Overcooked env with the API expected by HSP algorithms."""

    env_name = "RiskyOvercooked-v0"

    def __init__(
        self,
        all_args,
        run_dir: str | Path,
        baselines_reproducible: bool = False,
        featurize_type: tuple[str, str] = ("ppo", "ppo"),
        stuck_time: int = 4,
        rank: int | None = None,
    ):
        if baselines_reproducible:
            np.random.seed(0)

        self.all_args = all_args
        self.run_dir = Path(run_dir)
        self.rank = rank
        self.agent_idx = 0
        self.num_agents = getattr(all_args, "num_agents", 2)
        if self.num_agents != 2:
            raise ValueError("Risky Overcooked transplant v1 supports exactly 2 agents.")

        self.layout_name = getattr(all_args, "layout_name", DEFAULT_LAYOUT)

        self.episode_length = int(getattr(all_args, "episode_length", DEFAULT_EPISODE_LENGTH))
        self._initial_reward_shaping_factor = float(
            getattr(all_args, "initial_reward_shaping_factor", 1.0)
        )
        self.reward_shaping_factor = float(getattr(all_args, "reward_shaping_factor", 1.0))
        self.reward_shaping_horizon = int(getattr(all_args, "reward_shaping_horizon", 0))
        self.use_phi = bool(getattr(all_args, "use_phi", False))
        self.use_hsp = bool(getattr(all_args, "use_hsp", False))
        self.random_index = bool(getattr(all_args, "random_index", False))
        self.use_render = bool(getattr(all_args, "use_render", False))
        self.use_agent_policy_id = bool(getattr(all_args, "use_agent_policy_id", False))
        self.agent_policy_id = [-1.0 for _ in range(self.num_agents)]
        self.random_start_prob = float(getattr(all_args, "random_start_prob", 0.0))
        self.time_cost = float(getattr(all_args, "time_cost", DEFAULT_TIME_COST))
        self.stuck_time = int(stuck_time)
        self.history_sa = []
        self.traj_num = 0
        self.step_count = 0
        self.visualizer = StateVisualizer()
        self.render_traj: dict[str, list] | None = None
        self.last_render_gif_path: Path | None = None
        self.render_gif_paths: list[Path] = []

        if self.use_hsp:
            self.w0 = self.string2array(getattr(all_args, "w0"))
            self.w1 = self.string2array(getattr(all_args, "w1"))
            if len(self.w0) != len(HIDDEN_UTILITY_KEYS) + 1:
                raise ValueError(
                    f"w0 must have {len(HIDDEN_UTILITY_KEYS) + 1} values "
                    f"({len(HIDDEN_UTILITY_KEYS)} event weights + sparse weight), "
                    f"got {len(self.w0)}"
                )
            if len(self.w1) != len(HIDDEN_UTILITY_KEYS) + 1:
                raise ValueError(
                    f"w1 must have {len(HIDDEN_UTILITY_KEYS) + 1} values "
                    f"({len(HIDDEN_UTILITY_KEYS)} event weights + sparse weight), "
                    f"got {len(self.w1)}"
                )

        self._verify_action_mapping()
        self.base_mdp = self._make_mdp(all_args)
        self.base_env = OvercookedEnv.from_mdp(
            self.base_mdp,
            horizon=self.episode_length,
            start_state_fn=self._random_start_state,
            info_level=0,
            time_cost=self.time_cost,
        )
        self.featurize_fn_ppo = lambda state: self.base_env.lossless_state_encoding_mdp(state)
        self.featurize_fn_mapping = {
            "ppo": self.featurize_fn_ppo,
            "bc": self.featurize_fn_ppo,
        }
        self.reset_featurize_type(featurize_type=featurize_type)

    def _terrain_snapshot(self) -> list[list[str]]:
        return [list(row) for row in self.base_mdp.terrain_mtx]

    def _state_snapshot(self):
        if hasattr(self.base_env.state, "deepcopy"):
            return self.base_env.state.deepcopy()
        return self.base_env.state

    @staticmethod
    def _held_object_name(player) -> str | None:
        obj = getattr(player, "held_object", None)
        if obj is None:
            return None
        return obj.name

    def _apply_source_specific_pickups(
        self,
        prev_state,
        next_state,
        joint_action,
        step_shaped_info: list[OrderedDict[str, float]],
    ) -> None:
        for agent_idx, action in enumerate(joint_action):
            if action != Action.INTERACT:
                continue

            prev_player = prev_state.players[agent_idx]
            next_player = next_state.players[agent_idx]
            prev_held_name = self._held_object_name(prev_player)
            next_held_name = self._held_object_name(next_player)
            i_pos = Action.move_in_direction(prev_player.position, prev_player.orientation)
            terrain_type = self.base_mdp.get_terrain_type_at_pos(i_pos)

            if terrain_type == "X":
                if prev_held_name is not None or not prev_state.has_object(i_pos):
                    continue
                source_obj_name = prev_state.get_object(i_pos).name
                if next_held_name != source_obj_name:
                    continue
                category = COUNTER_PICKUP_CATEGORIES.get(source_obj_name)
                if category is not None:
                    step_shaped_info[agent_idx][category] += 1.0
                continue

            if prev_held_name is None:
                category = DISPENSER_PICKUP_CATEGORIES.get((terrain_type, next_held_name))
                if category is not None:
                    step_shaped_info[agent_idx][category] += 1.0
                continue

            if terrain_type == "P" and prev_held_name == "dish" and next_held_name == "soup":
                step_shaped_info[agent_idx]["SOUP_PICKUP"] += 1.0

    def _subgoal_activated_agents(self, prev_state, joint_action) -> list[bool]:
        """Mark only useful G presses that actually open active puddles."""
        subgoal_to_water = getattr(self.base_mdp, "subgoal_to_water", {})
        water_disable_timers = dict(getattr(self.base_mdp, "water_disable_timers", {}))
        if not subgoal_to_water or not water_disable_timers:
            return [False] * self.num_agents

        activated = [False] * self.num_agents
        simulated_timers = dict(water_disable_timers)
        disable_steps = int(
            getattr(self.base_mdp, "subgoal_disable_steps", DEFAULT_SUBGOAL_DISABLE_STEPS)
        )
        for agent_idx, (player, action) in enumerate(zip(prev_state.players, joint_action)):
            if action != Action.INTERACT:
                continue
            if self.base_mdp.get_terrain_type_at_pos(player.position) != "G":
                continue
            linked_waters = subgoal_to_water.get(player.position, ())
            opened_active_water = any(simulated_timers.get(pos, 0) <= 0 for pos in linked_waters)
            if opened_active_water:
                activated[agent_idx] = True
            for pos in linked_waters:
                simulated_timers[pos] = disable_steps
        return activated

    def _apply_subgoal_activated_event(
        self,
        event_infos: dict,
        activated: list[bool],
    ) -> None:
        if not any(activated):
            event_infos.setdefault("subgoal_activated", [False] * self.num_agents)
            return
        values = list(event_infos.get("subgoal_activated", [False] * self.num_agents))
        for agent_idx, did_activate in enumerate(activated):
            values[agent_idx] = bool(values[agent_idx] or did_activate)
        event_infos["subgoal_activated"] = values

    def _hsp_hidden_rewards(self, vec_hidden: np.ndarray, mdp_info: dict) -> tuple[float, float]:
        order_reward = float(
            np.sum(mdp_info.get("sparse_reward_by_agent", [0.0] * self.num_agents))
        )
        env_penalty = float(
            np.sum(mdp_info.get("dropped_reward_by_agent", [0.0] * self.num_agents))
            + self.time_cost
        )
        weights = (self.w0, self.w1) if self.agent_idx == 0 else (self.w1, self.w0)
        return (
            float(
                np.dot(weights[0][:-1], vec_hidden[0])
                + order_reward * weights[0][-1]
                + env_penalty
            ),
            float(
                np.dot(weights[1][:-1], vec_hidden[1])
                + order_reward * weights[1][-1]
                + env_penalty
            ),
        )

    def _render_hud_data(self, state):
        rewards_dict = {}
        for key in (
            "cumulative_shaped_rewards_by_agent",
            "cumulative_sparse_rewards_by_agent",
        ):
            if key in self.base_env.game_stats:
                rewards_dict[key] = np.asarray(self.base_env.game_stats[key]).copy()
        return StateVisualizer.default_hud_data(state, **rewards_dict)

    def _coerce_game_stats_reward_dtype(self) -> None:
        # Risky shaped rewards can be floats, while the upstream env initializes
        # cumulative reward stats as int arrays. Keep this adapter-local so the
        # original Risky Overcooked source remains untouched.
        for key in (
            "cumulative_shaped_rewards_by_agent",
            "cumulative_sparse_rewards_by_agent",
        ):
            if key in self.base_env.game_stats:
                self.base_env.game_stats[key] = np.asarray(
                    self.base_env.game_stats[key], dtype=np.float64
                )

    def _surface_to_rgb_array(self, surface: pygame.Surface) -> np.ndarray:
        buffer = pygame.surfarray.array3d(surface)
        image = np.asarray(buffer).copy()
        image = np.flip(np.rot90(image, 3), 1)
        image = cv2.resize(image, (2 * 528, 2 * 464))
        return image.astype(np.uint8, copy=False)

    def _render_frame(
        self,
        state=None,
        grid=None,
        hud_data=None,
        water_disable_timers: dict | None = None,
    ) -> np.ndarray:
        state = self.base_env.state if state is None else state
        grid = self.base_mdp.terrain_mtx if grid is None else grid
        hud_data = self._render_hud_data(state) if hud_data is None else hud_data
        render_mdp = self.base_mdp
        if water_disable_timers is not None:
            # Historical GIF frames must use their own timer state without ever
            # mutating the live MDP that drives transitions and slip behavior.
            render_mdp = copy.copy(self.base_mdp)
            render_mdp.water_disable_timers = dict(water_disable_timers)
        surface = self.visualizer.render_state(
            state=state,
            grid=grid,
            hud_data=hud_data,
            mdp=render_mdp,
        )
        return self._surface_to_rgb_array(surface)

    def _init_render_traj(self) -> None:
        self.render_traj = {
            "states": [],
            "grids": [],
            "hud_data": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "infos": [],
            "water_disable_timers": [],
        }
        self.last_render_gif_path = None

    def _record_render_step(
        self,
        joint_action,
        sparse_reward: float,
        done: bool,
        info: dict | None,
    ) -> None:
        save_limit = self._render_gif_save_limit()
        if save_limit is not None and len(self.render_gif_paths) >= save_limit:
            return
        if self.render_traj is None:
            self._init_render_traj()
        state = self._state_snapshot()
        self.render_traj["states"].append(state)
        self.render_traj["grids"].append(self._terrain_snapshot())
        self.render_traj["hud_data"].append(self._render_hud_data(state))
        self.render_traj["actions"].append(joint_action)
        self.render_traj["rewards"].append(float(sparse_reward))
        self.render_traj["dones"].append(bool(done))
        self.render_traj["infos"].append(dict(info or {}))
        self.render_traj["water_disable_timers"].append(
            dict(getattr(self.base_mdp, "water_disable_timers", {}) or {})
        )

    def _render_return_name(self, info: dict) -> str:
        episode = info.get("episode", {})
        value = episode.get("ep_sparse_r")
        if value is None:
            value = np.asarray(
                self.base_env.game_stats.get("cumulative_sparse_rewards_by_agent", [0.0, 0.0])
            ).sum()
        try:
            return f"{float(np.asarray(value).sum()):g}"
        except Exception:
            return str(value).replace("/", "_")

    def _render_gif_save_limit(self) -> int | None:
        if not hasattr(self.all_args, "render_eval_gif_episodes"):
            return None
        total = max(0, int(getattr(self.all_args, "render_eval_gif_episodes", 0) or 0))
        if total <= 0:
            return 0
        thread_count = max(1, int(getattr(self.all_args, "n_eval_rollout_threads", 1) or 1))
        enabled_ranks = min(total, thread_count)
        return (total + enabled_ranks - 1) // enabled_ranks

    def _save_render_gif(self, info: dict) -> Path | None:
        if not self.render_traj or not self.render_traj["states"]:
            return None
        save_limit = self._render_gif_save_limit()
        if save_limit is not None and len(self.render_gif_paths) >= save_limit:
            return None
        save_dir = self.run_dir / "gifs" / self.layout_name
        subdir = str(getattr(self.all_args, "render_gif_subdir", "") or "").strip()
        if subdir:
            save_dir = save_dir / _safe_path_component(subdir)
        if self.rank is not None:
            save_dir = save_dir / f"env_{self.rank}"
        save_dir = save_dir / f"traj_num_{self.traj_num}"
        save_dir.mkdir(parents=True, exist_ok=True)

        frame_keys = (
            "states",
            "grids",
            "hud_data",
            "actions",
            "rewards",
            "dones",
            "infos",
            "water_disable_timers",
        )
        frame_lengths = {key: len(self.render_traj.get(key, [])) for key in frame_keys}
        if len(set(frame_lengths.values())) != 1:
            raise ValueError(f"Render trajectory frame data is misaligned: {frame_lengths}")

        frames = [
            self._render_frame(
                state=state,
                grid=grid,
                hud_data=hud_data,
                water_disable_timers=water_disable_timers,
            )
            for state, grid, hud_data, water_disable_timers in zip(
                self.render_traj["states"],
                self.render_traj["grids"],
                self.render_traj["hud_data"],
                self.render_traj["water_disable_timers"],
            )
        ]
        output_path = save_dir / f"reward_{self._render_return_name(info)}.gif"
        imageio.mimsave(output_path, frames, duration=0.15)
        self.last_render_gif_path = output_path
        self.render_gif_paths.append(output_path)
        return output_path

    def _make_mdp(self, all_args) -> OvercookedGridworld:
        rew_shaping_params = dict(BASE_REW_SHAPING_PARAMS)
        rew_shaping_params.update(
            {   # float64 -> int64로 들어오도록 변환
                "PLACEMENT_IN_POT_REW": int(getattr(all_args, "placement_in_pot_rew", 3.0)),
                "DISH_PICKUP_REWARD": int(getattr(all_args, "dish_pickup_reward", 1.0)),
                "SOUP_PICKUP_REWARD": int(getattr(all_args, "soup_pickup_reward", 3.0)),
                "DISTANCE_SHAPING_REW": float(
                    getattr(all_args, "distance_shaping_rew", DEFAULT_DISTANCE_SHAPING_REW)
                ),
                "SUBGOAL_PRESS_REW": float(
                    getattr(all_args, "subgoal_press_rew", DEFAULT_SUBGOAL_PRESS_REW)
                ),
            }
        )
        mdp_kwargs = {
            "rew_shaping_params": rew_shaping_params,
            "p_slip": float(getattr(all_args, "p_slip", DEFAULT_P_SLIP)),
            "neglect_boarders": bool(getattr(all_args, "neglect_boarders", False)),
            "subgoal_disable_steps": int(
                getattr(all_args, "subgoal_disable_steps", DEFAULT_SUBGOAL_DISABLE_STEPS)
            ),
            "handoff_shaping": bool(getattr(all_args, "handoff_shaping", False)),
        }
        competitive_onion_supply = getattr(all_args, "competitive_onion_supply", None)
        if competitive_onion_supply is not None:
            mdp_kwargs["competitive_onion_supply"] = int(competitive_onion_supply)
        return OvercookedGridworld.from_layout_name(self.layout_name, **mdp_kwargs)

    def _random_start_state(self):
        if self.random_start_prob <= 0:
            return self.base_mdp.get_standard_start_state()
        if np.random.uniform(0, 1) <= self.random_start_prob:
            return self.base_mdp.get_random_start_state()
        return self.base_mdp.get_standard_start_state()

    def _verify_action_mapping(self) -> None:
        expected = [
            Direction.NORTH,
            Direction.SOUTH,
            Direction.EAST,
            Direction.WEST,
            Action.STAY,
            Action.INTERACT,
        ]
        actual = list(Action.INDEX_TO_ACTION)
        if actual != expected:
            raise RuntimeError(f"Unexpected Risky action order: {actual!r}")

    def reset_featurize_type(self, featurize_type: tuple[str, str] = ("ppo", "ppo")):
        if len(featurize_type) != 2:
            raise ValueError("featurize_type must contain one entry per agent.")
        normalized = tuple("ppo" if item == "bc" else item for item in featurize_type)
        if any(item != "ppo" for item in normalized):
            raise ValueError(f"Unsupported featurize_type for Risky adapter: {featurize_type}")
        self.featurize_type = normalized
        self.featurize_fn = lambda state: [
            self.featurize_fn_mapping[f](state)[i].astype(np.float32) * 255.0
            for i, f in enumerate(self.featurize_type)
        ]

        self.observation_space = []
        self.share_observation_space = []
        self.action_space = []
        self._setup_observation_space()
        for idx in range(self.num_agents):
            self.observation_space.append(self.ppo_observation_space)
            self.action_space.append(gym.spaces.Discrete(len(Action.ALL_ACTIONS)))
            self.share_observation_space.append(self._setup_share_observation_space())

    def _setup_observation_space(self) -> None:
        dummy_state = self.base_mdp.get_standard_start_state()
        obs_shape = self.featurize_fn_ppo(dummy_state)[0].shape
        high = np.ones(obs_shape, dtype=np.float32) * float("inf")
        low = np.zeros(obs_shape, dtype=np.float32)
        self.ppo_observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def _setup_share_observation_space(self):
        dummy_state = self.base_mdp.get_standard_start_state()
        base_shape = list(self.featurize_fn_ppo(dummy_state)[0].shape)
        if self.use_agent_policy_id:
            base_shape[-1] += 1
        share_shape = [base_shape[0], base_shape[1], base_shape[2] * self.num_agents]
        high = np.ones(share_shape, dtype=np.float32) * float("inf")
        low = np.zeros(share_shape, dtype=np.float32)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def _set_agent_policy_id(self, agent_policy_id):
        self.agent_policy_id = agent_policy_id

    def _gen_share_observation(self, state) -> np.ndarray:
        share_obs = [obs.astype(np.float32) for obs in self.featurize_fn_ppo(state)]
        if self.agent_idx == 1:
            share_obs = [share_obs[1], share_obs[0]]
        if self.use_agent_policy_id:
            for agent_idx in range(self.num_agents):
                id_plane = np.ones((*share_obs[agent_idx].shape[:2], 1), dtype=np.float32)
                share_obs[agent_idx] = np.concatenate(
                    [share_obs[agent_idx], id_plane * self.agent_policy_id[agent_idx]],
                    axis=-1,
                )
        share_obs0 = np.concatenate([share_obs[0], share_obs[1]], axis=-1) * 255.0
        share_obs1 = np.concatenate([share_obs[1], share_obs[0]], axis=-1) * 255.0
        return np.stack([share_obs0, share_obs1], axis=0).astype(np.float32)

    def _action_convertor(self, action) -> list[int]:
        converted = []
        for item in list(action):
            arr = np.asarray(item)
            converted.append(int(arr.reshape(-1)[0]))
        return converted

    def step(self, action):
        self.step_count += 1
        action_idx = self._action_convertor(action)
        if not all(self.action_space[0].contains(idx) for idx in action_idx):
            raise AssertionError(f"{action!r} contains an invalid action index.")

        agent_action, other_agent_action = [Action.INDEX_TO_ACTION[idx] for idx in action_idx]
        joint_action = (agent_action, other_agent_action)
        if self.agent_idx == 1:
            joint_action = (other_agent_action, agent_action)

        if self.stuck_time > 0 and self.history_sa:
            self.history_sa[-1][1] = joint_action

        prev_state = self._state_snapshot()
        subgoal_activated = self._subgoal_activated_agents(prev_state, joint_action)
        next_state, sparse_reward, done, info = self.base_env.step(
            joint_action, display_phi=self.use_phi, get_mdp_info=True
        )
        mdp_info = info.get("mdp_info", {})
        event_infos = mdp_info.get("event_infos", {})
        self._apply_subgoal_activated_event(event_infos, subgoal_activated)
        step_shaped_info = risky_events_to_category_info(event_infos)
        self._apply_source_specific_pickups(
            prev_state,
            next_state,
            joint_action,
            step_shaped_info,
        )
        step_category = shaped_info_to_array(step_shaped_info)
        self.cumulative_category_rewards_by_agent += step_category

        for agent_idx in range(self.num_agents):
            for key, value in step_shaped_info[agent_idx].items():
                self.cumulative_shaped_info[agent_idx][key] += value

        info["step_shaped_info_by_agent"] = step_shaped_info
        info["shaped_info_by_agent"] = self.cumulative_shaped_info
        info["vec_shaped_info_by_agent"] = step_category[:, : len(HIDDEN_UTILITY_KEYS)]
        info["risky_event_infos"] = event_infos
        info["risky_event_counts"] = self._risk_event_counts(event_infos)

        dense_reward = info["shaped_r_by_agent"]
        if self.use_phi:
            potential = info["phi_s_prime"] - info["phi_s"]
            shaped_reward_p0 = sparse_reward + self.reward_shaping_factor * potential
            shaped_reward_p1 = sparse_reward + self.reward_shaping_factor * potential
        elif self.use_hsp:
            vec_hidden = step_category[:, : len(HIDDEN_UTILITY_KEYS)]
            hidden_reward = self._hsp_hidden_rewards(vec_hidden, mdp_info)
            shaped_reward_p0 = hidden_reward[0] + self.reward_shaping_factor * dense_reward[0]
            shaped_reward_p1 = hidden_reward[1] + self.reward_shaping_factor * dense_reward[1]
        else:
            shaped_reward_p0 = sparse_reward + self.reward_shaping_factor * dense_reward[0]
            shaped_reward_p1 = sparse_reward + self.reward_shaping_factor * dense_reward[1]

        reward = [[shaped_reward_p0], [shaped_reward_p1]]
        if self.agent_idx == 1:
            reward = [[shaped_reward_p1], [shaped_reward_p0]]

        self.history_sa = self.history_sa[1:] + [[next_state, None]]

        info["stuck"] = self._stuck_info()
        if done:
            self._patch_episode_info(info)
        if self.use_render:
            self._record_render_step(joint_action, sparse_reward, done, info)
            if done:
                self._save_render_gif(info)

        ob_p0, ob_p1 = self.featurize_fn(next_state)
        both_agents_ob = (ob_p0, ob_p1)
        if self.agent_idx == 1:
            both_agents_ob = (ob_p1, ob_p0)

        share_obs = self._gen_share_observation(self.base_env.state)
        done_by_agent = [done, done]
        available_actions = np.ones((self.num_agents, len(Action.ALL_ACTIONS)), dtype=np.uint8)
        return both_agents_ob, share_obs, reward, done_by_agent, info, available_actions

    def _risk_event_counts(self, event_infos: dict) -> dict[str, list[float]]:
        counts = {}
        for key in RISKY_EVENT_KEYS:
            if key in event_infos:
                counts[key] = [float(bool(v)) for v in event_infos[key]]
        return counts

    def _patch_episode_info(self, info: dict) -> None:
        episode = info.setdefault("episode", {})
        episode["ep_category_r_by_agent"] = self.cumulative_category_rewards_by_agent.copy()
        episode["ep_length"] = self.base_env.state.timestep
        episode.setdefault("ep_sparse_r_by_agent", self.base_env.game_stats["cumulative_sparse_rewards_by_agent"])
        episode.setdefault("ep_shaped_r_by_agent", self.base_env.game_stats["cumulative_shaped_rewards_by_agent"])
        episode.setdefault("ep_sparse_r", sum(episode["ep_sparse_r_by_agent"]))
        episode.setdefault("ep_shaped_r", sum(episode["ep_shaped_r_by_agent"]))
        episode["ep_risky_event_counts"] = {
            key: [len(self.base_env.game_stats[key][0]), len(self.base_env.game_stats[key][1])]
            for key in RISKY_EVENT_KEYS
            if key in self.base_env.game_stats
        }

    def anneal_reward_shaping_factor(self, timesteps):
        new_factor = self._anneal(
            self._initial_reward_shaping_factor,
            timesteps,
            self.reward_shaping_horizon,
        )
        self.set_reward_shaping_factor(new_factor)

    def _anneal(self, start_v, curr_t, end_t, end_v=0, start_t=0):
        if end_t == 0:
            return start_v
        off_t = curr_t - start_t
        fraction = max(1 - float(off_t) / (end_t - start_t), 0)
        return fraction * start_v + (1 - fraction) * end_v

    def set_reward_shaping_factor(self, factor):
        self.reward_shaping_factor = factor

    def reset(self, reset_choose=True):
        if reset_choose:
            self.traj_num += 1
            self.step_count = 0
            self.base_env.reset()
            self._coerce_game_stats_reward_dtype()
            self.cumulative_shaped_info = [_ordered_zero_event_dict(), _ordered_zero_event_dict()]
            self.cumulative_category_rewards_by_agent = np.zeros(
                (self.num_agents, len(CATEGORY_KEYS)), dtype=np.float32
            )
            if self.use_render:
                self._init_render_traj()
                self._record_render_step(
                    joint_action=None,
                    sparse_reward=0.0,
                    done=False,
                    info={"initial_state": True},
                )

        if self.random_index:
            self.agent_idx = int(np.random.choice([0, 1]))

        self.mdp = self.base_env.mdp
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)
        if self.stuck_time > 0:
            self.history_sa = [None for _ in range(self.stuck_time - 1)] + [
                [self.base_env.state, None]
            ]

        both_agents_ob = (ob_p0, ob_p1)
        if self.agent_idx == 1:
            both_agents_ob = (ob_p1, ob_p0)

        share_obs = self._gen_share_observation(self.base_env.state)
        available_actions = np.ones((self.num_agents, len(Action.ALL_ACTIONS)), dtype=np.uint8)
        return both_agents_ob, share_obs, available_actions

    def _stuck_info(self):
        stuck_info = []
        for agent_id in range(self.num_agents):
            stuck, history_a = self.is_stuck(agent_id)
            if stuck:
                history_a_idxes = [Action.ACTION_TO_INDEX[a] for a in history_a]
                stuck_info.append([True, history_a_idxes])
            else:
                stuck_info.append([False, []])
        return stuck_info

    def is_stuck(self, agent_id):
        if self.stuck_time == 0 or None in self.history_sa:
            return False, []
        history_s = [sa[0] for sa in self.history_sa]
        history_a = [sa[1][agent_id] for sa in self.history_sa[:-1]]
        player_s = [state.players[agent_id] for state in history_s]
        pos_and_ors = [player.pos_and_or for player in player_s]
        cur_po = pos_and_ors[-1]
        if all(po[0] == cur_po[0] and po[1] == cur_po[1] for po in pos_and_ors):
            return True, history_a
        return False, []

    def string2array(self, weight: str) -> np.ndarray:
        return np.asarray([float(item) for item in weight.split(",")], dtype=np.float32)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return [seed]

    def close(self):
        return None

    def render(self, mode="human"):
        if mode == "human":
            print(self.base_env)
            return None
        if mode == "rgb_array":
            return self._render_frame()
        raise NotImplementedError(f"Unsupported render mode: {mode}")


Overcooked = RiskyOvercooked
