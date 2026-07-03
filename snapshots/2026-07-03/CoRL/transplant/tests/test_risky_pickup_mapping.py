from __future__ import annotations

import unittest
import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

from transplant.bootstrap import ensure_paths

ensure_paths()

missing_modules = [
    module_name
    for module_name in ("numpy", "gym", "cv2", "imageio", "pygame")
    if importlib.util.find_spec(module_name) is None
]

if not missing_modules:
    import numpy as np

    sys.modules.setdefault(
        "risky_overcooked_py.visualization.state_visualizer",
        types.SimpleNamespace(
            StateVisualizer=type(
                "StateVisualizer",
                (),
                {"default_hud_data": staticmethod(lambda state, **kwargs: kwargs)},
            )
        ),
    )

    from transplant.adapters.risky_overcooked_env import (
        RiskyOvercooked,
        risky_events_to_category_info,
        shaped_info_to_array,
    )
    from transplant.hsp_hidden_utility import (
        HSP_CORE_CATEGORY_KEYS,
        HSP_MANY_ORDERS_CORE_W0_SPEC,
        HIDDEN_UTILITY_KEYS,
        RISKY_MULTIPATH_EVENT_KEYS,
        build_hsp_w0_spec,
        build_hsp_w1_spec,
        ordered_zero_category_dict,
    )
    from transplant import train_risky_hsp, train_risky_hsp_s1_bundle
    from transplant.common import setup_device

    from risky_overcooked_py.mdp.actions import Action, Direction
    from risky_overcooked_py.mdp.overcooked_mdp import (
        BASE_REW_SHAPING_PARAMS,
        ObjectState,
        OvercookedGridworld,
        OvercookedState,
        PlayerState,
        Recipe,
    )

    Recipe.configure({})


PLAYER_POS = (1, 1)
TARGET_POS = (2, 1)

PICKUP_KEYS = (
    "pickup_onion_from_X",
    "pickup_onion_from_O",
    "pickup_tomato_from_X",
    "pickup_tomato_from_T",
    "pickup_dish_from_X",
    "pickup_dish_from_D",
    "pickup_soup_from_X",
    "SOUP_PICKUP",
)


class FakeMdp:
    def __init__(self, terrain_type: str):
        self.terrain_type = terrain_type

    def get_terrain_type_at_pos(self, pos):
        assert pos == TARGET_POS
        return self.terrain_type


def make_object(name: str, pos):
    return ObjectState(name, pos)


def make_player(idx: int, held_name: str | None = None):
    held_object = None
    if held_name is not None:
        held_object = make_object(held_name, PLAYER_POS)
    return PlayerState(
        PLAYER_POS if idx == 0 else (5, 1),
        Direction.EAST,
        idx,
        held_object=held_object,
    )


def make_state(player0_held: str | None = None, counter_object: str | None = None):
    objects = {}
    if counter_object is not None:
        objects[TARGET_POS] = make_object(counter_object, TARGET_POS)
    return OvercookedState(
        players=[make_player(0, player0_held), make_player(1)],
        objects=objects,
    )


@unittest.skipIf(missing_modules, f"Missing runtime dependencies: {missing_modules}")
class RiskyPickupMappingTest(unittest.TestCase):
    def apply_pickup_correction(
        self,
        terrain_type: str,
        prev_held: str | None,
        next_held: str | None,
        counter_object: str | None = None,
    ):
        env = object.__new__(RiskyOvercooked)
        env.base_mdp = FakeMdp(terrain_type)
        step_info = [ordered_zero_category_dict(), ordered_zero_category_dict()]
        env._apply_source_specific_pickups(
            make_state(prev_held, counter_object),
            make_state(next_held),
            (Action.INTERACT, Action.STAY),
            step_info,
        )
        return step_info[0]

    def assert_only_pickup(self, info, expected_key: str):
        for key in PICKUP_KEYS:
            expected_value = 1.0 if key == expected_key else 0.0
            self.assertEqual(info[key], expected_value, key)

    def test_raw_pickup_events_do_not_directly_fill_source_specific_categories(self):
        step_info = risky_events_to_category_info(
            {
                "onion_pickup": [True, False],
                "tomato_pickup": [True, False],
                "dish_pickup": [True, False],
                "soup_pickup": [True, False],
            }
        )

        for key in PICKUP_KEYS:
            self.assertEqual(step_info[0][key], 0.0, key)

    def test_subgoal_interact_logs_hidden_utility_event(self):
        mdp = OvercookedGridworld.from_layout_name(
            "risky_multipath_subgoal",
            p_slip=0.4,
            subgoal_disable_steps=60,
        )
        self.assertEqual(mdp.subgoal_disable_steps, 60)
        state = mdp.get_standard_start_state()
        state, _ = mdp.get_state_transition(state, (Action.STAY, Direction.EAST))

        env = object.__new__(RiskyOvercooked)
        env.base_mdp = mdp
        env.num_agents = 2
        activated = env._subgoal_activated_agents(state, (Action.STAY, Action.INTERACT))
        self.assertEqual(activated, [False, True])

        _, mdp_info = mdp.get_state_transition(state, (Action.STAY, Action.INTERACT))
        self.assertNotIn("subgoal_activated", mdp_info["event_infos"])
        env._apply_subgoal_activated_event(mdp_info["event_infos"], activated)

        self.assertEqual(mdp_info["event_infos"]["subgoal_activated"], [False, True])
        self.assertTrue(mdp.water_disable_timers)
        self.assertTrue(
            all(timer > 0 for timer in mdp.water_disable_timers.values()),
            mdp.water_disable_timers,
        )
        step_info = risky_events_to_category_info(mdp_info["event_infos"])
        self.assertEqual(step_info[1]["subgoal_activated"], 1.0)
        self.assertEqual(shaped_info_to_array(step_info).shape, (2, len(HIDDEN_UTILITY_KEYS)))

    def test_repeated_subgoal_interact_does_not_log_hidden_utility_event(self):
        rew_shaping_params = dict(BASE_REW_SHAPING_PARAMS)
        rew_shaping_params["DISTANCE_SHAPING_REW"] = 0.0
        rew_shaping_params["SUBGOAL_PRESS_REW"] = 2.0
        mdp = OvercookedGridworld.from_layout_name(
            "risky_multipath_subgoal",
            rew_shaping_params=rew_shaping_params,
            p_slip=0.4,
            subgoal_disable_steps=60,
        )
        state = mdp.get_standard_start_state()
        state, _ = mdp.get_state_transition(state, (Action.STAY, Direction.EAST))

        env = object.__new__(RiskyOvercooked)
        env.base_mdp = mdp
        env.num_agents = 2

        first_activated = env._subgoal_activated_agents(state, (Action.STAY, Action.INTERACT))
        next_state, first_info = mdp.get_state_transition(state, (Action.STAY, Action.INTERACT))
        env._apply_subgoal_activated_event(first_info["event_infos"], first_activated)
        first_step_info = risky_events_to_category_info(first_info["event_infos"])

        self.assertEqual(first_activated, [False, True])
        self.assertEqual(first_step_info[1]["subgoal_activated"], 1.0)
        self.assertAlmostEqual(first_info["shaped_reward_by_agent"][1], 2.0)
        self.assertTrue(
            all(timer > 0 for timer in mdp.water_disable_timers.values()),
            mdp.water_disable_timers,
        )

        repeated_activated = env._subgoal_activated_agents(
            next_state,
            (Action.STAY, Action.INTERACT),
        )
        _, repeated_info = mdp.get_state_transition(
            next_state,
            (Action.STAY, Action.INTERACT),
        )
        env._apply_subgoal_activated_event(repeated_info["event_infos"], repeated_activated)
        repeated_step_info = risky_events_to_category_info(repeated_info["event_infos"])

        self.assertEqual(repeated_activated, [False, False])
        self.assertEqual(repeated_info["event_infos"]["subgoal_activated"], [False, False])
        self.assertEqual(repeated_step_info[1]["subgoal_activated"], 0.0)
        self.assertAlmostEqual(repeated_info["shaped_reward_by_agent"][1], 0.0)

    def test_onion_pickup_distinguishes_counter_and_dispenser(self):
        info = self.apply_pickup_correction("X", None, "onion", counter_object="onion")
        self.assert_only_pickup(info, "pickup_onion_from_X")

        info = self.apply_pickup_correction("O", None, "onion")
        self.assert_only_pickup(info, "pickup_onion_from_O")

    def test_tomato_pickup_distinguishes_counter_and_dispenser(self):
        info = self.apply_pickup_correction("X", None, "tomato", counter_object="tomato")
        self.assert_only_pickup(info, "pickup_tomato_from_X")

        info = self.apply_pickup_correction("T", None, "tomato")
        self.assert_only_pickup(info, "pickup_tomato_from_T")

    def test_dish_pickup_distinguishes_counter_and_dispenser(self):
        info = self.apply_pickup_correction("X", None, "dish", counter_object="dish")
        self.assert_only_pickup(info, "pickup_dish_from_X")

        info = self.apply_pickup_correction("D", None, "dish")
        self.assert_only_pickup(info, "pickup_dish_from_D")

    def test_soup_pickup_distinguishes_counter_and_pot(self):
        info = self.apply_pickup_correction("X", None, "soup", counter_object="soup")
        self.assert_only_pickup(info, "pickup_soup_from_X")

        info = self.apply_pickup_correction("P", "dish", "soup", counter_object="soup")
        self.assert_only_pickup(info, "SOUP_PICKUP")


@unittest.skipIf(missing_modules, f"Missing runtime dependencies: {missing_modules}")
class HspWeightSpecTest(unittest.TestCase):
    def test_default_w0_uses_original_many_orders_core_and_zero_risky_extra(self):
        values = build_hsp_w0_spec().split(",")
        core_end = len(HSP_CORE_CATEGORY_KEYS)

        self.assertEqual(values[:core_end], list(HSP_MANY_ORDERS_CORE_W0_SPEC))
        self.assertEqual(
            values[core_end:-1],
            ["0"] * len(RISKY_MULTIPATH_EVENT_KEYS),
        )
        self.assertEqual(values[-1], "r[0:1:2]")
        self.assertEqual(len(values), len(HIDDEN_UTILITY_KEYS) + 1)

    def test_default_w1_keeps_only_sparse_reward_weight(self):
        values = build_hsp_w1_spec().split(",")

        self.assertEqual(values[:-1], ["0"] * len(HIDDEN_UTILITY_KEYS))
        self.assertEqual(values[-1], "1")


@unittest.skipIf(missing_modules, f"Missing runtime dependencies: {missing_modules}")
class HspFinalGifConfigTest(unittest.TestCase):
    def test_final_render_env_enables_exact_requested_gif_budget(self):
        created_envs = []

        class FakeEnv:
            def __init__(self, all_args, run_dir, rank=None):
                self.all_args = all_args
                self.run_dir = run_dir
                self.rank = rank
                self.use_render = False
                self.seed_value = None
                created_envs.append(self)

            def seed(self, value):
                self.seed_value = value

        args = types.SimpleNamespace(
            hsp_final_gif_episodes=3,
            use_render=False,
            random_index=True,
            n_rollout_threads=100,
            n_eval_rollout_threads=1,
            render_eval_gif_episodes=0,
            render_gif_subdir="",
            seed=7,
        )

        with mock.patch.object(train_risky_hsp, "RiskyOvercooked", FakeEnv), mock.patch.object(
            train_risky_hsp, "ShareDummyVecEnv", side_effect=lambda env_fns: env_fns
        ):
            env_fns = train_risky_hsp.make_final_render_env(args, Path("/tmp/hsp-gif-test"))
            for env_fn in env_fns:
                env_fn()

        self.assertEqual(len(created_envs), 3)
        for rank, env in enumerate(created_envs):
            self.assertTrue(env.use_render)
            self.assertTrue(env.all_args.use_render)
            self.assertFalse(env.all_args.random_index)
            self.assertEqual(env.all_args.n_rollout_threads, 3)
            self.assertEqual(env.all_args.n_eval_rollout_threads, 3)
            self.assertEqual(env.all_args.render_eval_gif_episodes, 3)
            self.assertEqual(env.all_args.render_gif_subdir, "final_hsp_s1")
            self.assertEqual(env.seed_value, args.seed * 70000 + rank * 10000 + 777)


@unittest.skipIf(missing_modules, f"Missing runtime dependencies: {missing_modules}")
class ComputeDeviceSafetyTest(unittest.TestCase):
    @staticmethod
    def args(*, cuda: bool):
        return types.SimpleNamespace(
            cuda=cuda,
            cuda_deterministic=True,
            n_training_threads=1,
        )

    def test_requested_cuda_fails_instead_of_falling_back_to_cpu(self):
        with mock.patch("transplant.common.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Refusing to fall back to CPU"):
                setup_device(self.args(cuda=True))

    def test_explicit_cpu_mode_remains_supported(self):
        with mock.patch("transplant.common.torch.cuda.is_available") as is_available:
            device = setup_device(self.args(cuda=False))

        self.assertEqual(device.type, "cpu")
        is_available.assert_not_called()

    def test_cuda_allocation_failure_is_fatal(self):
        with mock.patch(
            "transplant.common.torch.cuda.is_available", return_value=True
        ), mock.patch("transplant.common.torch.empty", side_effect=RuntimeError("driver lost")):
            with self.assertRaisesRegex(RuntimeError, "tensor allocation failed"):
                setup_device(self.args(cuda=True))

    def test_hsp_bundle_preflight_fails_before_wandb_initialization(self):
        base_args = types.SimpleNamespace()
        with mock.patch.object(
            train_risky_hsp_s1_bundle, "get_config", return_value=object()
        ), mock.patch.object(
            train_risky_hsp_s1_bundle, "parse_args", return_value=base_args
        ), mock.patch.object(
            train_risky_hsp_s1_bundle, "validate_hsp_args"
        ), mock.patch.object(
            train_risky_hsp_s1_bundle,
            "setup_device",
            side_effect=RuntimeError("CUDA unavailable"),
        ), mock.patch.object(train_risky_hsp_s1_bundle, "init_wandb") as init_wandb:
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                train_risky_hsp_s1_bundle.main([])

        init_wandb.assert_not_called()


class HspStageOneShellDefaultsTest(unittest.TestCase):
    def test_hsp_stage_one_uses_original_io_intervals(self):
        script = (
            Path(__file__).resolve().parents[1] / "scripts" / "03_train_hsp_s1_risky.sh"
        ).read_text()

        self.assertIn('export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"', script)
        self.assertIn('export LOG_INTERVAL="${LOG_INTERVAL:-10}"', script)


@unittest.skipIf(missing_modules, f"Missing runtime dependencies: {missing_modules}")
class HspRewardMappingTest(unittest.TestCase):
    def make_env(self, time_cost: float = 0.0, agent_idx: int = 0):
        env = object.__new__(RiskyOvercooked)
        env.num_agents = 2
        env.agent_idx = agent_idx
        env.time_cost = time_cost
        env.w0 = np.zeros(len(HIDDEN_UTILITY_KEYS) + 1, dtype=np.float32)
        env.w1 = np.zeros(len(HIDDEN_UTILITY_KEYS) + 1, dtype=np.float32)
        return env

    def zero_hidden(self):
        return np.zeros((2, len(HIDDEN_UTILITY_KEYS)), dtype=np.float32)

    def test_delivery_event_uses_delivery_weight(self):
        env = self.make_env()
        delivery_idx = HIDDEN_UTILITY_KEYS.index("delivery")
        vec_hidden = self.zero_hidden()
        vec_hidden[0, delivery_idx] = 1.0
        env.w0[delivery_idx] = -10.0

        hidden_reward = env._hsp_hidden_rewards(
            vec_hidden,
            {"sparse_reward_by_agent": [0.0, 0.0], "dropped_reward_by_agent": [0.0, 0.0]},
        )

        self.assertEqual(hidden_reward, (-10.0, 0.0))

    def test_order_reward_uses_sparse_reward_weight(self):
        env = self.make_env()
        vec_hidden = self.zero_hidden()
        env.w0[-1] = 1.0
        env.w1[-1] = 1.0

        hidden_reward = env._hsp_hidden_rewards(
            vec_hidden,
            {"sparse_reward_by_agent": [20.0, 0.0], "dropped_reward_by_agent": [0.0, 0.0]},
        )

        self.assertEqual(hidden_reward, (20.0, 20.0))

    def test_delivery_event_and_order_reward_are_separate_terms(self):
        env = self.make_env()
        delivery_idx = HIDDEN_UTILITY_KEYS.index("delivery")
        vec_hidden = self.zero_hidden()
        vec_hidden[0, delivery_idx] = 1.0
        env.w0[delivery_idx] = -10.0
        env.w0[-1] = 1.0

        hidden_reward = env._hsp_hidden_rewards(
            vec_hidden,
            {"sparse_reward_by_agent": [20.0, 0.0], "dropped_reward_by_agent": [0.0, 0.0]},
        )

        self.assertEqual(hidden_reward, (10.0, 0.0))

    def test_dropped_reward_is_not_gated_by_order_reward_weight(self):
        env = self.make_env()
        vec_hidden = self.zero_hidden()

        hidden_reward = env._hsp_hidden_rewards(
            vec_hidden,
            {"sparse_reward_by_agent": [20.0, 0.0], "dropped_reward_by_agent": [-4.0, 0.0]},
        )

        self.assertEqual(hidden_reward, (-4.0, -4.0))

    def test_time_cost_is_not_gated_by_order_reward_weight(self):
        env = self.make_env(time_cost=-0.3)
        vec_hidden = self.zero_hidden()

        hidden_reward = env._hsp_hidden_rewards(
            vec_hidden,
            {"sparse_reward_by_agent": [20.0, 0.0], "dropped_reward_by_agent": [0.0, 0.0]},
        )

        self.assertAlmostEqual(hidden_reward[0], -0.3)
        self.assertAlmostEqual(hidden_reward[1], -0.3)


if __name__ == "__main__":
    unittest.main()
