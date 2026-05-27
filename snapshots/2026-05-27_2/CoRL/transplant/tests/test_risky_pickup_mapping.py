from __future__ import annotations

import unittest
import importlib.util
import sys
import types

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
    )
    from transplant.hsp_hidden_utility import (
        HIDDEN_UTILITY_KEYS,
        ordered_zero_category_dict,
    )

    from risky_overcooked_py.mdp.actions import Action, Direction
    from risky_overcooked_py.mdp.overcooked_mdp import (
        ObjectState,
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
