from __future__ import annotations

import pickle
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from transplant.tests.mep_test import eval_mep_vs_random as target


class UniformRandomPolicyTest(unittest.TestCase):
    def _actions(self, seed: int):
        policy = target.UniformRandomEvalPolicy(seed=seed)
        policy.reset(5, 2)
        policy.register_control_agent(0, 1)
        return np.concatenate(
            [policy.step(np.zeros((1, 1)), [(0, 1)]) for _ in range(100)]
        )

    def test_actions_are_uniform_policy_action_indices_and_reproducible(self):
        first = self._actions(17)
        second = self._actions(17)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all(first >= 0))
        self.assertTrue(np.all(first < 6))
        self.assertGreater(len(np.unique(first)), 1)


class RoleAndMetricTest(unittest.TestCase):
    def test_role_split_is_three_agent0_and_two_agent1(self):
        roles = [target.mep_role_for_episode(index) for index in range(5)]
        self.assertEqual(roles, ["agent0", "agent1", "agent0", "agent1", "agent0"])

    def test_episode_rows_and_summary_use_real_delivery_counts(self):
        roles = [target.mep_role_for_episode(index) for index in range(5)]
        eval_info = {
            "eval_ep_sparse_r": [20, 40, 0, 20, 20],
            "eval_ep_sparse_r_by_agent0": [20, 0, 0, 20, 20],
            "eval_ep_sparse_r_by_agent1": [0, 40, 0, 0, 0],
            "eval_ep_delivery_by_agent0": [1, 0, 0, 1, 1],
            "eval_ep_delivery_by_agent1": [0, 2, 0, 0, 0],
        }
        gifs = [Path(f"episode_{index}.gif") for index in range(5)]
        rows = target.build_episode_rows(
            policy_name="mep1",
            roles=roles,
            eval_info=eval_info,
            gif_paths=gifs,
            device_used="cpu",
        )
        summary = target.summarize_policy(rows)

        self.assertEqual(len(rows), 5)
        self.assertEqual(summary["mean_sparse_reward"], 20.0)
        self.assertEqual(summary["mean_deliveries"], 1.0)
        self.assertEqual(summary["mean_mep_delivery_credit"], 0.8)
        self.assertEqual(summary["mean_random_delivery_credit"], 0.2)

    def test_eval_arguments_force_time_cost_zero(self):
        args = target.parse_args(["--time_cost", "-9.0", "--output-root", "/tmp/mep-test"])
        self.assertEqual(args.time_cost, 0.0)
        self.assertEqual(args.episode_length, 200)
        self.assertEqual(args.n_eval_rollout_threads, 5)

    def test_missing_gif_fails_exact_count_validation(self):
        class FakeVecEnv:
            @staticmethod
            def get_last_render_gif_paths():
                return [[], [], [], [], []]

        with self.assertRaisesRegex(RuntimeError, "exactly one GIF"):
            target._gif_paths_by_env(FakeVecEnv(), 5)


class PolicyPoolPreparationTest(unittest.TestCase):
    def test_prepares_exactly_twelve_final_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_results = root / "results"
            files_dir = (
                source_results
                / "RiskyOvercooked"
                / target.LAYOUT
                / "mep"
                / "mep-S1"
                / "wandb"
                / "run-test"
                / "files"
            )
            files_dir.mkdir(parents=True)
            metadata = types.SimpleNamespace(
                layout_name=target.LAYOUT,
                episode_length=200,
                p_slip=0.4,
                time_cost=-0.3,
            )
            with (files_dir / "policy_config.pkl").open("wb") as file:
                pickle.dump((metadata, None, None, None), file)
            for index in range(1, 13):
                actor_dir = files_dir / f"mep{index}"
                actor_dir.mkdir()
                (actor_dir / f"actor_periodic_{target.FINAL_ACTOR_VERSION}.pt").write_bytes(
                    f"actor-{index}".encode()
                )

            source_config = root / "mlp_policy_config.pkl"
            with source_config.open("wb") as file:
                pickle.dump((metadata, None, None, None), file)
            pool_root = root / "policy_pool"
            population_path = target.prepare_test_policy_pool(
                source_results_root=source_results,
                source_policy_config=source_config,
                policy_pool_root=pool_root,
                run_config_dir=root / "run" / "config",
            )

            population = yaml.safe_load(population_path.read_text())
            self.assertEqual(list(population), [f"mep{index}" for index in range(1, 13)])
            self.assertEqual(
                len(list((pool_root / target.LAYOUT / "mep" / "s1").glob("*_final_actor.pt"))),
                12,
            )


class DeviceFallbackTest(unittest.TestCase):
    def test_evaluation_falls_back_to_cpu(self):
        args = types.SimpleNamespace(allow_cpu_fallback=True, cuda=True)
        cpu_device = types.SimpleNamespace(type="cpu")
        with mock.patch.object(
            target,
            "setup_device",
            side_effect=[RuntimeError("CUDA broken"), cpu_device],
        ) as setup:
            device, label, error = target.select_eval_device(args)

        self.assertIs(device, cpu_device)
        self.assertEqual(label, "cpu")
        self.assertIn("CUDA broken", error)
        self.assertFalse(args.cuda)
        self.assertEqual(setup.call_count, 2)


class WandbPayloadTest(unittest.TestCase):
    def test_uses_dedicated_wandb_project_by_default(self):
        self.assertEqual(target.DEFAULT_WANDB_PROJECT, "mep_random_test")

    def test_policy_section_contains_table_and_five_gifs(self):
        class FakeWandb:
            @staticmethod
            def Table(**kwargs):
                return ("table", kwargs)

            @staticmethod
            def Video(path, format):
                return ("video", path, format)

        rows = [
            {
                "policy": "mep1",
                "episode": index,
                "mep_role": "agent0",
                "team_sparse_reward": 0.0,
                "agent0_sparse_reward": 0.0,
                "agent1_sparse_reward": 0.0,
                "team_deliveries": 0.0,
                "mep_delivery_credit": 0.0,
                "random_delivery_credit": 0.0,
                "device": "cpu",
                "gif_path": f"episode_{index}.gif",
            }
            for index in range(1, 6)
        ]
        payload = target._wandb_payload("mep1", rows, target.summarize_policy(rows), FakeWandb)

        self.assertIn("mep1/episodes/table", payload)
        self.assertIn("mep1/summary/mean_sparse_reward", payload)
        self.assertEqual(len([key for key in payload if key.startswith("mep1/gifs/")]), 5)


if __name__ == "__main__":
    unittest.main()
