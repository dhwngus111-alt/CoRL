from __future__ import annotations

import importlib.util
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from transplant.bootstrap import ensure_paths


ensure_paths()

missing_modules = [
    module_name
    for module_name in ("numpy", "gym", "cv2", "imageio", "pygame")
    if importlib.util.find_spec(module_name) is None
]

if not missing_modules:
    import numpy as np

    from transplant.adapters.risky_overcooked_env import RiskyOvercooked


def make_args(*, episode_length: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        num_agents=2,
        layout_name="risky_multipath_subgoal",
        episode_length=episode_length,
        use_render=True,
        random_start_prob=0.0,
        time_cost=0.0,
        p_slip=0.4,
        subgoal_disable_steps=60,
    )


@unittest.skipIf(missing_modules, f"Missing runtime dependencies: {missing_modules}")
class RiskyGifTimelineTest(unittest.TestCase):
    marker_color = np.asarray([40, 110, 230], dtype=np.uint8) if not missing_modules else None

    @staticmethod
    def action(agent0: int, agent1: int):
        return [[agent0], [agent1]]

    @staticmethod
    def marker_pixel_count(image) -> int:
        return int(np.all(image == RiskyGifTimelineTest.marker_color, axis=2).sum())

    @staticmethod
    def water_tiles(env: RiskyOvercooked) -> list[str]:
        return [
            env.base_mdp.terrain_mtx[y][x]
            for x, y in sorted(env.base_mdp.water_disable_timers)
        ]

    def make_env(self, root: Path, *, episode_length: int) -> RiskyOvercooked:
        return RiskyOvercooked(
            make_args(episode_length=episode_length),
            root,
            rank=0,
        )

    def test_step_zero_and_historical_timers_render_without_mutating_live_mdp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = self.make_env(Path(tmpdir), episode_length=100)
            env.reset()

            frame_keys = tuple(env.render_traj)
            self.assertTrue(all(len(env.render_traj[key]) == 1 for key in frame_keys))
            self.assertIsNone(env.render_traj["actions"][0])
            self.assertEqual(env.render_traj["infos"][0], {"initial_state": True})
            self.assertEqual(set(env.render_traj["water_disable_timers"][0].values()), {0})
            self.assertEqual(
                [env.render_traj["grids"][0][y][x] for x, y in sorted(env.base_mdp.water_disable_timers)],
                ["W", "W", "W"],
            )

            # Player 1 starts immediately left of G: move onto G, then interact.
            env.step(self.action(4, 2))
            env.step(self.action(4, 5))
            disabled_timers = dict(env.base_mdp.water_disable_timers)
            disabled_grid = [list(row) for row in env.base_mdp.terrain_mtx]
            self.assertEqual(set(disabled_timers.values()), {59})
            self.assertEqual(self.water_tiles(env), [" ", " ", " "])

            initial_image = env._render_frame(
                state=env.render_traj["states"][0],
                grid=env.render_traj["grids"][0],
                hud_data=env.render_traj["hud_data"][0],
                water_disable_timers=env.render_traj["water_disable_timers"][0],
            )
            disabled_image = env._render_frame(
                state=env.render_traj["states"][2],
                grid=env.render_traj["grids"][2],
                hud_data=env.render_traj["hud_data"][2],
                water_disable_timers=env.render_traj["water_disable_timers"][2],
            )

            self.assertEqual(self.marker_pixel_count(initial_image), 0)
            self.assertGreater(self.marker_pixel_count(disabled_image), 0)
            self.assertEqual(env.base_mdp.water_disable_timers, disabled_timers)
            self.assertEqual(env.base_mdp.terrain_mtx, disabled_grid)

            for _ in range(59):
                env.step(self.action(4, 4))

            active_timers = dict(env.base_mdp.water_disable_timers)
            active_grid = [list(row) for row in env.base_mdp.terrain_mtx]
            self.assertEqual(set(active_timers.values()), {0})
            self.assertEqual(self.water_tiles(env), ["W", "W", "W"])

            historical_disabled_image = env._render_frame(
                state=env.render_traj["states"][2],
                grid=env.render_traj["grids"][2],
                hud_data=env.render_traj["hud_data"][2],
                water_disable_timers=env.render_traj["water_disable_timers"][2],
            )
            self.assertGreater(self.marker_pixel_count(historical_disabled_image), 0)
            self.assertEqual(env.base_mdp.water_disable_timers, active_timers)
            self.assertEqual(env.base_mdp.terrain_mtx, active_grid)

    def test_episode_ending_disabled_does_not_contaminate_initial_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = self.make_env(Path(tmpdir), episode_length=3)
            env.reset()

            with mock.patch(
                "transplant.adapters.risky_overcooked_env.imageio.mimsave"
            ) as mimsave:
                env.step(self.action(4, 2))
                env.step(self.action(4, 5))
                env.step(self.action(4, 4))

            frames = mimsave.call_args.args[1]
            self.assertEqual(len(frames), 4)
            self.assertEqual(self.marker_pixel_count(frames[0]), 0)
            self.assertEqual(self.marker_pixel_count(frames[1]), 0)
            self.assertGreater(self.marker_pixel_count(frames[2]), 0)
            self.assertGreater(self.marker_pixel_count(frames[3]), 0)
            self.assertEqual(set(env.base_mdp.water_disable_timers.values()), {58})

    def test_full_episode_writes_201_aligned_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = self.make_env(Path(tmpdir), episode_length=200)
            env.reset()
            fake_frame = np.zeros((2, 2, 3), dtype=np.uint8)

            with mock.patch.object(env, "_render_frame", return_value=fake_frame) as render_frame, mock.patch(
                "transplant.adapters.risky_overcooked_env.imageio.mimsave"
            ) as mimsave:
                for _ in range(200):
                    env.step(self.action(4, 4))

            self.assertEqual({key: len(values) for key, values in env.render_traj.items()}, {
                "states": 201,
                "grids": 201,
                "hud_data": 201,
                "actions": 201,
                "rewards": 201,
                "dones": 201,
                "infos": 201,
                "water_disable_timers": 201,
            })
            self.assertEqual(render_frame.call_count, 201)
            self.assertEqual(len(mimsave.call_args.args[1]), 201)
            self.assertEqual(
                set(render_frame.call_args_list[0].kwargs["water_disable_timers"].values()),
                {0},
            )

    def test_misaligned_render_timeline_fails_before_encoding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = self.make_env(Path(tmpdir), episode_length=10)
            env.reset()
            env.render_traj["water_disable_timers"].clear()

            with mock.patch(
                "transplant.adapters.risky_overcooked_env.imageio.mimsave"
            ) as mimsave:
                with self.assertRaisesRegex(RuntimeError, "misaligned"):
                    env._save_render_gif({})

            mimsave.assert_not_called()


if __name__ == "__main__":
    unittest.main()
