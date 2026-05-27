#!/usr/bin/env python
"""Import and env smoke check for the Risky HSP transplant."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transplant.bootstrap import PATHS, TRANSPLANT_ROOT, ensure_paths

ensure_paths()

from hsp.config import get_config  # noqa: E402

from transplant.adapters.risky_overcooked_env import CATEGORY_KEYS, RiskyOvercooked  # noqa: E402
from transplant.common import RISKY_ENV_NAME, add_risky_overcooked_args, normalize_risky_args  # noqa: E402


def main():
    print("bootstrap:")
    for key, value in PATHS.items():
        print(f"  {key}: {value}")

    parser = get_config()
    add_risky_overcooked_args(parser, mode="hsp")
    all_args = parser.parse_known_args(
        [
            "--env_name",
            RISKY_ENV_NAME,
            "--algorithm_name",
            "mappo",
            "--experiment_name",
            "smoke",
            "--layout_name",
            "risky_multipath",
            "--num_agents",
            "2",
            "--episode_length",
            "20",
            "--overcooked_version",
            "risky",
            "--use_recurrent_policy",
            "--use_wandb",
        ]
    )[0]
    all_args = normalize_risky_args(all_args)
    env = RiskyOvercooked(all_args, TRANSPLANT_ROOT / "results" / "smoke")
    obs, share_obs, available_actions = env.reset()
    print("reset:")
    print(f"  obs: {[x.shape for x in obs]}")
    print(f"  share_obs: {share_obs.shape}")
    print(f"  available_actions: {available_actions.shape}, {available_actions.dtype}")

    frame = env.render(mode="rgb_array")
    print(
        "  render rgb_array: "
        f"shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}"
    )
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise AssertionError(f"Expected RGB frame, got shape {frame.shape}")
    if frame.dtype != np.uint8:
        raise AssertionError(f"Expected uint8 frame, got {frame.dtype}")
    if not np.any(frame):
        raise AssertionError("render('rgb_array') returned a blank frame")

    info = {}
    for step in range(5):
        action = np.random.randint(0, 6, size=(2, 1))
        obs, share_obs, reward, done, info, available_actions = env.step(action)
        print(f"  step {step + 1}: action={action.reshape(-1).tolist()} reward={reward} done={done}")
        if all(done):
            break

    print("last info keys:", sorted(info.keys()))
    print(f"  risky event dim: {len(CATEGORY_KEYS)}")
    if info.get("vec_shaped_info_by_agent") is not None:
        print(f"  vec_shaped_info_by_agent: {info['vec_shaped_info_by_agent'].shape}")
    if "episode" in info:
        print("episode keys:", sorted(info["episode"].keys()))
    env.close()

    render_args = parser.parse_known_args(
        [
            "--env_name",
            RISKY_ENV_NAME,
            "--algorithm_name",
            "mappo",
            "--experiment_name",
            "smoke-render",
            "--layout_name",
            "risky_multipath",
            "--num_agents",
            "2",
            "--episode_length",
            "2",
            "--overcooked_version",
            "risky",
            "--use_recurrent_policy",
            "--use_wandb",
            "--use_render",
        ]
    )[0]
    render_args = normalize_risky_args(render_args)
    render_env = RiskyOvercooked(render_args, TRANSPLANT_ROOT / "results" / "smoke_render")
    render_env.reset()
    for _ in range(render_args.episode_length):
        action = np.zeros((2, 1), dtype=np.int64)
        _, _, _, done, _, _ = render_env.step(action)
        if all(done):
            break
    gif_path = render_env.last_render_gif_path
    render_env.close()
    if gif_path is None or not gif_path.exists():
        raise AssertionError("use_render smoke did not create a GIF")
    print(f"  render gif: {gif_path}")
    print("smoke check complete")


if __name__ == "__main__":
    main()
