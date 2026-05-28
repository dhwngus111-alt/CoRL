import argparse
import copy
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pygame
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from risky_overcooked_py.agents.agent import AgentPair, RandomAgent
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
from risky_overcooked_py.visualization.state_visualizer import StateVisualizer


def render_frame(env: OvercookedEnv, visualizer: StateVisualizer) -> np.ndarray:
    rewards_dict = {
        "cumulative_shaped_rewards_by_agent": env.game_stats["cumulative_shaped_rewards_by_agent"],
        "cumulative_sparse_rewards_by_agent": env.game_stats["cumulative_sparse_rewards_by_agent"],
    }
    surface = visualizer.render_state(
        state=env.state,
        grid=env.mdp.terrain_mtx,
        hud_data=StateVisualizer.default_hud_data(env.state, **rewards_dict),
    )

    buffer = pygame.surfarray.array3d(surface)
    image = copy.deepcopy(buffer)
    image = np.flip(np.rot90(image, 3), 1)
    target_size = (2 * 528, 2 * 464)
    if cv2 is not None:
        image = cv2.resize(image, target_size)
    else:
        image = np.array(Image.fromarray(image).resize(target_size, Image.NEAREST))
    return image


def run_random_episode(
    layout_name: str = "risky_coordination_ring",
    horizon: int = 200,
    p_slip: float = 0.4,
    seed: int = 0,
    all_actions: bool = True,
    render: bool = False,
    fps: int = 8,
    save_gif: Optional[str] = "testing/outputs/random_smoke.gif",
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    mdp = OvercookedGridworld.from_layout_name(layout_name, p_slip=p_slip)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)

    agent_pair = AgentPair(
        RandomAgent(all_actions=all_actions),
        RandomAgent(all_actions=all_actions),
    )
    agent_pair.set_mdp(env.mdp)

    env.reset(regen_mdp=False)
    agent_pair.reset()
    visualizer = StateVisualizer() if (render or save_gif is not None) else None
    window_name = "Risky Overcooked RandomAgent"
    frames = []

    if render and os.environ.get("DISPLAY") is None:
        print("DISPLAY not found. Disabling live window render and continuing in headless mode.")
        render = False

    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        if visualizer is not None:
            frame = render_frame(env, visualizer)
            if save_gif is not None:
                frames.append(frame)
            if render:
                if cv2 is None:
                    print("cv2 not installed. Disabling live window render; GIF recording continues if enabled.")
                    render = False
                    continue
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(max(1, int(1000 / max(1, fps)))) & 0xFF
                if key == ord("q"):
                    print("render interrupted by user (q)")
                    break

        joint_action_and_infos = agent_pair.joint_action(env.state)
        joint_action, joint_action_info = zip(*joint_action_and_infos)

        _, reward, done, info = env.step(joint_action, joint_action_info)
        total_reward += reward
        steps += 1

    if render and visualizer is not None and cv2 is not None:
        frame = render_frame(env, visualizer)
        cv2.imshow(window_name, frame)
        cv2.waitKey(300)
        cv2.destroyAllWindows()
        time.sleep(0.05)

    if save_gif is not None:
        # The loop records pre-step states; append terminal state so the last timestep is visible.
        if visualizer is not None:
            frames.append(render_frame(env, visualizer))
        out_path = Path(save_gif)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out_path, frames, fps=max(1, fps))
        print(f"saved_gif={out_path} (frames={len(frames)})")

    if "episode" not in info:
        print("Episode finished early without terminal stats (likely manual stop).")
        return

    ep = info["episode"]
    sparse_total = ep["ep_sparse_r"]
    shaped_total = ep["ep_shaped_r"]

    print("=== Random Agent Smoke Test ===")
    print(f"layout={layout_name}, p_slip={p_slip}, horizon={horizon}, seed={seed}")
    print(f"steps={steps}")
    print(f"total_reward(step reward sum)={total_reward}")
    print(f"episode_sparse_reward={sparse_total}")
    print(f"episode_shaped_reward={shaped_total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run random-agent smoke test for Risky Overcooked env")
    parser.add_argument("--layout", type=str, default="risky_coordination_ring")
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--p-slip", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--motion-only", action="store_true", help="Use only motion actions (no interact action)")
    parser.add_argument("--render", action="store_true", help="Render real-time window (press q to stop)")
    parser.add_argument("--fps", type=int, default=8, help="Render FPS when --render is enabled")
    parser.add_argument(
        "--save-gif",
        type=str,
        default="testing/outputs/random_smoke.gif",
        help="Save rollout as GIF (works on headless remote)",
    )
    args = parser.parse_args()

    run_random_episode(
        layout_name=args.layout,
        horizon=args.horizon,
        p_slip=args.p_slip,
        seed=args.seed,
        all_actions=not args.motion_only,
        render=args.render,
        fps=args.fps,
        save_gif=args.save_gif,
    )
