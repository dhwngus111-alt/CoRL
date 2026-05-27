#!/usr/bin/env python
"""Render a short GIF that exercises a Risky Overcooked subgoal tile."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import imageio
import numpy as np
import pygame

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transplant.bootstrap import TRANSPLANT_ROOT, ensure_paths

ensure_paths()

from risky_overcooked_py.mdp.actions import Action, Direction
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
from risky_overcooked_py.visualization.state_visualizer import StateVisualizer


MOVE_TO_DIRECTION = {
    (0, -1): Direction.NORTH,
    (0, 1): Direction.SOUTH,
    (1, 0): Direction.EAST,
    (-1, 0): Direction.WEST,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a quick GIF showing G activating/deactivating W tiles."
    )
    parser.add_argument("--layout", default="risky_multipath_subgoal")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRANSPLANT_ROOT / "results" / "subgoal_render",
    )
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--tile-size", type=int, default=70)
    parser.add_argument("--wait-steps", type=int, default=25)
    parser.add_argument("--subgoal-disable-steps", type=int, default=None)
    parser.add_argument("--p-slip", type=float, default=None)
    return parser.parse_args()


def terrain_snapshot(mdp):
    return [list(row) for row in mdp.terrain_mtx]


def surface_to_rgb_array(surface):
    buffer = pygame.surfarray.array3d(surface)
    image = np.asarray(buffer).copy()
    return np.flip(np.rot90(image, 3), 1).astype(np.uint8, copy=False)


def render_frame(visualizer, state, mdp):
    surface = visualizer.render_state(
        state=state,
        grid=terrain_snapshot(mdp),
        hud_data=None,
        mdp=mdp,
    )
    return surface_to_rgb_array(surface)


def joint_action(agent_idx, action):
    actions = [Action.STAY, Action.STAY]
    actions[agent_idx] = action
    return tuple(actions)


def subgoal_activation_actions(env, mdp):
    subgoals = sorted(mdp.terrain_pos_dict.get("G", []), key=lambda pos: (pos[1], pos[0]))
    if not subgoals:
        raise ValueError(f"Layout {mdp.layout_name!r} does not contain a G subgoal tile.")

    subgoal_pos = subgoals[0]
    for agent_idx, player in enumerate(env.state.players):
        px, py = player.position
        dx = subgoal_pos[0] - px
        dy = subgoal_pos[1] - py
        if (dx, dy) == (0, 0):
            return [joint_action(agent_idx, Action.INTERACT)]
        if (dx, dy) in MOVE_TO_DIRECTION:
            return [
                joint_action(agent_idx, MOVE_TO_DIRECTION[(dx, dy)]),
                joint_action(agent_idx, Action.INTERACT),
            ]

    starts = [player.position for player in env.state.players]
    raise ValueError(
        f"No player starts on or next to subgoal {subgoal_pos}; starts={starts}."
    )


def main():
    args = parse_args()

    mdp_kwargs = {}
    if args.subgoal_disable_steps is not None:
        mdp_kwargs["subgoal_disable_steps"] = args.subgoal_disable_steps
    if args.p_slip is not None:
        mdp_kwargs["p_slip"] = args.p_slip

    mdp = OvercookedGridworld.from_layout_name(args.layout, **mdp_kwargs)
    env = OvercookedEnv.from_mdp(mdp, horizon=args.wait_steps + 8, info_level=0)
    visualizer = StateVisualizer(tile_size=args.tile_size, is_rendering_hud=False)

    frames = []

    def record(repeat=1):
        frame = render_frame(visualizer, env.state, mdp)
        frames.extend([frame] * repeat)

    record(repeat=4)

    actions = subgoal_activation_actions(env, mdp)
    for idx, action in enumerate(actions):
        env.step(action)
        record(repeat=6 if idx == len(actions) - 1 else 3)

    for _ in range(args.wait_steps):
        env.step((Action.STAY, Action.STAY))
        record(repeat=1)

    output_dir = args.output_dir / "gifs" / args.layout
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "subgoal_toggle.gif"
    imageio.mimsave(output_path, frames, duration=1.0 / args.fps, loop=0)

    print(output_path)


if __name__ == "__main__":
    main()
