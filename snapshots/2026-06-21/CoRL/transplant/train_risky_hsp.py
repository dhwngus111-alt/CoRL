#!/usr/bin/env python
"""Train HSP S1 biased policies on Risky Overcooked."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transplant.bootstrap import ensure_paths

ensure_paths()

from hsp.config import get_config  # noqa: E402
from hsp.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv  # noqa: E402

from transplant.adapters.risky_overcooked_env import RiskyOvercooked  # noqa: E402
from transplant.common import (  # noqa: E402
    add_risky_overcooked_args,
    finish_logging,
    init_wandb,
    make_run_dir,
    normalize_risky_args,
    sample_hsp_weights,
    set_process_title,
    set_seeds,
    setup_device,
)


def make_train_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            env = RiskyOvercooked(all_args, run_dir)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            env = RiskyOvercooked(all_args, run_dir, rank=rank)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def make_final_render_env(all_args, run_dir):
    gif_episodes = max(0, int(getattr(all_args, "hsp_final_gif_episodes", 0) or 0))
    if gif_episodes <= 0:
        return None

    render_args = copy.deepcopy(all_args)
    render_args.use_render = True
    render_args.random_index = False
    render_args.n_rollout_threads = gif_episodes
    # RiskyOvercooked uses the eval-GIF budget as its per-env save guard.
    # Keep it aligned with the dedicated HSP final-render pool; leaving the
    # parser default (0) silently disables every GIF write.
    render_args.n_eval_rollout_threads = gif_episodes
    render_args.render_eval_gif_episodes = gif_episodes
    render_args.render_gif_subdir = "final_hsp_s1"

    def get_env_fn(rank):
        def init_env():
            env = RiskyOvercooked(render_args, run_dir, rank=rank)
            env.use_render = True
            env.seed(all_args.seed * 70000 + rank * 10000 + 777)
            return env

        return init_env

    return ShareDummyVecEnv([get_env_fn(i) for i in range(gif_episodes)])


def parse_args(args, parser):
    add_risky_overcooked_args(parser, mode="hsp")
    return normalize_risky_args(parser.parse_known_args(args)[0])


def validate_hsp_args(all_args) -> None:
    if all_args.algorithm_name in ["rmappo", "rmappg"]:
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, (
            "check recurrent policy!"
        )
    elif all_args.algorithm_name in ["mappo", "mappg"]:
        assert not all_args.use_recurrent_policy and not all_args.use_naive_recurrent_policy, (
            "check recurrent policy!"
        )
    else:
        raise NotImplementedError


def train_hsp_s1(all_args, run_dir=None, run=None, finish_wandb: bool = True, local_run_dir=None):
    validate_hsp_args(all_args)
    all_args = sample_hsp_weights(all_args)
    device = setup_device(all_args)
    if run_dir is None:
        run_dir = make_run_dir(all_args)
    active_run = run if run is not None else init_wandb(all_args, run_dir)
    set_process_title(all_args)
    set_seeds(all_args.seed)

    envs = make_train_env(all_args, run_dir)
    eval_envs = make_eval_env(all_args, run_dir) if all_args.use_eval else None
    final_render_envs = make_final_render_env(all_args, run_dir)
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "final_render_envs": final_render_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir,
    }
    if local_run_dir is not None:
        config["local_run_dir"] = local_run_dir

    if all_args.share_policy:
        from transplant.runners.risky_overcooked_runner import RiskyOvercookedRunner as Runner
    else:
        from transplant.runners.risky_overcooked_separated_runner import (
            RiskySeparatedOvercookedRunner as Runner,
        )

    runner = Runner(config)
    try:
        runner.run()
    finally:
        envs.close()
        if all_args.use_eval and eval_envs is not envs:
            eval_envs.close()
        if final_render_envs is not None:
            final_render_envs.close()
        if finish_wandb:
            finish_logging(all_args, active_run, runner)
    return runner


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    train_hsp_s1(all_args)


if __name__ == "__main__":
    main(sys.argv[1:])
