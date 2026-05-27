#!/usr/bin/env python
"""Train MEP S1 or HSP S2 adaptive policy on Risky Overcooked."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transplant.bootstrap import ensure_paths

ensure_paths()

from hsp.config import get_config  # noqa: E402
from hsp.envs.env_wrappers import (  # noqa: E402
    ChooseDummyVecEnv,
    ChooseSubprocVecEnv,
    ShareDummyVecEnv,
    ShareSubprocVecEnv,
)
from hsp.envs.wrappers.env_policy import PartialPolicyEnv  # noqa: E402

from transplant.adapters.risky_overcooked_env import RiskyOvercooked  # noqa: E402
from transplant.common import (  # noqa: E402
    add_risky_overcooked_args,
    finish_logging,
    init_wandb,
    make_run_dir,
    normalize_risky_args,
    set_process_title,
    set_seeds,
    setup_device,
)


def make_train_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            env = RiskyOvercooked(all_args, run_dir)
            env = PartialPolicyEnv(all_args, env)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    # HSP's population runner calls envs.load_policy(), which only exists on
    # ShareSubprocVecEnv in the original wrapper implementation.
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            env = RiskyOvercooked(all_args, run_dir, rank=rank)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ChooseDummyVecEnv([get_env_fn(0)])
    return ChooseSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    add_risky_overcooked_args(parser, mode="adaptive")
    return normalize_risky_args(parser.parse_known_args(args)[0])


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    assert all_args.algorithm_name in ["mep", "adaptive"]
    all_args.wandb_job_type = "training"

    device = setup_device(all_args)
    run_dir = make_run_dir(all_args)
    set_process_title(all_args)
    set_seeds(all_args.seed)

    envs = make_train_env(all_args, run_dir)
    eval_envs = make_eval_env(all_args, run_dir) if all_args.use_eval else None
    run = init_wandb(all_args, run_dir)
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    if all_args.share_policy:
        from transplant.runners.risky_overcooked_runner import RiskyOvercookedRunner as Runner
    else:
        from hsp.runner.separated.overcooked_runner import MPERunner as Runner

    runner = Runner(config)

    population_config = yaml.load(open(all_args.population_yaml_path), yaml.Loader)
    override_policy_config = {}
    agent_name = all_args.adaptive_agent_name
    override_policy_config[agent_name] = (
        Namespace(
            env_name=all_args.env_name,
            use_agent_policy_id=all_args.use_agent_policy_id,
            predict_other_shaped_info=False,
            predict_shaped_info_horizon=all_args.predict_shaped_info_horizon,
            predict_shaped_info_event_count=all_args.predict_shaped_info_event_count,
            shaped_info_coef=all_args.shaped_info_coef,
            policy_group_normalization=all_args.policy_group_normalization,
            num_v_out=all_args.num_v_out,
            use_task_v_out=all_args.use_task_v_out,
            use_policy_vhead=all_args.use_policy_vhead,
        ),
        *runner.policy_config[1:],
    )
    for policy_name in population_config:
        if policy_name != agent_name:
            override_policy_config[policy_name] = (None, None, runner.policy_config[2], None)

    runner.policy.load_population(
        all_args.population_yaml_path,
        evaluation=False,
        override_policy_config=override_policy_config,
    )
    runner.trainer.init_population()
    runner.train_mep()

    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()
    finish_logging(all_args, run, runner)


if __name__ == "__main__":
    main(sys.argv[1:])
