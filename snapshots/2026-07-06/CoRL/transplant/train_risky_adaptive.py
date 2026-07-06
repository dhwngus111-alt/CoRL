#!/usr/bin/env python
"""Train MEP S1 or HSP S2 adaptive policy on Risky Overcooked."""

from __future__ import annotations

import copy
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

ADAPTIVE_POLICY_OVERRIDE_FIELDS = (
    # Environment/runtime behavior.
    "env_name",
    "episode_length",
    "share_policy",
    "use_agent_policy_id",
    "predict_shaped_info_horizon",
    "predict_shaped_info_event_count",
    "shaped_info_coef",
    "policy_group_normalization",
    "num_v_out",
    "use_task_v_out",
    "use_policy_vhead",
    # Network architecture.
    "hidden_size",
    "layer_N",
    "cnn_layers_params",
    # Optimizer and PPO update hyperparameters.
    "lr",
    "critic_lr",
    "opti_eps",
    "weight_decay",
    "clip_param",
    "ppo_epoch",
    "num_mini_batch",
    "data_chunk_length",
    "policy_value_loss_coef",
    "value_loss_coef",
    "entropy_coef",
    "max_grad_norm",
    "huber_delta",
    # Return/buffer and loss flags.
    "gamma",
    "gae_lambda",
    "use_gae",
    "use_popart",
    "use_valuenorm",
    "use_proper_time_limits",
    "use_max_grad_norm",
    "use_clipped_value_loss",
    "use_huber_loss",
    "use_value_active_masks",
    "use_policy_active_masks",
)


def make_adaptive_policy_override(all_args):
    override = {
        field: getattr(all_args, field)
        for field in ADAPTIVE_POLICY_OVERRIDE_FIELDS
        if hasattr(all_args, field)
    }
    override["predict_other_shaped_info"] = False
    return Namespace(**override)


def validate_adaptive_policy_override(agent_name, expected_args, policy_args):
    mismatches = []
    for field, expected in expected_args._get_kwargs():
        actual = getattr(policy_args, field, None)
        if actual != expected:
            mismatches.append(f"{field}: expected {expected!r}, got {actual!r}")
    if mismatches:
        detail = "; ".join(mismatches)
        raise RuntimeError(f"adaptive policy override failed for {agent_name}: {detail}")


def log_adaptive_policy_config(agent_name, policy_args, trainer, run):
    fields = [
        "lr",
        "critic_lr",
        "ppo_epoch",
        "num_mini_batch",
        "clip_param",
        "gamma",
        "gae_lambda",
        "max_grad_norm",
        "value_loss_coef",
        "policy_value_loss_coef",
        "entropy_coef",
        "hidden_size",
        "layer_N",
        "cnn_layers_params",
    ]
    values = {
        field: getattr(policy_args, field)
        for field in fields
        if hasattr(policy_args, field)
    }
    actor_optimizer = getattr(trainer.policy, "actor_optimizer", None)
    critic_optimizer = getattr(trainer.policy, "critic_optimizer", None)
    if actor_optimizer is not None:
        values["actor_optimizer_lr"] = actor_optimizer.param_groups[0]["lr"]
    if critic_optimizer is not None:
        values["critic_optimizer_lr"] = critic_optimizer.param_groups[0]["lr"]
    summary = ", ".join(f"{key}={value}" for key, value in values.items())
    print(f"adaptive_policy_config {agent_name}: {summary}")
    if run is not None:
        run.config.update({f"adaptive_policy/{key}": value for key, value in values.items()}, allow_val_change=True)


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


def make_final_render_env(all_args, run_dir):
    gif_episodes = max(0, int(getattr(all_args, "mep_final_gif_episodes", 0) or 0))
    if (
        gif_episodes <= 0
        or getattr(all_args, "algorithm_name", "") != "mep"
        or int(getattr(all_args, "stage", 1)) != 1
    ):
        return None

    render_args = copy.deepcopy(all_args)
    render_args.use_render = True
    render_args.random_index = False
    render_args.n_eval_rollout_threads = gif_episodes
    render_args.render_gif_subdir = "final_mep_s1"

    def get_env_fn(rank):
        def init_env():
            env = RiskyOvercooked(render_args, run_dir, rank=rank)
            env.use_render = True
            env.seed(all_args.seed * 90000 + rank * 10000 + 991)
            return env

        return init_env

    return ChooseSubprocVecEnv([get_env_fn(i) for i in range(gif_episodes)])


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
    render_run_dir = run_dir
    if all_args.use_wandb and run is not None and getattr(run, "dir", None):
        render_run_dir = Path(run.dir)
    final_render_envs = make_final_render_env(all_args, render_run_dir)
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "final_render_envs": final_render_envs,
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
    adaptive_override_args = make_adaptive_policy_override(all_args)
    override_policy_config[agent_name] = (
        adaptive_override_args,
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
    validate_adaptive_policy_override(
        agent_name,
        adaptive_override_args,
        runner.policy.policy_config[agent_name][0],
    )
    log_adaptive_policy_config(
        agent_name,
        runner.policy.policy_config[agent_name][0],
        runner.trainer.trainer_pool[agent_name],
        run if all_args.use_wandb else None,
    )
    runner.train_mep()

    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()
    if final_render_envs is not None:
        final_render_envs.close()
    finish_logging(all_args, run, runner)


if __name__ == "__main__":
    main(sys.argv[1:])
