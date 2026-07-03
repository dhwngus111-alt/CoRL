#!/usr/bin/env python
"""Train a CoMeDi convention population on Risky Overcooked."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

from comedi_transplant.bootstrap import COMEDI_TRANSPLANT_ROOT, POLICY_POOL_ROOT, ensure_paths
from comedi_transplant.comedi_trainer import CoMeDiPPOTrainer, collect_rollout
from comedi_transplant.policy_configs import (
    DEFAULT_CNN_LAYERS,
    build_policy_configs,
    write_population_yamls,
)


ensure_paths()

from hsp.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy  # noqa: E402
from hsp.config import get_config  # noqa: E402
from hsp.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv  # noqa: E402
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


def _add_comedi_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--comedi_population_size", type=int, default=8)
    parser.add_argument("--comedi_alpha", type=float, default=0.5)
    parser.add_argument("--comedi_beta", type=float, default=1.0)
    parser.add_argument("--comedi_select_interval", type=int, default=1)
    parser.add_argument("--comedi_eval_episodes", type=int, default=1)
    parser.add_argument("--comedi_save_every", type=int, default=0)
    parser.add_argument("--comedi_skip_policy_config", action="store_true")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = get_config()
    add_risky_overcooked_args(parser, mode="hsp")
    _add_comedi_args(parser)
    parser.set_defaults(
        algorithm_name="mappo",
        experiment_name="comedi-S1",
        n_rollout_threads=50,
        episode_length=200,
        num_env_steps=1_000_000,
        ppo_epoch=10,
        num_mini_batch=1,
        lr=1e-2,
        critic_lr=1e-2,
        use_linear_lr_decay=True,
        entropy_coef=0.0,
        hidden_size=64,
        layer_N=2,
        activation_id=1,
        cnn_layers_params=DEFAULT_CNN_LAYERS,
        use_recurrent_policy=False,
        use_naive_recurrent_policy=False,
    )
    all_args = parser.parse_known_args(argv)[0]
    all_args = normalize_risky_args(all_args)
    if all_args.algorithm_name != "mappo":
        raise ValueError("CoMeDi population v1 uses mappo/MLP policies")
    return all_args


def make_envs(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            env = RiskyOvercooked(all_args, run_dir, rank=rank)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_policy(all_args, envs, device, actor_path: Path | None = None, critic_path: Path | None = None):
    share_obs_space = (
        envs.share_observation_space[0]
        if all_args.use_centralized_V
        else envs.observation_space[0]
    )
    policy = R_MAPPOPolicy(
        all_args,
        envs.observation_space[0],
        share_obs_space,
        envs.action_space[0],
        device=device,
    )
    ckpt = {}
    if actor_path is not None:
        ckpt["actor"] = str(actor_path)
    if critic_path is not None:
        ckpt["critic"] = str(critic_path)
    if ckpt:
        policy.load_checkpoint(ckpt)
    return policy


def save_policy(policy: R_MAPPOPolicy, path_root: Path, name: str) -> tuple[Path, Path]:
    path_root.mkdir(parents=True, exist_ok=True)
    actor_path = path_root / f"{name}_actor.pt"
    critic_path = path_root / f"{name}_critic.pt"
    torch.save(policy.actor.state_dict(), actor_path)
    torch.save(policy.critic.state_dict(), critic_path)
    return actor_path, critic_path


def _select_partner(
    all_args,
    envs,
    current_policy,
    prior_policies: list[tuple[str, R_MAPPOPolicy]],
    rng: np.random.Generator,
) -> tuple[str, R_MAPPOPolicy, float]:
    best_name = prior_policies[0][0]
    best_policy = prior_policies[0][1]
    best_score = -float("inf")
    for name, policy in prior_policies:
        scores = []
        for _ in range(max(1, int(all_args.comedi_eval_episodes))):
            rollout = collect_rollout(
                all_args,
                envs,
                current_policy,
                mode="xp",
                partner_policy=policy,
                partner_name=name,
                rng=rng,
            )
            scores.append(rollout.mean_sparse)
        score = float(np.mean(scores))
        if score > best_score:
            best_name, best_policy, best_score = name, policy, score
    return best_name, best_policy, best_score


def _log(run, payload: dict, step: int) -> None:
    if run is None:
        return
    try:
        import wandb

        wandb.log(payload, step=step)
    except Exception:
        pass


def train_population(all_args) -> None:
    if not all_args.comedi_skip_policy_config:
        build_policy_configs(all_args.layout_name, all_args.episode_length, POLICY_POOL_ROOT)

    device = setup_device(all_args)
    run_dir = make_run_dir(all_args)
    set_process_title(all_args)
    set_seeds(all_args.seed)
    run = init_wandb(all_args, run_dir)
    envs = make_envs(all_args, run_dir)
    save_root = POLICY_POOL_ROOT / all_args.layout_name / "comedi" / "s1"
    prior_policies: list[tuple[str, R_MAPPOPolicy]] = []
    rng = np.random.default_rng(all_args.seed)

    try:
        episodes = int(all_args.num_env_steps) // all_args.episode_length // all_args.n_rollout_threads
        episodes = max(1, episodes)
        global_step = 0
        for idx in range(1, int(all_args.comedi_population_size) + 1):
            convention_name = f"comedi{idx}"
            print(f"=== training {convention_name}/{all_args.comedi_population_size} ===")
            current_policy = make_policy(all_args, envs, device)
            trainer = CoMeDiPPOTrainer(all_args, current_policy, device=device)
            active_partner_name = ""
            active_partner_policy = None
            active_partner_score = 0.0
            start = time.time()

            for episode in range(episodes):
                if all_args.use_linear_lr_decay:
                    current_policy.lr_decay(episode, episodes)

                if prior_policies and (
                    active_partner_policy is None
                    or episode % max(1, int(all_args.comedi_select_interval)) == 0
                ):
                    active_partner_name, active_partner_policy, active_partner_score = _select_partner(
                        all_args, envs, current_policy, prior_policies, rng
                    )

                sp_rollout = collect_rollout(
                    all_args, envs, current_policy, mode="sp", rng=rng
                )
                weighted_rollouts = [("sp", sp_rollout.buffer, 1.0)]
                logs = {
                    f"{convention_name}/sp_mean_reward": sp_rollout.mean_reward,
                    f"{convention_name}/sp_sparse": sp_rollout.mean_sparse,
                }

                if active_partner_policy is not None:
                    xp_rollout = collect_rollout(
                        all_args,
                        envs,
                        current_policy,
                        mode="xp",
                        partner_policy=active_partner_policy,
                        partner_name=active_partner_name,
                        rng=rng,
                    )
                    mp_rollout = collect_rollout(
                        all_args,
                        envs,
                        current_policy,
                        mode="mp",
                        partner_policy=active_partner_policy,
                        partner_name=active_partner_name,
                        rng=rng,
                    )
                    weighted_rollouts.append(("xp", xp_rollout.buffer, -all_args.comedi_alpha))
                    weighted_rollouts.append(("mp", mp_rollout.buffer, all_args.comedi_beta))
                    logs.update(
                        {
                            f"{convention_name}/partner": active_partner_name,
                            f"{convention_name}/partner_selection_sparse": active_partner_score,
                            f"{convention_name}/xp_sparse": xp_rollout.mean_sparse,
                            f"{convention_name}/mp_sparse": mp_rollout.mean_sparse,
                            f"{convention_name}/alpha": all_args.comedi_alpha,
                            f"{convention_name}/beta": all_args.comedi_beta,
                        }
                    )

                train_info = trainer.train(weighted_rollouts)
                global_step += all_args.episode_length * all_args.n_rollout_threads
                logs.update({f"{convention_name}/train/{k}": v for k, v in train_info.items()})
                _log(run, logs, global_step)

                if episode % all_args.log_interval == 0:
                    fps = int(global_step / max(time.time() - start, 1e-6))
                    print(
                        f"{all_args.layout_name} {convention_name} episode "
                        f"{episode + 1}/{episodes} step {global_step} FPS {fps} "
                        f"sp={sp_rollout.mean_sparse:.3f} partner={active_partner_name or '-'}"
                    )

                if all_args.comedi_save_every and (episode + 1) % all_args.comedi_save_every == 0:
                    save_policy(current_policy, save_root, f"{convention_name}_periodic_{episode + 1}")

            actor_path, critic_path = save_policy(current_policy, save_root, convention_name)
            print(f"saved {convention_name}: {actor_path}, {critic_path}")
            frozen_policy = make_policy(all_args, envs, device, actor_path, critic_path)
            frozen_policy.to(device)
            prior_policies.append((convention_name, frozen_policy))

        write_population_yamls(all_args.layout_name, int(all_args.comedi_population_size))
    finally:
        envs.close()
        class _RunnerShim:
            pass

        runner_shim = _RunnerShim()
        runner_shim.save_dir = str(save_root)
        runner_shim.run_dir = str(run_dir)
        finish_logging(all_args, run, runner_shim)


def main(argv: list[str] | None = None) -> None:
    all_args = parse_args(sys.argv[1:] if argv is None else argv)
    train_population(all_args)


if __name__ == "__main__":
    main()
