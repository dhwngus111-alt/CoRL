#!/usr/bin/env python
"""Train a CoMeDi convention population on Risky Overcooked."""

from __future__ import annotations

import argparse
import copy
import sys
import time
import copy
import csv
from pathlib import Path

import numpy as np
import torch

from comedi_transplant.bootstrap import COMEDI_TRANSPLANT_ROOT, POLICY_POOL_ROOT, ensure_paths
from comedi_transplant.comedi_trainer import CoMeDiPPOTrainer, collect_rollout, _empty_rnn, _t2n
from comedi_transplant.mc_policy import CoMeDiMCPolicy
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
    parser.add_argument("--comedi_mix_prob", type=float, default=0.5)  # 원본 mix_prob
    # MP 전용 env 슬롯 수. 원본 collect_mp_episode는 envs_mp=episode_length-1개 슬롯으로 굴려
    # switch 시점 1..episode_length-1을 전부 커버한다(n_rollout_threads와 분리). 0이면 자동으로
    # episode_length-1을 쓴다. 자원이 빠듯하면(예: 스모크) 낮춰서 switch 커버 밀도만 줄인다.
    parser.add_argument("--comedi_mp_threads", type=int, default=0)
    parser.add_argument("--comedi_select_interval", type=int, default=1)
    parser.add_argument("--comedi_eval_episodes", type=int, default=1)
    parser.add_argument("--comedi_save_every", type=int, default=0)
    # convention 학습 종료 시 SP로 재생해 만들 GIF 개수(해당 convention wandb 섹션에 로깅).
    parser.add_argument("--comedi_final_gif_episodes", type=int, default=3)
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


def make_envs(all_args, run_dir, n_threads=None):
    n = int(all_args.n_rollout_threads if n_threads is None else n_threads)

    def get_env_fn(rank):
        def init_env():
            env = RiskyOvercooked(all_args, run_dir, rank=rank)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if n == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(n)])


def make_policy(
    all_args,
    envs,
    device,
    actor_path: Path | None = None,
    critic_path: Path | None = None,
    multi_critic: bool = False,
    num_priors: int = 0,
):
    share_obs_space = (
        envs.share_observation_space[0]
        if all_args.use_centralized_V
        else envs.observation_space[0]
    )
    # 학습 중인 convention은 목적별 critic(SP/MP + prior별 XP0/XP1)을 갖는 CoMeDiMCPolicy,
    # 고정된 prior convention(파트너)은 actor만 쓰므로 기본 R_MAPPOPolicy.
    if multi_critic:
        policy = CoMeDiMCPolicy(
            all_args,
            envs.observation_space[0],
            share_obs_space,
            envs.action_space[0],
            device=device,
            num_priors=num_priors,
        )
    else:
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
) -> tuple[int, str, R_MAPPOPolicy, float]:
    """원본 get_best와 동일하게 XP 점수를 양방향(seat0+seat1) 합으로 best_i를 고른다."""
    best_idx = 0
    best_name = prior_policies[0][0]
    best_policy = prior_policies[0][1]
    best_score = -float("inf")
    for i, (name, policy) in enumerate(prior_policies):
        scores = []
        for _ in range(max(1, int(all_args.comedi_eval_episodes))):
            xp0 = collect_rollout(
                all_args, envs, current_policy, mode="xp0",
                partner_policy=policy, partner_name=name, rng=rng,
            )
            xp1 = collect_rollout(
                all_args, envs, current_policy, mode="xp1",
                partner_policy=policy, partner_name=name, rng=rng,
            )
            # 원본 get_best와 동일하게 학습 reward(shaped 포함) 평균으로 best_i를 고른다.
            scores.append(xp0.mean_reward + xp1.mean_reward)
        score = float(np.mean(scores))
        if score > best_score:
            best_idx, best_name, best_policy, best_score = i, name, policy, score
    return best_idx, best_name, best_policy, best_score


def _log(run, payload: dict, step: int) -> None:
    if run is None:
        return
    try:
        import wandb

        wandb.log(payload, step=step)
    except Exception:
        pass


def _write_constants_csv(save_root, convention_name, all_args) -> Path:
    """상수 하이퍼파라미터(alpha/beta/weight)를 wandb 그래프 대신 CSV로 기록(요청2)."""
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    csv_path = save_root / "constants.csv"
    alpha = float(all_args.comedi_alpha)
    beta = float(all_args.comedi_beta)
    row = {
        "convention": convention_name,
        "alpha": alpha,
        "beta": beta,
        "sp_weight": 1.0,
        "xp0_weight": -alpha / 2.0,
        "xp1_weight": -alpha / 2.0,
        "mp_weight": beta,
    }
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return csv_path


def _render_sp_gifs(all_args, run_dir, convention_name, current_policy, run, global_step) -> None:
    """convention 학습 종료 후 SP(양쪽 seat=ego)로 재생해 GIF n개 생성 → 해당 convention 섹션에 로깅(요청1)."""
    gif_episodes = max(0, int(getattr(all_args, "comedi_final_gif_episodes", 3) or 0))
    if gif_episodes <= 0:
        return

    render_args = copy.deepcopy(all_args)
    render_args.use_render = True
    render_args.random_index = False
    render_args.n_rollout_threads = gif_episodes
    render_args.n_eval_rollout_threads = gif_episodes
    render_args.render_eval_gif_episodes = gif_episodes  # RiskyOvercooked의 per-env GIF 저장 가드
    render_args.render_gif_subdir = f"comedi_s1/{convention_name}"

    def get_env_fn(rank):
        def init_env():
            env = RiskyOvercooked(render_args, run_dir, rank=rank)
            env.use_render = True
            env.seed(all_args.seed * 70000 + rank * 10000 + 777)
            return env

        return init_env

    render_envs = ShareDummyVecEnv([get_env_fn(i) for i in range(gif_episodes)])
    try:
        current_policy.prep_rollout()
        obs, _, _ = render_envs.reset()
        obs = np.stack(obs)
        rnn = _empty_rnn(all_args, gif_episodes)
        masks = np.ones((gif_episodes, all_args.num_agents, 1), dtype=np.float32)
        for _ in range(int(all_args.episode_length)):
            with torch.no_grad():
                action, rnn_out = current_policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn),
                    np.concatenate(masks),
                    deterministic=True,
                )
            actions = np.array(np.split(_t2n(action), gif_episodes))
            rnn = np.array(np.split(_t2n(rnn_out), gif_episodes))
            obs, _, _, dones, _, _ = render_envs.step(actions)
            obs = np.stack(obs)
            dones = np.asarray(dones)
            rnn[dones == True] = 0.0
            masks = np.ones((gif_episodes, all_args.num_agents, 1), dtype=np.float32)
            masks[dones == True] = 0.0

        gif_paths = []
        if hasattr(render_envs, "get_last_render_gif_paths"):
            for raw in render_envs.get_last_render_gif_paths() or []:
                if isinstance(raw, (list, tuple)):
                    gif_paths.extend(Path(p) for p in raw if p)
                elif raw:
                    gif_paths.append(Path(raw))
            gif_paths = [p for p in gif_paths if p.exists()]
        print(f"{convention_name}: generated {len(gif_paths)} SP GIF(s)")

        if run is not None and getattr(all_args, "use_wandb", False) and gif_paths:
            import wandb

            payload = {}
            for i, gif_path in enumerate(gif_paths[:gif_episodes], start=1):
                payload[f"{convention_name}/sp_gif_{i}"] = wandb.Video(str(gif_path), format="gif")
            wandb.log(payload, step=int(global_step))
    finally:
        render_envs.close()


def train_population(all_args) -> None:
    if not all_args.comedi_skip_policy_config:
        build_policy_configs(all_args.layout_name, all_args.episode_length, POLICY_POOL_ROOT)

    device = setup_device(all_args)
    run_dir = make_run_dir(all_args)
    set_process_title(all_args)
    set_seeds(all_args.seed)
    run = init_wandb(all_args, run_dir)
    envs = make_envs(all_args, run_dir)
    # MP 전용 env: 원본 collect_mp_episode처럼 switch 시점 1..episode_length-1을 전부 커버하도록
    # episode_length-1개 슬롯으로 굴린다(SP/XP는 n_rollout_threads 유지). MP는 comedi2+에서만
    # 쓰이므로 첫 사용 시 지연 생성한다(comedi1 학습 동안 유휴 프로세스 방지).
    mp_threads = max(1, int(all_args.comedi_mp_threads) or (int(all_args.episode_length) - 1))
    mp_envs = None
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
            # 이 convention을 학습할 시점의 prior 개수만큼 XP critic(seat0/seat1)을 만든다.
            current_policy = make_policy(
                all_args, envs, device, multi_critic=True, num_priors=len(prior_policies)
            )
            trainer = CoMeDiPPOTrainer(all_args, current_policy, device=device)
            active_partner_idx = 0
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
                    (
                        active_partner_idx,
                        active_partner_name,
                        active_partner_policy,
                        active_partner_score,
                    ) = _select_partner(all_args, envs, current_policy, prior_policies, rng)

                # SP: self-play (ego가 양쪽 seat) — SP critic
                current_policy.set_sp()
                sp_rollout = collect_rollout(
                    all_args, envs, current_policy, mode="sp", rng=rng
                )
                weighted_rollouts = [("sp", sp_rollout.buffer, 1.0, "sp", None)]
                # transplant 표준 이름 규약: average_episode_rewards(per-episode shaped), ep_sparse_r.
                logs = {
                    f"{convention_name}/sp_average_episode_rewards": sp_rollout.mean_reward,
                    f"{convention_name}/sp_ep_sparse_r": sp_rollout.mean_sparse,
                }

                if active_partner_policy is not None:
                    # XP-seat0: ego가 seat0, 파트너 π*가 seat1 — best_i 전용 XP critic 0
                    current_policy.set_xp(0, active_partner_idx)
                    xp0_rollout = collect_rollout(
                        all_args,
                        envs,
                        current_policy,
                        mode="xp0",
                        partner_policy=active_partner_policy,
                        partner_name=active_partner_name,
                        rng=rng,
                    )
                    # XP-seat1: ego가 seat1 — best_i 전용 XP critic 1
                    current_policy.set_xp(1, active_partner_idx)
                    xp1_rollout = collect_rollout(
                        all_args,
                        envs,
                        current_policy,
                        mode="xp1",
                        partner_policy=active_partner_policy,
                        partner_name=active_partner_name,
                        rng=rng,
                    )
                    # MP: mixed-play — MP critic. 전용 env(episode_length-1 슬롯)로 굴려 switch
                    # 시점을 전 구간 커버(원본 envs_mp와 동일). 첫 사용 시 지연 생성.
                    if mp_envs is None:
                        mp_envs = make_envs(all_args, run_dir, n_threads=mp_threads)
                    current_policy.set_mp()
                    mp_rollout = collect_rollout(
                        all_args,
                        mp_envs,
                        current_policy,
                        mode="mp",
                        partner_policy=active_partner_policy,
                        partner_name=active_partner_name,
                        rng=rng,
                    )
                    # 논문 α(=comedi_alpha, cross-play 총 가중치)를 두 seat 방향에 절반씩 나눠 적용.
                    # (원본: seat0/seat1 각각 -xp_weight, 두 방향 합이 논문 Table 6의 α)
                    xp_weight_per_seat = -all_args.comedi_alpha / 2.0
                    weighted_rollouts.append(
                        ("xp0", xp0_rollout.buffer, xp_weight_per_seat, "xp0", active_partner_idx)
                    )
                    weighted_rollouts.append(
                        ("xp1", xp1_rollout.buffer, xp_weight_per_seat, "xp1", active_partner_idx)
                    )
                    weighted_rollouts.append(
                        ("mp", mp_rollout.buffer, all_args.comedi_beta, "mp", None)
                    )
                    logs.update(
                        {
                            f"{convention_name}/partner": active_partner_name,
                            f"{convention_name}/partner_selection_average_episode_rewards": active_partner_score,
                            f"{convention_name}/xp0_ep_sparse_r": xp0_rollout.mean_sparse,
                            f"{convention_name}/xp1_ep_sparse_r": xp1_rollout.mean_sparse,
                            f"{convention_name}/mp_ep_sparse_r": mp_rollout.mean_sparse,
                            f"{convention_name}/xp0_average_episode_rewards": xp0_rollout.mean_reward,
                            f"{convention_name}/xp1_average_episode_rewards": xp1_rollout.mean_reward,
                            f"{convention_name}/mp_average_episode_rewards": mp_rollout.mean_reward,
                        }
                    )
                    # alpha/beta(상수)는 wandb 그래프 대신 CSV로 기록 → 아래 convention 종료 시점.

                train_info = trainer.train(weighted_rollouts)
                global_step += all_args.episode_length * all_args.n_rollout_threads
                # weight(sp/xp0/xp1/mp_weight)는 상수 → wandb 그래프에서 빼고 CSV로 기록(요청2).
                logs.update(
                    {
                        f"{convention_name}/train/{k}": v
                        for k, v in train_info.items()
                        if not k.endswith("_weight")
                    }
                )
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
            # 요청2: 상수(alpha/beta/weight)는 CSV로. 요청1: SP 재생 GIF를 convention 섹션에 로깅.
            _write_constants_csv(save_root, convention_name, all_args)
            _render_sp_gifs(all_args, run_dir, convention_name, current_policy, run, global_step)
            frozen_policy = make_policy(all_args, envs, device, actor_path, critic_path)
            frozen_policy.to(device)
            prior_policies.append((convention_name, frozen_policy))

        write_population_yamls(all_args.layout_name, int(all_args.comedi_population_size))
    finally:
        envs.close()
        if mp_envs is not None:
            mp_envs.close()
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
