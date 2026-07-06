#!/usr/bin/env python
"""Train a CoMeDi convention-aware agent (paper Eq. 2) on Risky Overcooked.

논문 "Diverse Conventions for Human-AI Collaboration"의 convention-aware agent는
best-response(보상 기반)가 아니라 **self-play PPO + pool convention들에 대한 behavior
cloning(BC)** 으로 학습된다 (Eq. 2):

    L(π̂, D) = -J(π̂, π̂) - (λ/|D|) Σ_{π∈D} E_{(o,a)∼π}[log π̂(a|o)]

원본 ``MultiConvention/xd.py`` 의 실제 구현을 그대로 따른다:
  - SP 항: π̂ self-play rollout에 대한 clipped PPO (critic도 여기서 갱신), weight 1.
  - BC 항: 각 convention을 self-play로 굴려 그 (obs, action)을 π̂이 흉내 —
           neglogp = -mean(log π̂(a|o)) + (-entropy_coef·entropy). 보상/advantage 미사용.
  - actor loss = SP + Σ BC (각 convention 가중치 1로 합산; Eq.2의 1/|D|는 원본에서
    gradient가 아니라 로깅에만 반영). ``--comedi_bc_weight`` 로 조절 가능(기본 1).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from comedi_transplant.bootstrap import POLICY_POOL_ROOT, ensure_paths
from comedi_transplant.comedi_trainer import collect_rollout
from comedi_transplant.policy_configs import (
    DEFAULT_CNN_LAYERS,
    build_policy_configs,
    policy_entry,
    write_yaml,
)
from comedi_transplant.train_comedi_population import make_envs, make_policy


ensure_paths()

from hsp.algorithms.utils.util import check  # noqa: E402
from hsp.config import get_config  # noqa: E402
from hsp.utils.util import get_gard_norm, huber_loss, mse_loss  # noqa: E402
from hsp.utils.valuenorm import ValueNorm  # noqa: E402
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


def _t2n(x):
    return x.detach().cpu().numpy()


class CoMeDiCBRTrainer:
    """convention-aware agent 학습기: SP clipped-PPO + 각 convention에 대한 BC."""

    def __init__(self, args, policy, device=torch.device("cpu")):
        self.args = args
        self.policy = policy
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.bc_weight = float(getattr(args, "comedi_bc_weight", 1.0))

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        if self._use_popart and self._use_valuenorm:
            raise ValueError("use_popart and use_valuenorm cannot both be true")
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    # ------------------------------------------------------------------
    # value / advantage helpers (SP 항 전용)
    # ------------------------------------------------------------------
    def _cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            denom = active_masks_batch.sum()
            if denom.item() <= 0:
                return value_loss.sum() * 0.0
            value_loss = (value_loss * active_masks_batch).sum() / denom
        else:
            value_loss = value_loss.mean()
        return value_loss

    def _advantages(self, buffer) -> np.ndarray:
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(
                buffer.value_preds[:-1]
            )
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        finite = advantages_copy[np.isfinite(advantages_copy)]
        mean_adv = float(np.mean(finite)) if finite.size else 0.0
        std_adv = float(np.std(finite)) if finite.size else 1.0
        if std_adv < 1e-8:
            std_adv = 1.0
        return (advantages - mean_adv) / (std_adv + 1e-5)

    def _generator(self, buffer, advantages):
        if self._use_recurrent_policy:
            return buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
        if self._use_naive_recurrent:
            return buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
        return buffer.feed_forward_generator(advantages, self.num_mini_batch)

    @torch.no_grad()
    def compute_returns(self, buffer) -> None:
        self.policy.actor.eval()
        self.policy.critic.eval()
        next_values = self.policy.get_values(
            np.concatenate(buffer.share_obs[-1]),
            np.concatenate(buffer.rnn_states_critic[-1]),
            np.concatenate(buffer.masks[-1]),
        )
        next_values = np.array(np.split(_t2n(next_values), buffer.n_rollout_threads))
        buffer.compute_returns(next_values, self.value_normalizer)

    # ------------------------------------------------------------------
    # SP (clipped PPO) 및 BC loss
    # ------------------------------------------------------------------
    def _ppo_loss(self, sample):
        (
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch,
            actions_batch, value_preds_batch, return_batch, masks_batch,
            active_masks_batch, old_action_log_probs_batch, adv_targ,
            available_actions_batch, *_,
        ) = sample

        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        if active_masks_batch.sum().item() <= 0:
            zero = torch.zeros((), **self.tpdv)
            return zero, zero, zero, zero
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        values, action_log_probs, dist_entropy, _, _ = self.policy.evaluate_actions(
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch,
            actions_batch, masks_batch, available_actions_batch, active_masks_batch,
        )

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        if self._use_policy_active_masks:
            denom = active_masks_batch.sum()
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / denom
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        actor_loss = policy_action_loss - dist_entropy * self.entropy_coef
        value_loss = self._cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        return actor_loss, value_loss, policy_action_loss, dist_entropy

    def _bc_loss(self, sample):
        (
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch,
            actions_batch, value_preds_batch, return_batch, masks_batch,
            active_masks_batch, old_action_log_probs_batch, adv_targ,
            available_actions_batch, *_,
        ) = sample
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        _, action_log_probs, dist_entropy, _, _ = self.policy.evaluate_actions(
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch,
            actions_batch, masks_batch, available_actions_batch, active_masks_batch,
        )
        # 원본 bc_update: neglogp = -mean(log π̂(a|o)), ent_loss = -entropy_coef·entropy
        log_prob = action_log_probs.mean()
        entropy = dist_entropy.mean()
        neglogp = -log_prob
        ent_loss = -self.entropy_coef * entropy
        bc_loss = neglogp + ent_loss
        return self.bc_weight * bc_loss, neglogp, torch.exp(log_prob)

    def train(self, sp_buffer, conv_buffers) -> dict:
        """SP buffer(π̂ self-play) + convention self-play buffer들로 Eq.2 업데이트."""
        self.compute_returns(sp_buffer)
        sp_adv = self._advantages(sp_buffer)
        zero_advs = [np.zeros_like(buf.returns[:-1]) for buf in conv_buffers]

        info = {
            "sp_policy_loss": 0.0, "sp_value_loss": 0.0, "sp_entropy": 0.0,
            "bc_neglogp": 0.0, "bc_prob_true_act": 0.0,
            "actor_grad_norm": 0.0, "critic_grad_norm": 0.0,
        }
        n_bc = max(len(conv_buffers), 1)
        self.policy.actor.train()
        self.policy.critic.train()

        for _ in range(self.ppo_epoch):
            self.policy.actor_optimizer.zero_grad()
            self.policy.critic_optimizer.zero_grad()
            actor_total = None

            # --- SP 항 (PPO): actor 누적, critic 즉시 갱신 ---
            critic_loss_total = None
            for sample in self._generator(sp_buffer, sp_adv):
                actor_loss, value_loss, policy_loss, dist_entropy = self._ppo_loss(sample)
                if actor_loss.requires_grad:
                    actor_total = actor_loss if actor_total is None else actor_total + actor_loss
                if value_loss.requires_grad:
                    v = value_loss * self.value_loss_coef
                    critic_loss_total = v if critic_loss_total is None else critic_loss_total + v
                info["sp_policy_loss"] += float(policy_loss.detach().cpu())
                info["sp_value_loss"] += float(value_loss.detach().cpu())
                info["sp_entropy"] += float(dist_entropy.detach().cpu())
            if critic_loss_total is not None:
                critic_loss_total.backward()
                if self._use_max_grad_norm:
                    cgn = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
                else:
                    cgn = get_gard_norm(self.policy.critic.parameters())
                self.policy.critic_optimizer.step()
                info["critic_grad_norm"] += float(cgn)

            # --- BC 항: 각 convention self-play (obs, action) 흉내 ---
            for buf, zadv in zip(conv_buffers, zero_advs):
                for sample in self._generator(buf, zadv):
                    bc_loss, neglogp, prob_true = self._bc_loss(sample)
                    if bc_loss.requires_grad:
                        actor_total = bc_loss if actor_total is None else actor_total + bc_loss
                    info["bc_neglogp"] += float(neglogp.detach().cpu()) / n_bc
                    info["bc_prob_true_act"] += float(prob_true.detach().cpu()) / n_bc

            # --- actor: SP + Σ BC 합쳐 한 번 갱신 (원본 xd.py:437) ---
            if actor_total is not None:
                actor_total.backward()
                if self._use_max_grad_norm:
                    agn = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
                else:
                    agn = get_gard_norm(self.policy.actor.parameters())
                self.policy.actor_optimizer.step()
                info["actor_grad_norm"] += float(agn)

        denom = max(self.ppo_epoch, 1)
        for key in info:
            info[key] /= denom
        return info


# ----------------------------------------------------------------------
# CLI / orchestration
# ----------------------------------------------------------------------
def _add_cbr_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--comedi_population_size", type=int, default=8)
    parser.add_argument("--comedi_adaptive_agent_name", type=str, default="comedi_adaptive")
    parser.add_argument("--comedi_bc_weight", type=float, default=1.0)
    parser.add_argument("--comedi_skip_policy_config", action="store_true")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = get_config()
    add_risky_overcooked_args(parser, mode="hsp")
    _add_cbr_args(parser)
    # Table 7 / adap_cbr.sh: 200k steps, ppo 100, lr 1e-2, entropy 1e-3, MLP.
    parser.set_defaults(
        algorithm_name="mappo",
        experiment_name="comedi-cbr",
        n_rollout_threads=50,
        episode_length=200,
        num_env_steps=200_000,
        ppo_epoch=100,
        num_mini_batch=1,
        lr=1e-2,
        critic_lr=1e-2,
        use_linear_lr_decay=True,
        entropy_coef=1e-3,
        hidden_size=64,
        layer_N=2,
        activation_id=1,
        cnn_layers_params=DEFAULT_CNN_LAYERS,
        use_recurrent_policy=False,
        use_naive_recurrent_policy=False,
    )
    all_args = normalize_risky_args(parser.parse_known_args(argv)[0])
    if all_args.algorithm_name != "mappo":
        raise ValueError("CoMeDi convention-aware agent uses mappo/MLP policies")
    return all_args


def _load_conventions(all_args, envs, device):
    s1_root = POLICY_POOL_ROOT / all_args.layout_name / "comedi" / "s1"
    conventions = []
    for idx in range(1, int(all_args.comedi_population_size) + 1):
        actor_path = s1_root / f"comedi{idx}_actor.pt"
        critic_path = s1_root / f"comedi{idx}_critic.pt"
        if not actor_path.exists():
            raise FileNotFoundError(f"convention actor not found: {actor_path}")
        policy = make_policy(
            all_args, envs, device, actor_path,
            critic_path if critic_path.exists() else None,
        )
        policy.to(device)
        conventions.append((f"comedi{idx}", policy))
    return conventions


def _register_final(all_args, policy) -> Path:
    s2_root = POLICY_POOL_ROOT / all_args.layout_name / "comedi" / "s2"
    s2_root.mkdir(parents=True, exist_ok=True)
    name = all_args.comedi_adaptive_agent_name
    actor_dst = s2_root / f"{name}_actor.pt"
    critic_dst = s2_root / f"{name}_critic.pt"
    torch.save(policy.actor.state_dict(), actor_dst)
    torch.save(policy.critic.state_dict(), critic_dst)
    actor_rel = f"{all_args.layout_name}/comedi/s2/{actor_dst.name}"
    # 논문 convention-aware agent은 MLP이므로 mlp 정책 설정으로 eval.yml에 등록
    # (기존 update_adaptive_eval_yaml은 rnn으로 등록해 MLP actor 로드와 불일치)
    eval_path = s2_root / "eval.yml"
    data = yaml.safe_load(eval_path.open()) if eval_path.exists() else {}
    data[name] = policy_entry(all_args.layout_name, "mlp", train=False, actor_path=actor_rel)
    write_yaml(eval_path, data)
    return actor_dst


def train_cbr(all_args) -> None:
    if not all_args.comedi_skip_policy_config:
        build_policy_configs(all_args.layout_name, all_args.episode_length, POLICY_POOL_ROOT)

    device = setup_device(all_args)
    run_dir = make_run_dir(all_args)
    set_process_title(all_args)
    set_seeds(all_args.seed)
    run = init_wandb(all_args, run_dir)
    envs = make_envs(all_args, run_dir)
    rng = np.random.default_rng(all_args.seed)

    try:
        conventions = _load_conventions(all_args, envs, device)
        agent = make_policy(all_args, envs, device)  # π̂: 처음부터 학습
        trainer = CoMeDiCBRTrainer(all_args, agent, device=device)

        episodes = int(all_args.num_env_steps) // all_args.episode_length // all_args.n_rollout_threads
        episodes = max(1, episodes)
        global_step = 0
        start = time.time()
        for episode in range(episodes):
            if all_args.use_linear_lr_decay:
                agent.lr_decay(episode, episodes)

            # π̂ self-play rollout (SP 항)
            sp_rollout = collect_rollout(all_args, envs, agent, mode="sp", rng=rng)
            # 각 convention self-play rollout (BC 대상)
            conv_buffers = []
            for _, conv_policy in conventions:
                cr = collect_rollout(all_args, envs, conv_policy, mode="sp", rng=rng)
                conv_buffers.append(cr.buffer)

            info = trainer.train(sp_rollout.buffer, conv_buffers)
            global_step += all_args.episode_length * all_args.n_rollout_threads

            if run is not None:
                try:
                    import wandb

                    wandb.log(
                        {
                            "cbr/sp_sparse": sp_rollout.mean_sparse,
                            **{f"cbr/{k}": v for k, v in info.items()},
                        },
                        step=global_step,
                    )
                except Exception:
                    pass

            if episode % all_args.log_interval == 0:
                fps = int(global_step / max(time.time() - start, 1e-6))
                print(
                    f"{all_args.layout_name} comedi-cbr episode {episode + 1}/{episodes} "
                    f"step {global_step} FPS {fps} sp_sparse={sp_rollout.mean_sparse:.3f} "
                    f"bc_prob={info['bc_prob_true_act']:.3f}"
                )

        actor_dst = _register_final(all_args, agent)
        print(f"saved convention-aware agent: {actor_dst}")
    finally:
        envs.close()

        class _RunnerShim:
            pass

        shim = _RunnerShim()
        shim.save_dir = str(POLICY_POOL_ROOT / all_args.layout_name / "comedi" / "s2")
        shim.run_dir = str(run_dir)
        finish_logging(all_args, run, shim)


def main(argv: list[str] | None = None) -> None:
    all_args = parse_args(sys.argv[1:] if argv is None else argv)
    train_cbr(all_args)


if __name__ == "__main__":
    main()
