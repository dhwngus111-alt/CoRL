"""CoMeDi PPO trainer and rollout collection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from comedi_transplant.bootstrap import ensure_paths


ensure_paths()

from hsp.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy  # noqa: E402
from hsp.algorithms.utils.util import check  # noqa: E402
from hsp.utils.util import get_gard_norm, huber_loss, mse_loss  # noqa: E402
from hsp.utils.shared_buffer import SharedReplayBuffer  # noqa: E402
from hsp.utils.valuenorm import ValueNorm  # noqa: E402


def _t2n(x):
    return x.detach().cpu().numpy()


@dataclass
class RolloutResult:
    label: str
    buffer: SharedReplayBuffer
    mean_reward: float
    mean_sparse: float
    partner_name: str = ""


class CoMeDiPPOTrainer:
    """PPO trainer with signed actor objective for SP/XP/MP buffers."""

    def __init__(self, args, policy: R_MAPPOPolicy, device=torch.device("cpu")):
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

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_policy_vhead = getattr(args, "use_policy_vhead", False)

        if self._use_popart and self._use_valuenorm:
            raise ValueError("use_popart and use_valuenorm cannot both be true")
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def prep_rollout(self) -> None:
        self.policy.prep_rollout()

    def prep_training(self) -> None:
        self.policy.actor.train()
        self.policy.critic.train()

    def to(self, device) -> None:
        self.policy.to(device)

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

    def _advantages(self, buffer: SharedReplayBuffer) -> np.ndarray:
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(
                buffer.value_preds[:-1]
            )
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        finite_advantages = advantages_copy[np.isfinite(advantages_copy)]
        if finite_advantages.size == 0:
            mean_advantages = 0.0
            std_advantages = 1.0
        else:
            mean_advantages = float(np.mean(finite_advantages))
            std_advantages = float(np.std(finite_advantages))
        if std_advantages < 1e-8:
            std_advantages = 1.0
        return (advantages - mean_advantages) / (std_advantages + 1e-5)

    def _generator(self, buffer: SharedReplayBuffer, advantages: np.ndarray):
        if self._use_recurrent_policy:
            return buffer.recurrent_generator(
                advantages, self.num_mini_batch, self.data_chunk_length
            )
        if self._use_naive_recurrent:
            return buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
        return buffer.feed_forward_generator(advantages, self.num_mini_batch)

    @torch.no_grad()
    def compute_returns(self, buffer: SharedReplayBuffer) -> None:
        self.prep_rollout()
        next_values = self.policy.get_values(
            np.concatenate(buffer.share_obs[-1]),
            np.concatenate(buffer.rnn_states_critic[-1]),
            np.concatenate(buffer.masks[-1]),
        )
        next_values = np.array(np.split(_t2n(next_values), buffer.n_rollout_threads))
        buffer.compute_returns(next_values, self.value_normalizer)

    def _loss_from_sample(self, sample, adv_weight: float):
        (
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            *_,
        ) = sample

        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        if active_masks_batch.sum().item() <= 0:
            zero = torch.zeros((), **self.tpdv)
            return zero, zero, zero, zero, zero

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv) * float(adv_weight)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        values, action_log_probs, dist_entropy, _, _ = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            denom = active_masks_batch.sum()
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / denom
        else:
            policy_action_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        actor_loss = policy_action_loss - dist_entropy * self.entropy_coef
        value_loss = self._cal_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch
        )
        return actor_loss, value_loss, policy_action_loss, dist_entropy, ratio.mean()

    def train(self, weighted_rollouts: Iterable[tuple[str, SharedReplayBuffer, float]]) -> dict:
        rollouts = list(weighted_rollouts)
        for _, buffer, _ in rollouts:
            self.compute_returns(buffer)

        advantages = {label: self._advantages(buffer) for label, buffer, _ in rollouts}
        train_info = {
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "dist_entropy": 0.0,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
            "ratio": 0.0,
        }
        for label, _, _ in rollouts:
            train_info[f"{label}_weight"] = 0.0
            train_info[f"{label}_policy_loss"] = 0.0
            train_info[f"{label}_value_loss"] = 0.0

        update_count = 0
        self.prep_training()
        for _ in range(self.ppo_epoch):
            self.policy.actor_optimizer.zero_grad()
            self.policy.critic_optimizer.zero_grad()
            total_loss = None
            samples_this_epoch = 0

            for label, buffer, weight in rollouts:
                for sample in self._generator(buffer, advantages[label]):
                    actor_loss, value_loss, policy_loss, dist_entropy, ratio = self._loss_from_sample(
                        sample, weight
                    )
                    loss = actor_loss + value_loss * self.value_loss_coef
                    total_loss = loss if total_loss is None else total_loss + loss
                    samples_this_epoch += 1
                    update_count += 1
                    train_info["value_loss"] += float(value_loss.detach().cpu())
                    train_info["policy_loss"] += float(policy_loss.detach().cpu())
                    train_info["dist_entropy"] += float(dist_entropy.detach().cpu())
                    train_info["ratio"] += float(ratio.detach().cpu())
                    train_info[f"{label}_weight"] += float(weight)
                    train_info[f"{label}_policy_loss"] += float(policy_loss.detach().cpu())
                    train_info[f"{label}_value_loss"] += float(value_loss.detach().cpu())

            if total_loss is None:
                continue
            (total_loss / max(samples_this_epoch, 1)).backward()

            if self._use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(), self.max_grad_norm
                )
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.critic.parameters(), self.max_grad_norm
                )
            else:
                actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
                critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

            self.policy.actor_optimizer.step()
            self.policy.critic_optimizer.step()
            train_info["actor_grad_norm"] += float(actor_grad_norm)
            train_info["critic_grad_norm"] += float(critic_grad_norm)

        normalizer = max(update_count, 1)
        for key in list(train_info):
            if key in {"actor_grad_norm", "critic_grad_norm"}:
                train_info[key] /= max(self.ppo_epoch, 1)
            else:
                train_info[key] /= normalizer
        return train_info


def make_buffer(args, envs) -> SharedReplayBuffer:
    return SharedReplayBuffer(
        args,
        args.num_agents,
        envs.observation_space[0],
        envs.share_observation_space[0],
        envs.action_space[0],
        n_rollout_threads=args.n_rollout_threads,
    )


def _empty_rnn(args, n_envs: int) -> np.ndarray:
    return np.zeros(
        (n_envs, args.num_agents, args.recurrent_N, args.hidden_size),
        dtype=np.float32,
    )


def _policy_outputs(policy, share_obs, obs, rnn_states, masks):
    n_envs, n_agents = obs.shape[:2]
    values, actions, action_log_probs, next_rnn, next_rnn_critic = policy.get_actions(
        np.concatenate(share_obs),
        np.concatenate(obs),
        np.concatenate(rnn_states),
        np.concatenate(rnn_states.copy()),
        np.concatenate(masks),
    )
    return (
        np.array(np.split(_t2n(values), n_envs)),
        np.array(np.split(_t2n(actions), n_envs)),
        np.array(np.split(_t2n(action_log_probs), n_envs)),
        np.array(np.split(_t2n(next_rnn), n_envs)),
        np.array(np.split(_t2n(next_rnn_critic), n_envs)),
    )


def _partner_actions(policy, obs, rnn_states, masks):
    n_envs = obs.shape[0]
    actions, next_rnn = policy.act(
        np.concatenate(obs),
        np.concatenate(rnn_states),
        np.concatenate(masks),
        deterministic=False,
    )
    return np.array(np.split(_t2n(actions), n_envs)), np.array(np.split(_t2n(next_rnn), n_envs))


def collect_rollout(
    args,
    envs,
    current_policy: R_MAPPOPolicy,
    mode: str,
    partner_policy: R_MAPPOPolicy | None = None,
    partner_name: str = "",
    rng: np.random.Generator | None = None,
) -> RolloutResult:
    """Collect one rollout for SP, XP, or MP."""

    if mode not in {"sp", "xp", "mp"}:
        raise ValueError(f"unknown rollout mode: {mode}")
    if mode in {"xp", "mp"} and partner_policy is None:
        raise ValueError(f"{mode} rollout requires a partner policy")

    rng = rng or np.random.default_rng()
    obs, share_obs, _ = envs.reset()
    obs = np.stack(obs)
    if not args.use_centralized_V:
        share_obs = obs

    buffer = make_buffer(args, envs)
    buffer.obs[0] = obs.copy()
    buffer.share_obs[0] = share_obs.copy()

    n_envs = args.n_rollout_threads
    current_rnn = _empty_rnn(args, n_envs)
    current_masks = np.ones((n_envs, args.num_agents, 1), dtype=np.float32)
    partner_rnn = _empty_rnn(args, n_envs)
    partner_masks = np.ones_like(current_masks)
    episode_rewards = []
    sparse_scores = []
    switch_steps = rng.integers(1, args.episode_length + 1, size=n_envs)

    for step in range(args.episode_length):
        with torch.no_grad():
            current_policy.prep_rollout()
            values, current_actions, current_log_probs, current_next_rnn, current_next_rnn_critic = (
                _policy_outputs(current_policy, share_obs, obs, current_rnn, current_masks)
            )
            if partner_policy is not None:
                partner_policy.prep_rollout()
                partner_actions, partner_next_rnn = _partner_actions(
                    partner_policy, obs, partner_rnn, partner_masks
                )
            else:
                partner_actions = current_actions
                partner_next_rnn = partner_rnn

        current_controls = np.ones((n_envs, args.num_agents, 1), dtype=bool)
        if mode == "xp":
            current_controls[:, :, :] = False
            current_controls[::2, 0, :] = True
            current_controls[1::2, 1, :] = True
        elif mode == "mp":
            in_phase1 = step < switch_steps
            random_controls = rng.integers(0, 2, size=(n_envs, args.num_agents, 1)).astype(bool)
            current_controls = np.where(in_phase1[:, None, None], random_controls, True)

        actions = np.where(current_controls, current_actions, partner_actions)
        action_log_probs = np.where(current_controls, current_log_probs, 0.0)
        values = np.where(current_controls, values, 0.0)

        next_obs, next_share_obs, rewards, dones, infos, _ = envs.step(actions)
        next_obs = np.stack(next_obs)
        if not args.use_centralized_V:
            next_share_obs = next_obs

        dones = np.asarray(dones)
        masks = np.ones((n_envs, args.num_agents, 1), dtype=np.float32)
        masks[dones == True] = 0.0
        current_next_rnn[dones == True] = 0.0
        current_next_rnn_critic[dones == True] = 0.0
        partner_next_rnn[dones == True] = 0.0

        active_masks = current_controls.astype(np.float32)
        if mode == "mp":
            phase2 = (step >= switch_steps)[:, None, None]
            active_masks = np.where(phase2, 1.0, 0.0).astype(np.float32)
        rewards_to_store = np.asarray(rewards, dtype=np.float32) * active_masks

        buffer.insert(
            next_share_obs,
            next_obs,
            current_next_rnn,
            current_next_rnn_critic,
            actions,
            action_log_probs,
            values,
            rewards_to_store,
            masks,
            active_masks=active_masks,
        )
        buffer.active_masks[step] = active_masks.copy()

        episode_rewards.append(float(np.mean(rewards)))
        if infos:
            sparse_scores = [
                float(info.get("episode", {}).get("ep_sparse_r", 0.0))
                for info in infos
                if "episode" in info
            ]

        obs = next_obs
        share_obs = next_share_obs
        current_rnn = current_next_rnn
        current_masks = masks
        partner_rnn = partner_next_rnn
        partner_masks = masks

    mean_sparse = float(np.mean(sparse_scores)) if sparse_scores else 0.0
    return RolloutResult(
        label=mode,
        buffer=buffer,
        mean_reward=float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        mean_sparse=mean_sparse,
        partner_name=partner_name,
    )
