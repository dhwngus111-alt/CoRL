"""CoMeDi PPO trainer and rollout collection utilities.

원본 CoMeDi ``train/XD/xd.py``를 본떠, 하나의 actor를 SP/XP/MP objective 조합으로
업데이트하되 **목적별 전용 critic**(SP / MP / prior convention·seat별 XP)으로 value baseline을
분리한다. actor 목적함수는 논문 Eq.(8) 부호(SP +1, XP -alpha, MP +beta)를 그대로 따르며,
XP 총 가중치 alpha는 seat0/seat1 두 방향에 절반씩 나눠 적용한다.
"""

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

from comedi_transplant.mc_policy import CoMeDiMCPolicy  # noqa: E402


# critic 종류 (idx는 xp0/xp1일 때만 prior convention 인덱스)
CRITIC_KINDS = ("sp", "xp0", "xp1", "mp")


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
    """SP/XP/MP 버퍼에 대해 signed actor objective + 목적별 critic으로 PPO 업데이트."""

    def __init__(self, args, policy: CoMeDiMCPolicy, device=torch.device("cpu")):
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

        # 원본 xd.py와 동일하게 value normalizer는 단일 공유
        if self._use_popart:
            self.value_normalizer = self.policy.sp_critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def prep_rollout(self) -> None:
        self.policy.prep_rollout()

    def prep_training(self) -> None:
        self.policy.prep_training()

    def to(self, device) -> None:
        self.policy.to(device)

    def _cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        normalizer = self.value_normalizer
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        if self._use_popart or self._use_valuenorm:
            normalizer.update(return_batch)
            error_clipped = normalizer.normalize(return_batch) - value_pred_clipped
            error_original = normalizer.normalize(return_batch) - values
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
        normalizer = self.value_normalizer
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - normalizer.denormalize(
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
    def compute_returns(self, buffer: SharedReplayBuffer, kind: str, idx: int | None = None) -> None:
        """buffer의 return을 (kind, idx) 전용 critic으로 계산."""
        self.policy.set_critic(kind, idx)
        self.prep_rollout()
        next_values = self.policy.get_values(
            np.concatenate(buffer.share_obs[-1]),
            np.concatenate(buffer.rnn_states_critic[-1]),
            np.concatenate(buffer.masks[-1]),
        )
        next_values = np.array(np.split(_t2n(next_values), buffer.n_rollout_threads))
        buffer.compute_returns(next_values, self.value_normalizer)

    def _loss_from_sample(self, sample, adv_weight: float):
        """활성 critic이 이미 세팅돼 있다고 가정. actor/critic loss를 분리 반환."""
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

    def train(self, weighted_rollouts: Iterable[tuple]) -> dict:
        """weighted_rollouts: (label, buffer, weight, kind, idx) 튜플들.

        원본 xd.py.train처럼, 목적별 critic은 각자 즉시 갱신하고 actor는 모든 objective의
        weighted policy loss를 누적해 epoch당 한 번 갱신한다.
        """
        rollouts = [self._normalize_rollout(r) for r in weighted_rollouts]

        for label, buffer, weight, kind, idx in rollouts:
            self.compute_returns(buffer, kind, idx)

        advantages = {label: self._advantages(buffer) for label, buffer, weight, kind, idx in rollouts}

        train_info = {
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "dist_entropy": 0.0,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
            "ratio": 0.0,
        }
        for label, _, _, _, _ in rollouts:
            train_info[f"{label}_weight"] = 0.0
            train_info[f"{label}_policy_loss"] = 0.0
            train_info[f"{label}_value_loss"] = 0.0

        update_count = 0
        critic_update_count = 0
        self.prep_training()
        for _ in range(self.ppo_epoch):
            self.policy.actor_optimizer.zero_grad()
            actor_loss_total = None
            actor_samples = 0

            for label, buffer, weight, kind, idx in rollouts:
                # 이 objective 전용 critic 활성화
                self.policy.set_critic(kind, idx)
                self.policy.critic_optimizer.zero_grad()
                critic_loss_total = None
                critic_batches = 0

                for sample in self._generator(buffer, advantages[label]):
                    actor_loss, value_loss, policy_loss, dist_entropy, ratio = self._loss_from_sample(
                        sample, weight
                    )
                    if actor_loss.requires_grad:
                        actor_loss_total = (
                            actor_loss if actor_loss_total is None else actor_loss_total + actor_loss
                        )
                        actor_samples += 1
                    if value_loss.requires_grad:
                        weighted_vloss = value_loss * self.value_loss_coef
                        critic_loss_total = (
                            weighted_vloss
                            if critic_loss_total is None
                            else critic_loss_total + weighted_vloss
                        )
                        critic_batches += 1

                    update_count += 1
                    train_info["value_loss"] += float(value_loss.detach().cpu())
                    train_info["policy_loss"] += float(policy_loss.detach().cpu())
                    train_info["dist_entropy"] += float(dist_entropy.detach().cpu())
                    train_info["ratio"] += float(ratio.detach().cpu())
                    train_info[f"{label}_weight"] += float(weight)
                    train_info[f"{label}_policy_loss"] += float(policy_loss.detach().cpu())
                    train_info[f"{label}_value_loss"] += float(value_loss.detach().cpu())

                # objective별 critic 즉시 갱신
                if critic_loss_total is not None:
                    (critic_loss_total / max(critic_batches, 1)).backward()
                    if self._use_max_grad_norm:
                        critic_grad_norm = nn.utils.clip_grad_norm_(
                            self.policy.critic.parameters(), self.max_grad_norm
                        )
                    else:
                        critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
                    self.policy.critic_optimizer.step()
                    train_info["critic_grad_norm"] += float(critic_grad_norm)
                    critic_update_count += 1

            # actor는 누적된 weighted loss를 그대로 backward (원본 xd.py:333처럼 합산; 평균 X).
            # 평균을 내면 (1) objective 스케일이 1/N로 줄고 (2) convention마다 N이 달라
            # (첫 convention은 SP뿐 → /1, 이후 → /4) gradient scale이 불일치한다.
            if actor_loss_total is not None:
                actor_loss_total.backward()
                if self._use_max_grad_norm:
                    actor_grad_norm = nn.utils.clip_grad_norm_(
                        self.policy.actor.parameters(), self.max_grad_norm
                    )
                else:
                    actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
                self.policy.actor_optimizer.step()
                train_info["actor_grad_norm"] += float(actor_grad_norm)

        normalizer = max(update_count, 1)
        for key in list(train_info):
            if key == "actor_grad_norm":
                train_info[key] /= max(self.ppo_epoch, 1)
            elif key == "critic_grad_norm":
                train_info[key] /= max(critic_update_count, 1)
            else:
                train_info[key] /= normalizer
        return train_info

    @staticmethod
    def _normalize_rollout(rollout) -> tuple:
        """(label, buffer, weight, kind, idx) 형태로 정규화."""
        if len(rollout) == 5:
            label, buffer, weight, kind, idx = rollout
        elif len(rollout) == 4:
            label, buffer, weight, kind = rollout
            idx = None
        else:
            raise ValueError(f"invalid rollout tuple: {rollout}")
        if kind not in CRITIC_KINDS:
            raise ValueError(f"unknown critic kind '{kind}' for rollout '{label}'")
        if kind in {"xp0", "xp1"} and idx is None:
            raise ValueError(f"critic kind '{kind}' requires a convention idx")
        return label, buffer, weight, kind, idx


def make_buffer(args, envs, n_threads: int | None = None) -> SharedReplayBuffer:
    return SharedReplayBuffer(
        args,
        args.num_agents,
        envs.observation_space[0],
        envs.share_observation_space[0],
        envs.action_space[0],
        n_rollout_threads=args.n_rollout_threads if n_threads is None else n_threads,
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


# 지원하는 rollout 모드: self-play / cross-play(ego seat0) / cross-play(ego seat1) / mixed-play
ROLLOUT_MODES = {"sp", "xp0", "xp1", "mp"}


def collect_rollout(
    args,
    envs,
    current_policy: R_MAPPOPolicy,
    mode: str,
    partner_policy: R_MAPPOPolicy | None = None,
    partner_name: str = "",
    rng: np.random.Generator | None = None,
) -> RolloutResult:
    """Collect one rollout for SP, XP(seat0/seat1), or MP.

    value_preds는 current_policy.get_actions가 참조하는 활성 critic에서 나오므로,
    호출 직전에 오케스트레이터가 목적별 critic을 세팅해야 한다.
    """

    if mode not in ROLLOUT_MODES:
        raise ValueError(f"unknown rollout mode: {mode}")
    if mode in {"xp0", "xp1", "mp"} and partner_policy is None:
        raise ValueError(f"{mode} rollout requires a partner policy")

    rng = rng or np.random.default_rng()
    obs, share_obs, _ = envs.reset()
    obs = np.stack(obs)
    if not args.use_centralized_V:
        share_obs = obs

    # 실제 env 인스턴스의 스레드 수에 맞춘다. MP는 별도 env(=episode_length-1개 슬롯)로 굴려
    # n_rollout_threads보다 클 수 있으므로 args가 아니라 reset 결과에서 n_envs를 유도한다.
    n_envs = obs.shape[0]
    buffer = make_buffer(args, envs, n_threads=n_envs)
    buffer.obs[0] = obs.copy()
    buffer.share_obs[0] = share_obs.copy()
    current_rnn = _empty_rnn(args, n_envs)
    current_masks = np.ones((n_envs, args.num_agents, 1), dtype=np.float32)
    partner_rnn = _empty_rnn(args, n_envs)
    partner_masks = np.ones_like(current_masks)
    episode_rewards = []
    sparse_scores = []
    # switch point(혼합모드→혼자모드 전환 시점)를 env마다 1..episode_length-1 "전 구간"에 균등 배치.
    # 원본 collect_mp_episode는 envs_mp를 episode_length-1개 슬롯으로 따로 굴려 switch 시점
    # 1..episode_length-1을 전부 커버한다(n_rollout_threads와 무관). 우리는 MP를 n_envs에 묶으므로,
    # n_envs가 episode_length보다 작아도 linspace로 초반~후반 switch를 전 구간에 걸쳐 고르게
    # 샘플한다(예: n_envs=50, L=200 → 1,5,9,...,195,199).
    # 상한은 episode_length-1: step 루프가 0..episode_length-1이라 switch=episode_length면
    # phase2(step>=switch)가 한 번도 참이 안 돼 그 env의 self-play 회복 샘플이 0이 되기 때문
    # (원본도 max switch = length-1로 최소 1개 회복 스텝을 보장). arange%L은 1..n_envs만 덮어
    # 후반부 회복 케이스를 놓치므로 사용하지 않는다.
    switch_steps = np.linspace(1, args.episode_length - 1, n_envs).round().astype(np.int64)

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
        if mode == "xp0":
            # ego가 seat0, 파트너 π*가 seat1 (모든 env)
            current_controls[:, :, :] = False
            current_controls[:, 0, :] = True
        elif mode == "xp1":
            # ego가 seat1, 파트너 π*가 seat0 (모든 env)
            current_controls[:, :, :] = False
            current_controls[:, 1, :] = True
        elif mode == "mp":
            # 원본 collect_mp_episode 이식: phase1엔 *양쪽 seat 모두* 매 스텝 mix_prob 확률로
            # ego(current_policy, self-play transition) 또는 best_i partner(cross-play transition)로
            # 독립 혼합되고, phase2(switch 이후)엔 양쪽 seat 모두 ego(self-play recovery).
            # 학습 신호는 phase2만 사용하되 양쪽 seat 모두 씀(아래 active_masks mp 블록에서 처리) —
            # 논문 Algorithm 1(버퍼=phase2 self-play, a1·a2 둘 다) 및 원본(get_gen 전체 버퍼, 양쪽 active)과 일치.
            # phase1(혼합)은 seat 무관 active_mask=0·reward=0으로 제외 → phase2 advantage에 영향 없음.
            mix_prob = float(getattr(args, "comedi_mix_prob", 0.5))
            in_phase1 = step < switch_steps  # (n_envs,)
            current_controls[:, :, :] = False
            seat0_coin = rng.random(n_envs) < mix_prob  # phase1: True→ego, False→best_i
            seat1_coin = rng.random(n_envs) < mix_prob
            current_controls[:, 0, 0] = np.where(in_phase1, seat0_coin, True)  # phase2엔 항상 ego
            current_controls[:, 1, 0] = np.where(in_phase1, seat1_coin, True)

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
            # 논문 Algorithm 1: 버퍼엔 phase2(switch 이후)만 담고, a1·a2를 둘 다 저장한다.
            # phase2엔 양쪽 seat 모두 ego(self-play)이므로 seat0/seat1 둘 다 학습 신호로 쓴다
            # (원본도 MixedAgent가 partner seat active=1로 기록 → 양쪽 active, 전체 buffer generator 사용).
            # 이는 SP 모드(양쪽 seat active)와도 일관 — phase2는 SP와 동일한 self-play이기 때문.
            # phase1(혼합)은 seat 무관 제외.
            phase2 = (step >= switch_steps).astype(np.float32)  # (n_envs,)
            active_masks = np.zeros((n_envs, args.num_agents, 1), dtype=np.float32)
            active_masks[:, 0, 0] = phase2
            active_masks[:, 1, 0] = phase2
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
    # transplant runner 표준과 스케일 일치: average_episode_rewards = mean(step reward) × episode_length.
    # (다른 baseline은 per-episode로 로깅하므로, per-step 평균에 episode_length를 곱해 맞춘다.
    #  partner 선택은 argmax라 스케일 무관 → 영향 없음.)
    mean_reward = (
        float(np.mean(episode_rewards)) * float(args.episode_length) if episode_rewards else 0.0
    )
    return RolloutResult(
        label=mode,
        buffer=buffer,
        mean_reward=mean_reward,
        mean_sparse=mean_sparse,
        partner_name=partner_name,
    )
