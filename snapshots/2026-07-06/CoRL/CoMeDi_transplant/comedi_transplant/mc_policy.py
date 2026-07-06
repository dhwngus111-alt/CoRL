"""Multi-critic policy for CoMeDi transplant.

원본 CoMeDi ``train/XD/MCPolicy.py``를 그대로 본떠, 하나의 actor에 목적별 critic을 붙인다:
  - ``sp_critic``  : self-play (base critic 재사용)
  - ``mp_critic``  : mixed-play
  - ``xp_critic0[i]`` / ``xp_critic1[i]`` : prior convention i별 × ego-seat별 cross-play critic

원본과 동일하게 XP critic은 **prior convention마다** 둔다. best_i(π*)가 학습 도중 바뀌어도
각 convention이 전용 value baseline을 유지하도록 하기 위함. 기본 모드(use_average=False)에서는
매 step best_i의 critic만 갱신된다.
"""

from __future__ import annotations

import torch

from comedi_transplant.bootstrap import ensure_paths


ensure_paths()

from hsp.algorithms.r_mappo.algorithm.r_actor_critic import R_Critic  # noqa: E402
from hsp.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy  # noqa: E402
from hsp.utils.util import update_linear_schedule  # noqa: E402


class CoMeDiMCPolicy(R_MAPPOPolicy):
    """SP / MP / (convention·seat별) XP 전용 critic을 갖는 policy."""

    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu"), num_priors: int = 0):
        super().__init__(args, obs_space, share_obs_space, act_space, device)

        self.num_priors = int(num_priors)

        # base critic을 SP 전용으로 재사용 (원본 MCPolicy와 동일)
        self.sp_critic = self.critic
        self.sp_critic_optimizer = self.critic_optimizer

        self.mp_critic = R_Critic(args, self.share_obs_space, self.device)
        self.mp_critic_optimizer = self._make_opt(self.mp_critic)

        # prior convention마다 seat0/seat1 XP critic (원본 xp_critic0[i]/xp_critic1[i])
        self.xp_critic0 = [R_Critic(args, self.share_obs_space, self.device) for _ in range(self.num_priors)]
        self.xp_critic1 = [R_Critic(args, self.share_obs_space, self.device) for _ in range(self.num_priors)]
        self.xp_critic0_optimizer = [self._make_opt(c) for c in self.xp_critic0]
        self.xp_critic1_optimizer = [self._make_opt(c) for c in self.xp_critic1]

        # 시작 시 SP critic을 활성으로
        self.set_sp()

    def _make_opt(self, critic):
        return torch.optim.Adam(
            critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    # ------------------------------------------------------------------
    # critic 스왑 (get_actions/get_values/evaluate_actions가 self.critic 참조)
    # ------------------------------------------------------------------
    def set_sp(self) -> None:
        self.critic = self.sp_critic
        self.critic_optimizer = self.sp_critic_optimizer

    def set_mp(self) -> None:
        self.critic = self.mp_critic
        self.critic_optimizer = self.mp_critic_optimizer

    def set_xp(self, seat: int, idx: int) -> None:
        if seat == 0:
            self.critic = self.xp_critic0[idx]
            self.critic_optimizer = self.xp_critic0_optimizer[idx]
        else:
            self.critic = self.xp_critic1[idx]
            self.critic_optimizer = self.xp_critic1_optimizer[idx]

    def set_critic(self, kind: str, idx: int | None = None) -> None:
        if kind == "sp":
            self.set_sp()
        elif kind == "mp":
            self.set_mp()
        elif kind == "xp0":
            self.set_xp(0, idx)
        elif kind == "xp1":
            self.set_xp(1, idx)
        else:
            raise ValueError(f"unknown critic kind: {kind}")

    def all_critics(self):
        return [self.sp_critic, self.mp_critic] + self.xp_critic0 + self.xp_critic1

    def all_critic_optimizers(self):
        return (
            [self.sp_critic_optimizer, self.mp_critic_optimizer]
            + self.xp_critic0_optimizer
            + self.xp_critic1_optimizer
        )

    # ------------------------------------------------------------------
    # 오버라이드: 모든 critic을 함께 다뤄야 함
    # ------------------------------------------------------------------
    def lr_decay(self, episode, episodes) -> None:
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        for opt in self.all_critic_optimizers():
            update_linear_schedule(opt, episode, episodes, self.critic_lr)

    def to(self, device) -> None:
        self.actor.to(device)
        for crit in self.all_critics():
            crit.to(device)

    def prep_training(self) -> None:
        self.actor.train()
        for crit in self.all_critics():
            crit.train()

    def prep_rollout(self) -> None:
        self.actor.eval()
        for crit in self.all_critics():
            crit.eval()
