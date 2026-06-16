import numpy as np
import time
import math

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from hsp.utils.util import get_gard_norm, huber_loss, mse_loss
from hsp.utils.valuenorm import ValueNorm
from hsp.algorithms.utils.util import check
from .algorithm.rMAPPOPolicy import R_MAPPOPolicy

class R_MAPPO():
    # R_MAPPO는 actor/critic 네트워크 하나를 PPO 방식으로 학습시키는 trainer다.
    # runner가 rollout을 모아 buffer에 넣으면, 이 클래스가 buffer를 읽어서 update를 수행한다.
    def __init__(self,
                 args,
                 policy: R_MAPPOPolicy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        # PPO 학습에 필요한 hyperparameter들을 args에서 꺼내 저장한다.
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.policy_value_loss_coef = args.policy_value_loss_coef
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.shaped_info_coef = getattr(args, "shaped_info_coef", 0.5)
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta
        self.share_policy = args.share_policy

        # 아래 flag들은 어떤 loss/normalization/RNN 옵션을 켤지 정한다.
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_policy_vhead = args.use_policy_vhead
        self._predict_other_shaped_info = (args.env_name == "Overcooked" and getattr(args, "predict_other_shaped_info", False))
        self._policy_group_normalization = (args.env_name == "Overcooked" and getattr(args, "policy_group_normalization", False))
        self._use_task_v_out = getattr(args, "use_task_v_out", False)
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        # critic target(return)의 scale을 안정화하기 위한 value normalizer 설정.
        # PopArt나 ValueNorm 둘 중 하나만 사용할 수 있다.
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
            if self._use_policy_vhead:
                self.policy_value_normalizer = self.policy.actor.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device = self.device)
            if self._use_policy_vhead:
                self.policy_value_normalizer = ValueNorm(1, device = self.device)
        else:
            self.value_normalizer = None
            if self._use_policy_vhead:
                self.policy_value_normalizer = None

    # critic이 예측한 value와 실제 target return 사이의 차이를 loss로 만든다.
    # PPO에서는 value도 너무 급격히 바뀌지 않게 clipped value loss를 쓸 수 있다.
    def cal_value_loss(self, value_normalizer, values, value_preds_batch, return_batch, active_masks_batch):
        # 새 value(values)가 rollout 당시 value(value_preds_batch)에서 너무 멀리 가지 않도록 제한한다.
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        
        # PopArt/ValueNorm을 쓰면 return을 normalized scale로 바꿔서 value loss를 계산한다.
        if self._use_popart or self._use_valuenorm:
            value_normalizer.update(return_batch)
            error_clipped = value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        # error를 MSE 또는 Huber loss로 바꾼다. Huber는 큰 error에 조금 덜 민감하다.
        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        # clipped/original 중 더 큰 loss를 쓰면 critic update가 과하게 튀는 것을 막는 효과가 있다.
        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        # active mask는 episode가 끝난 뒤 padding처럼 들어간 transition을 loss에서 빼기 위한 mask다.
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    # Overcooked 보조 과제: 상대 agent의 shaped-info/event class를 예측하는 loss다.
    # policy 표현이 상대 행동 특징을 더 잘 담도록 돕는 auxiliary loss로 쓰인다.
    def cal_shaped_info_loss(self, shaped_info_batch, pred_shaped_info):
        loss = - (shaped_info_batch * pred_shaped_info.log()).sum(dim=-1)
        return loss.mean()

    # mini-batch 하나에 대해 PPO actor update와 critic update를 한 번 수행한다.
    # train()이 buffer를 여러 mini-batch로 쪼갠 뒤 이 함수를 반복해서 호출한다.
    def ppo_update(self, sample, turn_on=True): # 여기가 PPO 업데이트하는 과정
        # share_policy=True일 때는 Overcooked/HSP용 보조 정보도 sample에 같이 들어온다.
        if self.share_policy:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, shaped_info_batch, other_policy_id_batch = sample
        else:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample
            shaped_info_batch = other_policy_id_batch = None


        # numpy batch를 torch tensor로 바꾸고 device(cpu/cuda)에 올린다.
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        if self._predict_other_shaped_info and shaped_info_batch is not None:
            shaped_info_batch = check(shaped_info_batch).to(**self.tpdv)
        if self._use_task_v_out and other_policy_id_batch is not None:
            other_policy_id_batch = check(other_policy_id_batch).to(**self.tpdv)

        # 현재 policy 기준으로 "buffer에 저장된 action"의 log prob과 value를 다시 계산한다.
        # PPO는 old log prob과 new log prob의 비율을 써서 policy를 업데이트한다.
        values, action_log_probs, dist_entropy, policy_values, pred_shaped_info = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch,
                                                                              task_id=other_policy_id_batch if self._use_task_v_out else None)
        # actor update: ratio = pi_new(a|s) / pi_old(a|s)
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

        # PPO clipped surrogate objective.
        # ratio가 1에서 너무 멀어지면 clamp해서 policy가 한 번에 너무 크게 변하지 못하게 한다.
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        # advantage가 좋은 action은 확률을 올리고, 나쁜 action은 확률을 내리도록 만드는 actor loss.
        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # actor 쪽에도 value head를 붙여 쓰는 옵션이 켜져 있으면 policy loss에 value loss를 더한다.
        if self._use_policy_vhead:
            policy_value_loss = self.cal_value_loss(self.policy_value_normalizer, policy_values, value_preds_batch, return_batch, active_masks_batch)       
            policy_loss = policy_action_loss + policy_value_loss * self.policy_value_loss_coef
        else:
            policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        # turn_on=False이면 critic warmup처럼 actor update를 잠시 막을 수 있다.
        if turn_on:
            # entropy는 exploration을 유지하기 위한 보너스라 loss에서 빼준다.
            loss = policy_loss - dist_entropy * self.entropy_coef
            if self._predict_other_shaped_info:
                shaped_info_loss = self.cal_shaped_info_loss(shaped_info_batch, pred_shaped_info)
                shaped_info_error = abs(shaped_info_batch - pred_shaped_info)
                shaped_info_error = shaped_info_error.mean(dim=list(range(len(shaped_info_error.shape)-1))).cpu().detach()
                avg_pred_shaped_info = pred_shaped_info.mean(dim=list(range(len(pred_shaped_info.shape)-1))).cpu().detach()
                loss += shaped_info_loss * self.shaped_info_coef
            else:
                shaped_info_loss = torch.tensor(0.)
                avg_pred_shaped_info = shaped_info_error = torch.tensor([0.])
            loss.backward()
        else:
            shaped_info_loss = torch.tensor(0.)
            avg_pred_shaped_info = shaped_info_error = torch.tensor([0.])

        # gradient exploding을 막기 위해 actor gradient norm을 clip할 수 있다.
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update: value prediction이 return_batch에 가까워지도록 value loss를 줄인다.
        value_loss = self.cal_value_loss(self.value_normalizer, values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        # critic도 actor와 마찬가지로 gradient norm을 clip할 수 있다.
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, shaped_info_loss, dist_entropy, actor_grad_norm, ratio, shaped_info_error, avg_pred_shaped_info

    # actor gradient가 이미 계산되어 있을 때 optimizer step만 따로 수행하는 helper다.
    # 일반적인 PPO 흐름에서는 ppo_update() 안에서 바로 actor step을 한다.
    def update_actor(self):
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

    # advantage = return - value prediction.
    # "예상보다 얼마나 좋았는가"를 나타내며 actor loss의 방향을 정한다.
    def compute_advantages(self, buffer):
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        return advantages

    # buffer에 모인 rollout 전체를 가지고 PPO 학습을 수행한다.
    # advantage 계산 -> mini-batch 생성 -> ppo_update 반복 -> logging 값 평균 순서로 진행된다.
    def train(self, buffer, turn_on=True):
        # 먼저 각 transition의 advantage를 계산한다.
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        # policy group별로 advantage normalization을 따로 할 수 있는 Overcooked/HSP용 옵션.
        if self._policy_group_normalization:
            other_policy_type = buffer.other_policy_type[:-1]
            for policy_type in range(5):
                mask = (other_policy_type == policy_type).astype(np.int32)
                if mask.sum() == 0:
                    continue
                mean_advantages = np.nanmean(advantages_copy[mask])
                std_advantages = np.nanmean(advantages_copy[mask])
                advantages[mask] = (advantages[mask] - mean_advantages) / (std_advantages + 1e-5)
                # print(f"policy_type={policy_type}, mean_adv={mean_advantages}, std_adv={std_advantages}, mask={mask.sum()}")
        else:
            # 기본 PPO처럼 전체 batch 기준으로 advantage를 평균 0, 표준편차 1에 가깝게 정규화한다.
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)


        # W&B/TensorBoard에 남길 학습 통계값들을 누적한다.
        train_info = defaultdict(float)

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        # 같은 rollout 데이터를 ppo_epoch번 재사용해서 여러 번 업데이트한다.
        for _ in range(self.ppo_epoch):
            # RNN을 쓰는지에 따라 buffer를 mini-batch로 자르는 방식이 달라진다.
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                # mini-batch 하나로 actor/critic을 업데이트한다.
                value_loss, critic_grad_norm, policy_loss, shaped_info_loss, dist_entropy, actor_grad_norm, ratio, shaped_info_error, avg_pred_shaped_info \
                    = self.ppo_update(sample, turn_on)

                # 여러 mini-batch의 logging 값들을 누적했다가 마지막에 평균낸다.
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['shaped_info_loss'] += shaped_info_loss.item()

                for c in range(len(shaped_info_error)):
                    train_info[f"shaped_info_error_class{c}"] += shaped_info_error[c]
                    train_info[f"pred_shaped_info_class{c}"] += avg_pred_shaped_info[c]
                
                if int(torch.__version__[2]) < 5:
                    train_info['actor_grad_norm'] += actor_grad_norm
                    train_info['critic_grad_norm'] += critic_grad_norm
                else:
                    train_info['actor_grad_norm'] += actor_grad_norm.item()
                    train_info['critic_grad_norm'] += critic_grad_norm.item()

                train_info['ratio'] += ratio.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        # 누적한 값들을 update 횟수로 나눠 평균 logging 값으로 만든다.
        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    # 학습 전 actor/critic을 train mode로 바꾼다.
    # dropout/batchnorm 같은 layer가 있으면 train/eval mode에 따라 동작이 달라진다.
    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    # rollout/evaluation 전 actor/critic을 eval mode로 바꾼다.
    # action을 뽑을 때는 gradient가 필요 없으므로 보통 eval mode로 둔다.
    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
    
    # policy 안의 actor/critic 전체를 지정한 device로 옮긴다.
    # policy pool에서 사용하지 않는 policy를 CPU로 내리거나 active policy를 GPU로 올릴 때 쓰인다.
    def to(self, device):
        self.policy.to(device)
