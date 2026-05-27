import numpy as np
import torch
from hsp.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from hsp.utils.util import update_linear_schedule


class R_MAPPOPolicy:
    # Actor/Critic 네트워크와 각각의 optimizer를 묶어 관리하는 policy wrapper.
    # trainer는 이 객체를 통해 action sampling, value prediction, PPO 평가를 호출한다.
    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):

        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    # 학습 episode 진행도에 맞춰 actor/critic learning rate를 선형으로 줄인다.
    # runner에서 매 episode 또는 update마다 호출되는 scheduler 역할이다.
    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    # rollout 중에 actor로 action과 log prob을 뽑고, critic으로 value를 예측한다.
    # buffer에 저장할 values/actions/log_probs/rnn_states를 한 번에 반환한다.
    def get_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False, task_id=None, **kwargs):
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic) # 얘는 action을 뽑고
        values, rnn_states_critic = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id) # 얘는 value를 뽑는다.
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    # critic만 호출해서 현재 shared observation의 value를 다시 계산한다.
    # rollout 마지막 state의 bootstrap value나 value-only 평가에 쓰인다.
    def get_values(self, share_obs, rnn_states_critic, masks, task_id=None):
        values, _ = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id)
        return values

    # PPO update 때 old action을 현재 actor 기준으로 다시 평가한다.
    # 새 action_log_probs, entropy, critic value를 반환해서 policy/value loss 계산에 쓴다.
    def evaluate_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, available_actions=None, active_masks=None, task_id=None):
        action_log_probs, dist_entropy, policy_values, pred_shaped_info = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, available_actions, active_masks)
        values, _ = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id)
        return values, action_log_probs, dist_entropy, policy_values, pred_shaped_info

    # trajectory/model-based 계열 코드에서 transition 단위로 action을 평가한다.
    # evaluate_actions와 비슷하지만 actor RNN state까지 다음 상태로 넘겨준다.
    def evaluate_transitions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, available_actions=None, active_masks=None, task_id=None):
        action_log_probs, dist_entropy, policy_values, rnn_states_actor = self.actor.evaluate_transitions(obs, rnn_states_actor, action, masks, available_actions, active_masks)
        values, _ = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id)
        return values, action_log_probs, dist_entropy, policy_values, rnn_states_actor

    # 평가/rendering처럼 value가 필요 없을 때 actor action만 뽑는다.
    # deterministic=True면 보통 greedy action, False면 sampling action을 반환한다.
    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, **kwargs):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor

    # 현재 observation에서 가능한 action들의 확률분포를 반환한다.
    # policy 분석, action probability logging, scripted comparison 등에 쓸 수 있다.
    def get_probs(self, obs, rnn_states_actor, masks, available_actions=None):
        action_probs, rnn_states_actor = self.actor.get_probs(obs, rnn_states_actor, masks, available_actions=available_actions)
        return action_probs, rnn_states_actor
    
    # 특정 action 하나의 probability를 현재 actor 기준으로 계산한다.
    # 내부에서는 log prob을 받은 뒤 exp를 취해서 probability로 바꿔 반환한다.
    def get_action_probs(self, obs, rnn_states_actor, action, masks, available_actions=None, active_masks=None):
        action_log_probs, _, _, rnn_states_actor = self.actor.get_action_probs(obs, rnn_states_actor, action, masks, available_actions, active_masks)
        return action_log_probs.exp(), rnn_states_actor

    # 저장된 actor/critic checkpoint를 불러와 policy weight를 복원한다.
    # ckpt_path dict에 'actor', 'critic' key가 있는 것만 선택적으로 load한다.
    def load_checkpoint(self, ckpt_path):
        if 'actor' in ckpt_path:
            self.actor.load_state_dict(torch.load(ckpt_path["actor"], map_location=self.device))
        if 'critic' in ckpt_path:
            self.critic.load_state_dict(torch.load(ckpt_path["critic"], map_location=self.device))
    
    # actor와 critic module을 지정한 device(cpu/cuda)로 옮긴다.
    # policy pool에서 쓰지 않는 policy를 CPU로 내리거나 학습 policy를 GPU로 올릴 때 쓰인다.
    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)

    # rollout/evaluation 전에 actor와 critic을 eval mode로 전환한다.
    # dropout/batchnorm 같은 layer가 있다면 학습 모드와 다르게 동작하게 된다.
    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()
