"""
전체 흐름에서의 단계:
    Risky Overcooked DDQN baseline의 agent/model 구성과 action 선택, Q-learning update 단계.

호출 위치:
    Trainer 또는 CurriculumTrainer가 model_object로 SelfPlay_QRE_OSA_CPT를 받아 초기화할 때 사용된다.

전체 역할:
    MDP encoding을 입력으로 받아 모든 joint action의 Q-value를 예측하는 DQN network를 만든다.
    QRE 기반으로 joint action을 선택하고, replay memory의 transition으로 policy network와 target network를 업데이트한다.
    CPT 버전에서는 다음 상태 가치 기대값 계산에 위험 민감한 prospect-theory 변환을 적용한다.
"""

import random
import matplotlib
import matplotlib.pyplot as plt
from risky_overcooked_py.mdp.actions import Action
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools
from risky_overcooked_rl.algorithms.DDQN.utils.memory import ReplayMemory_Prospect
from risky_overcooked_rl.utils.risk_sensitivity import CumulativeProspectTheory_Compiled
from risky_overcooked_rl.algorithms.DDQN import get_absolute_save_dir
from risky_overcooked_rl.utils.state_utils import invert_obs
from risky_overcooked_rl.algorithms.DDQN.utils.game_thoery import QuantalResponse_torch

import numpy as np
import warnings
import os
plt.ion()

class SelfPlay_QRE_OSA(object):
    """QRE action selection과 DDQN update를 함께 관리하는 기본 self-play agent wrapper."""

    @classmethod
    def from_file(cls,obs_shape, n_actions, agents_config, fname,save_dir = None):
        # 저장된 .pt weight를 찾아 같은 구조의 agent wrapper에 policy/target/checkpoint model로 로드한다.

        # instantiate base class -------------
        agents = cls(obs_shape, n_actions, agents_config)

        # find saved models absolute dir -------------
        dir = get_absolute_save_dir() if save_dir is None else save_dir

        # select file to load ---------------
        files = os.listdir(dir)
        files = [f for f in files if (fname in f and '.pt' in f)]
        if len(files) == 0: raise FileNotFoundError(f'No files found with name: {dir}{fname}')
        elif len(files) == 1: loads_fname = files[0]
        elif len(files) > 1:
            loads_fname = files[-1]
            warnings.warn(f'Multiple files found with fname: {loads_fname}. Using latest file...')

        else: raise ValueError('Unexpected error occurred')
        PATH = dir + loads_fname

        print(f'\n#########################################')
        print(f'Loading model from: {loads_fname}')
        print(f'#########################################\n')

        # Load file and update base class ---------
        try:
            loaded_model = torch.load(PATH, weights_only=True, map_location=agents_config['model']['device'])
        except:
            loaded_model = torch.load(PATH, weights_only=False, map_location=agents_config['model']['device'])
        agents.model.load_state_dict(loaded_model)
        agents.target.load_state_dict(loaded_model)
        agents.checkpoint_model.load_state_dict(loaded_model)
        # is_same = np.all([torch.all(agents.model.state_dict()[key] == agents.model.state_dict()[key]) for key in
        #         agents.model.state_dict().keys()])
        return agents

    def __init__(self, obs_shape, n_actions, agents_config,**kwargs):

        # Instatiate Base Config -------------
        # joint_action_dim은 두 agent의 action 조합 수이고, player_action_dim은 agent 1명의 action 수다.
        self.num_agents = 2
        self.joint_action_dim = n_actions
        self.player_action_dim = int(np.sqrt(n_actions))
        # joint_action_space: 실제 env.step에 넣을 (agent0_action, agent1_action) tuple 전체 목록.
        self.joint_action_space = list(itertools.product(Action.ALL_ACTIONS, repeat=2))

        # Parse Equilibrium Config -------------
        # QRE는 Q-value normal-form game을 받아 확률적 equilibrium action distribution을 계산한다.
        agents_config['equilibrium']['joint_action_space'] = self.joint_action_space
        self.QRE = QuantalResponse_torch(**agents_config['equilibrium'])

        # Parse NN Model config ---------------
        # model_config는 DQN network 구조, optimizer, target update, replay memory 크기를 결정한다.
        model_config = agents_config['model']
        self.clip_grad = model_config['clip_grad']
        self.num_hidden_layers = model_config['num_hidden_layers']
        self.size_hidden_layers = model_config['size_hidden_layers']
        self.learning_rate = model_config['lr']
        self.device = model_config['device']
        self.gamma = model_config['gamma']
        self.tau = model_config['tau']
        self.mem_size = model_config['replay_memory_size']
        self.minibatch_size = model_config['minibatch_size']

        # Define Memory
        # replay memory에는 state/action/reward와 stochastic next-state prospects가 저장된다.
        self._memory = ReplayMemory_Prospect(self.mem_size,self.device)
        self._memory_batch_size = self.minibatch_size

        # Define Model
        # model은 학습되는 Q-network, target은 TD target 계산용 network, checkpoint_model은 저장용 best snapshot이다.
        self.model = DQN_vector_feature(obs_shape, n_actions,**model_config).to( self.device)
        self.target = DQN_vector_feature(obs_shape, n_actions, **model_config).to(self.device)
        self.checkpoint_model = DQN_vector_feature(obs_shape, n_actions, **model_config).to(self.device)
        # self.model = DQN_vector_feature(obs_shape, n_actions,self.num_hidden_layers, self.size_hidden_layers).to(self.device)
        # self.target = DQN_vector_feature(obs_shape, n_actions,self.num_hidden_layers, self.size_hidden_layers).to(self.device)
        # self.checkpoint_model = DQN_vector_feature(obs_shape, n_actions,self.num_hidden_layers, self.size_hidden_layers).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.checkpoint_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, amsgrad=True)


    def update_checkpoint(self):
        # 현재 policy network weight를 checkpoint snapshot으로 복사한다.
        self.checkpoint_model.load_state_dict(self.model.state_dict())


    ###################################################
    # Nash Utils ######################################
    ###################################################

    def get_normal_form_game(self, obs, with_model=None):
        """ Batch compute the NF games for each observation"""
        # DQN output 36개 joint-action Q값을 2명 agent 관점의 normal-form game tensor로 재배열한다.
        batch_size = obs.shape[0]
        all_games = torch.zeros([batch_size, self.num_agents, self.player_action_dim, self.player_action_dim], device=self.device)
        for i in range(self.num_agents):
            # partner 관점 Q값은 observation을 invert해서 같은 network를 공유하는 self-play 형태로 계산한다.
            if i == 1: obs = invert_obs(obs)
            if with_model is not None: q_values = with_model(obs).detach()
            else: q_values = self.model(obs).detach()
            # if use_target: q_values = self.target(obs).detach()
            # else: q_values = self.model(obs).detach()
            q_values = q_values.reshape(batch_size, self.player_action_dim, self.player_action_dim)
            all_games[:, i, :, :] = q_values if i == 0 else torch.transpose(q_values, -1, -2)
        return all_games

    def get_expected_equilibrium_value(self, nf_games, dists):
        # QRE가 만든 각 agent action distribution을 joint distribution으로 바꿔 expected Q를 계산한다.
        ego, partner = 0, 1
        joint_dist_mat = torch.bmm(dists[:,ego].unsqueeze(-1), torch.transpose(dists[:,partner].unsqueeze(-1), -1, -2))
        value = torch.cat([torch.sum(nf_games[:, ego, :] * joint_dist_mat, dim=(-1, -2)).unsqueeze(-1),
                           torch.sum(nf_games[:, partner, :] * joint_dist_mat, dim=(-1, -2)).unsqueeze(-1)], dim=1)
        return value

    def choose_joint_action(self, obs, epsilon=0.0, feasible_JAs= None, debug=False):
        # epsilon 확률로 random joint action을 고르고, 그 외에는 QRE equilibrium에서 action을 고른다.
        sample = random.random()

        # Explore -------------------------------------
        if sample < epsilon:
            # feasible_JAs가 있으면 불가능한 joint action의 확률을 0으로 만들고 다시 normalize한다.
            action_probs = np.ones(self.joint_action_dim) / self.joint_action_dim
            if feasible_JAs is not None:
                action_probs = feasible_JAs*action_probs
                action_probs = action_probs/np.sum(action_probs)
            joint_action_idx = np.random.choice(np.arange(self.joint_action_dim), p=action_probs)
            joint_action = self.joint_action_space[joint_action_idx]

        # Exploit -------------------------------------
        else:
            with torch.no_grad():
                # 현재 observation의 Q값을 normal-form game으로 바꾼 뒤 QRE로 joint action을 선택한다.
                NF_Game = self.get_normal_form_game(obs)
                joint_action, joint_action_idx, action_probs = self.QRE.choose_actions(NF_Game)


        return joint_action, joint_action_idx, action_probs


    ###################################################
    # Update Utils ######################################
    ###################################################

    def update(self):
        # replay memory에서 minibatch를 뽑아 rational DDQN target으로 policy network를 한 번 업데이트한다.
        # if (
        #         # self.memory_len < self.mem_size/2
        #         len(self._memory) < self._memory_batch_size
        # ):  return 0

        # transitions = self.memory_sample()
        transitions = self._memory.sample(self._memory_batch_size)

        batch = self._memory.transition(*zip(*transitions))
        BATCH_SIZE = len(transitions)
        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        reward = torch.cat(batch.reward)
        done = torch.cat(batch.done)
        #
        # # Q-Learning with target network
        # q_value: 실제 선택했던 joint action index에 대한 현재 policy network의 Q(s,a).
        q_value = self.model(state).gather(1, action)

        # Batch calculate Q(s'|pi) and form mask for later condensation to expectation
        all_next_states,all_p_next_states,prospect_idxs = self.flatten_next_prospects(batch.next_prospects)

        # Compute equalib value for each outcome ---------------
        # NF_games = self.get_normal_form_game(torch.cat(all_next_states), use_target=True)
        NF_games = self.get_normal_form_game(torch.cat(all_next_states), with_model=self.target)
        all_next_a_dists, all_next_q_value = self.QRE.compute_EQ(NF_games)

        # Convert to numpy then back ----------------------------
        all_next_q_value = all_next_q_value[:, 0].reshape(-1, 1).detach().cpu().numpy()
        all_p_next_states = np.array(all_p_next_states).reshape(-1, 1)

        expected_q_value = self.prospect_value_expectations(reward = reward,
                                                            done = done,
                                                            prospect_masks=prospect_idxs,
                                                            prospect_next_q_values= all_next_q_value,
                                                            prospect_p_next_states= all_p_next_states)

        # Optimize ----------------
        # 현재 Q(s,a)가 target expectation에 가까워지도록 MSE loss로 DQN을 학습한다.
        loss = F.mse_loss(q_value, expected_q_value.detach(), reduction='none')
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        if self.clip_grad is not None:
            # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        return loss.item()

    def prospect_value_expectations(self,reward,done,prospect_masks,prospect_next_q_values,prospect_p_next_states):
        """Rational expectation used for modification when class inherited by CPT version
        - condenses prospects back into expecations of |batch_size|
        """

        # stochastic transition의 여러 next-state prospect를 확률 가중 평균해 TD target 하나로 압축한다.
        BATCH_SIZE = len(prospect_masks)
        # done = done.detach().cpu().numpy()
        # rewards = reward.detach().cpu().numpy()
        expected_td_targets = np.nan * np.ones([BATCH_SIZE, 1])
        for i in range(BATCH_SIZE):
            prospect_mask = prospect_masks[i]
            prospect_values = prospect_next_q_values[prospect_mask, :]
            prospect_probs = prospect_p_next_states[prospect_mask, :]
            prospect_td_targets = reward[i, :] + (self.gamma) * prospect_values * (1 - done[i, :])
            assert np.sum(prospect_probs) == 1, 'prospect probs should sum to 1'
            expected_td_targets[i] = np.sum(prospect_td_targets * prospect_probs)  # rational
        assert not np.any(np.isnan(expected_td_targets)), 'prospect expectations not filled'
        return torch.FloatTensor(expected_td_targets).to(self.device)


    def flatten_next_prospects(self,next_prospects):
        """
        Used for flattening next_state prospects into list of outcomes for batch processing
         - improve model-value prediction speed
         - condensed to back to |batch_size| after using expectation
        """

        # batch마다 다른 개수의 next-state outcomes를 하나의 큰 list로 펼치고, 원래 batch별 mask를 저장한다.
        all_next_states = []
        all_p_next_states = []
        prospect_idxs = []
        total_outcomes = 0
        for i, prospect in enumerate(next_prospects):
            n_outcomes = len(prospect)
            all_next_states += [outcome[1] for outcome in prospect]
            all_p_next_states += [outcome[2] for outcome in prospect]
            prospect_idxs.append(np.arange(total_outcomes, total_outcomes + n_outcomes))
            total_outcomes += n_outcomes
        return all_next_states,all_p_next_states,prospect_idxs


    def update_target(self):
        # tau 비율로 policy network weight를 target network에 soft update한다.
        # self.target = soft_update(self.model, self.target, self.tau)
        target_net_state_dict = self.target.state_dict()
        policy_net_state_dict = self.model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * (self.tau) + target_net_state_dict[key] * (1 - self.tau)
        self.target.load_state_dict(target_net_state_dict)
        return self.target

class SelfPlay_QRE_OSA_CPT(SelfPlay_QRE_OSA):
    """기본 DDQN target expectation 대신 CPT expectation을 쓰는 위험 민감 self-play agent."""

    def __init__(self, obs_shape, n_actions, agents_config, **kwargs):
        super().__init__(obs_shape, n_actions, agents_config, **kwargs)
        # self.CPT = CumulativeProspectTheory(**agents_config['cpt'])
        # CPT는 reward/value와 probability를 인간 위험 성향 파라미터로 왜곡해 expectation을 계산한다.
        self.CPT = CumulativeProspectTheory_Compiled(**agents_config['cpt'])

        self.frozen = False
        self.rational_ref_model = None
        self.qval_range = None

    def update(self):
        # replay memory minibatch로 CPT-DDQN target을 만들고 policy network를 한 번 업데이트한다.
        if self.frozen: raise ValueError('Model is frozen, cannot update')

        transitions = self._memory.sample(self._memory_batch_size)
        batch = self._memory.transition(*zip(*transitions))
        BATCH_SIZE = len(transitions)
        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        reward = np.vstack(batch.reward)

        # # Q-Learning with target network
        # q_value = self.model(state).gather(1, action)
        # qA는 모든 joint action의 현재 Q값이고, q_value는 실제 선택한 action의 Q값이다.
        qA = self.model(state)
        q_value = qA.gather(1, action)
        self.qval_range = f'[{torch.round(torch.min(qA))}, {torch.round(torch.max(qA))}]'  # (for logger)


        # Batch calculate Q(s'|pi) and form mask for later condensation to expectation --------------------------------
        all_next_states, all_p_next_states, prospect_idxs = self.flatten_next_prospects(batch.next_prospects)

        # Compute equalib value for each outcome ----------------------------------------------------------------------
        with torch.no_grad():
            NF_games = self.get_normal_form_game(torch.cat(all_next_states), with_model=self.target)

            # self.qval_range = f'[{torch.round(torch.min(NF_games,0))}, {torch.round(torch.max(qA,0))}]'  # (for logger)

            all_next_a_dists, all_next_q_value = self.QRE.compute_EQ(NF_games)

            # Convert to numpy then back ----------------------------
            all_next_q_value = all_next_q_value[:, 0].reshape(-1, 1).detach().cpu().numpy() # grab only ego values
            all_p_next_states = np.array(all_p_next_states).reshape(-1, 1)

            # rational 평균 대신 CPT expectation으로 next-state prospects를 하나의 target value로 압축한다.
            expected_value = self.CPT.expectation_samples(all_next_q_value, all_p_next_states,
                                                               prospect_idxs, reward, self.gamma)

            # self.qval_range = f'[{np.round(np.min(expected_value))}, {np.round(np.max(expected_value))}]' # (for logger)

            expected_value = torch.from_numpy(expected_value).float().cuda().reshape(q_value.shape)

        # Optimize ----------------
        # CPT target을 detach한 뒤 현재 Q(s,a)를 그 target에 맞추도록 network를 업데이트한다.
        loss = F.mse_loss(q_value, expected_value.detach(), reduction='none')
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        return loss.item()



class DQN_vector_feature(nn.Module):
    """Lossless vector observation을 받아 joint action별 Q-value를 출력하는 MLP Q-network."""

    def __init__(self, obs_shape, n_actions,num_hidden_layers,size_hidden_layers,**kwargs):
        # obs_shape[0]이 input dimension, n_actions가 output dimension이다.
        self.num_hidden_layers = num_hidden_layers
        self.size_hidden_layers = size_hidden_layers

        # config에 지정된 activation 이름을 실제 torch activation class로 변환한다.
        self.activation_function_name = kwargs.get('activation', 'LeakyReLU')
        if self.activation_function_name.lower() == 'ReLU'.lower():
            self.mlp_activation = nn.ReLU
        elif self.activation_function_name.lower() == 'LeakyReLU'.lower():
            self.mlp_activation = nn.LeakyReLU
        elif self.activation_function_name.lower() == 'ELU'.lower():
            self.mlp_activation = nn.ELU
        elif self.activation_function_name.lower() == 'Tanh'.lower():
            self.mlp_activation = nn.Tanh
        else:
            raise ValueError(f'Unknown activation function: {self.activation_function_name}')

        super(DQN_vector_feature, self).__init__()
        # hidden layer를 num_hidden_layers 기준으로 쌓고 마지막 layer는 joint-action Q값 n_actions개를 출력한다.
        layer_buffer = [ nn.Linear(obs_shape[0], self.size_hidden_layers),self.mlp_activation()]
        for i in range(1,self.num_hidden_layers-1):
            layer_buffer.extend([nn.Linear(self.size_hidden_layers, self.size_hidden_layers),self.mlp_activation()])
        layer_buffer.extend([nn.Linear(self.size_hidden_layers, n_actions)])

        self.layers = nn.Sequential(*layer_buffer)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def load_state_dict(self, state_dict,**kwargs):
        # Remove old keys that are not in the new model
        # 예전 checkpoint의 layer1/layer2/layer3 key가 있으면 제거해서 새 Sequential 구조와 맞춘다.
        try:
            super().load_state_dict(state_dict,**kwargs)
        except:
            state_dict.pop('layer1.weight', None)
            state_dict.pop('layer1.bias', None)
            state_dict.pop('layer2.weight', None)
            state_dict.pop('layer2.bias', None)
            state_dict.pop('layer3.weight', None)
            state_dict.pop('layer3.bias', None)
            super().load_state_dict(state_dict, **kwargs)

    def forward(self, x):
        # 입력 observation batch를 MLP에 통과시켜 각 joint action의 Q-value vector를 반환한다.
        for module in self.layers:
            x = module(x)
        return x
