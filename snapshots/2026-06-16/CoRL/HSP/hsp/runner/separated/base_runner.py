    
import time
from typing import Any, Dict
import wandb
import os
import socket
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter
import pickle

from hsp.utils.separated_buffer import SeparatedReplayBuffer
from hsp.utils.util import update_linear_schedule

import psutil
import slackweb
import socket
webhook_url = " https://hooks.slack.com/services/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"

def _t2n(x):
    return x.detach().cpu().numpy()

def make_trainer_policy_cls(algorithm_name, use_single_network=False):
    # algorithm_name에 맞는 학습 알고리즘 클래스와 policy 클래스를 고른다.
    # HSP Stage 1 shell script에서는 algorithm_name="mappo"라서
    # R_MAPPO(학습기)와 R_MAPPOPolicy(actor/critic 네트워크)를 사용한다.
    if "mappo" in algorithm_name:
        if use_single_network:
            from hsp.algorithms.r_mappo_single.r_mappo_single import R_MAPPO as TrainAlgo
            from hsp.algorithms.r_mappo_single.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        else:
            from hsp.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from hsp.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
    elif "mappg" in algorithm_name:
        if use_single_network:
            from hsp.algorithms.r_mappg_single.r_mappg_single import R_MAPPG as TrainAlgo
            from hsp.algorithms.r_mappg_single.algorithm.rMAPPGPolicy import R_MAPPGPolicy as Policy
        else:
            from hsp.algorithms.r_mappg.r_mappg import R_MAPPG as TrainAlgo
            from hsp.algorithms.r_mappg.algorithm.rMAPPGPolicy import R_MAPPGPolicy as Policy
    elif "overcooked_bc" == algorithm_name:
        from hsp.envs.overcooked.human_aware_rl.imitation.behavior_cloning import BehaviorCloneTrainer as TrainAlgo
        from hsp.envs.overcooked.human_aware_rl.imitation.behavior_cloning import BehaviorClonePolicy as Policy
    elif "ft" in algorithm_name:
        print("use frontier-based algorithm")
    elif "population" == algorithm_name:
        from hsp.algorithms.population.trainer_pool import TrainerPool as TrainAlgo
        from hsp.algorithms.population.policy_pool import PolicyPool as Policy
    elif algorithm_name == "mep":
        from hsp.algorithms.population.mep import MEP_Trainer as TrainAlgo
        from hsp.algorithms.population.policy_pool import PolicyPool as Policy
    elif algorithm_name == "traj":
        from hsp.algorithms.population.traj import Traj_Trainer as TrainAlgo
        from hsp.algorithms.population.policy_pool import PolicyPool as Policy
    else:
        raise NotImplementedError
    return TrainAlgo, Policy



class Runner(object):
    def __init__(self, config):

        # train_overcooked_hsp.py에서 만든 config dict를 Runner 내부 변수로 푼다.
        # envs는 병렬 Overcooked 학습 환경, device는 cuda/cpu, num_agents는 보통 2다.
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.use_single_network = self.all_args.use_single_network
        self.recurrent_N = self.all_args.recurrent_N  #  policy가 RNN/GRU 같은 recurrent policy를 쓸 때, hidden state가 몇 층짜리인가?

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.run_dir = self.save_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        # 여기서 실제로 사용할 policy/trainer 타입이 결정된다.
        # mappo이면 Policy=R_MAPPOPolicy, TrainAlgo=R_MAPPO가 된다.
        TrainAlgo, Policy = make_trainer_policy_cls(self.algorithm_name, use_single_network=self.use_single_network)

        # dump policy config to allow loading population in yaml form
        # policy_config.pkl은 나중에 policy_pool yaml에서 policy를 다시 불러올 때 쓰는 설정 파일이다.
        
        # 여기서는 observation_space와 shared_observation_space가 나옴
        # 전자는 agent가 actor로 action을 고를 때 보는 관측 공간
        # 후자는 critic이 value를 계산할 때 보는 centralized 관측 공간 => MAPPO의 centralized critic
        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0] 
        
        # 얘가 Stage 2에서 어떤 파트너와 플레이 중인지 critic에게 알려주려고 policy id feature를 share observation에 추가하는 코드
        if hasattr(self.all_args, "use_agent_policy_id") and self.all_args.use_agent_policy_id:    
            share_observation_space = share_observation_space[:-1] + [[self.all_args.num_agents, 1, False, config['policy_pool_size']]] +  [share_observation_space[-1]] # 여기 가운데 있는게 policy id feature에 대한 term이다.
            share_observation_space[0] += self.all_args.num_agents
        self.policy_config = (
            self.all_args,   # 여기는 policy를 다시 만들 때 필요한 설정이 들어감
            self.envs.observation_space[0], # actor의 입력 크기/형태 
            share_observation_space,  # critic 입력 크기/형태
            self.envs.action_space[0] # action의 개수
            ) 
        policy_config_path = os.path.join(self.run_dir, 'policy_config.pkl')  # 학습 결과 폴더에 policy_config.pkl이 생김 =>
        pickle.dump(self.policy_config, open(policy_config_path, "wb"))
        print(f"Pickle dump policy config at {policy_config_path}")

        # MEP나 population학습 쪽에서 사용
        if self.algorithm_name in ["population", "mep", "traj"]:
            # population/mep/traj 계열은 PolicyPool 기반이라 policy 하나가 population을 관리한다.
            # HSP Stage 1의 mappo 경로는 아래 else branch로 간다.
            # policy network
            self.policy = Policy(self.all_args, # 여기는 policypool
                                self.envs.observation_space[0],
                                share_observation_space,
                                self.envs.action_space[0],
                                device = self.device)

            if self.model_dir is not None:
                self.restore()

            # algorithm
            self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)

        else: # mappo는 else 부분으로 넘어감
            # separated runner의 핵심:
            # agent0, agent1이 policy를 공유하지 않고 각각 자기 policy network를 가진다.
            # HSP S1에서는 w0/w1처럼 서로 다른 reward weight를 받을 수 있으므로 이 구조가 필요하다.
            self.policy = []  
            for agent_id in range(self.num_agents):
                #print(len(self.envs.share_observation_space))
                #print(len(self.envs.observation_space))
                share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
                # policy network
                po = Policy(self.all_args,
                            self.envs.observation_space[agent_id],
                            share_observation_space,
                            self.envs.action_space[agent_id],
                            device = self.device)
                self.policy.append(po)

            if self.model_dir is not None:
                self.restore()

            # agent마다 policy trainer buffer를 따로 만든다.
            # trainer는 PPO 업데이트를 담당하고, buffer는 rollout 데이터를 저장한다.
            self.trainer = [] 
            self.buffer = [] # [0]은 agent0 rollout 저장소 // [1]은 agent1 rollout 저장소 // 각 에이전트는 policy, trainer, buffer 를 담고 있다.
            for agent_id in range(self.num_agents):
                # algorithm
                tr = TrainAlgo(self.all_args, self.policy[agent_id], device = self.device)
                # buffer
                share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
                #print("Base runner", agent_id, share_observation_space)
                bu = SeparatedReplayBuffer(self.all_args,
                                        self.envs.observation_space[agent_id],
                                        share_observation_space,
                                        self.envs.action_space[agent_id])
                self.buffer.append(bu)
                self.trainer.append(tr)
            
    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        # 각 agent의 buffer에 대해 bootstrap value를 구하고 return/advantage를 계산한다.
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1], 
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        # 각 agent가 자기 buffer로 자기 policy를 따로 업데이트한다.
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training() ## PPO Loss 계산
            train_info = self.trainer[agent_id].train(self.buffer[agent_id]) # 여기서 학습 호출함. 각 agent의 buffer에 쌓인 rollout data 로 PPO를 업데이트한다.
            train_infos.append(train_info)       
            self.buffer[agent_id].after_update()
        self.log_system()
        return train_infos

    # 얘가 각 agent의 모델 checkpoint를 저장하는 함수
    # steps=25  -> _25.pt
    # steps=None -> .pt
    def save(self, steps=None):
        postfix = f"_{steps}.pt" if steps else ".pt"
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model = self.trainer[agent_id].policy.model
                torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_agent" + str(agent_id) + postfix)
            else: # agent 2개니까 이 쪽으로 저장
                policy_actor = self.trainer[agent_id].policy.actor #  action 뽑히는 네트워크
                torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + postfix)
                policy_critic = self.trainer[agent_id].policy.critic # value 예측하는 네트워크
                torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + postfix)

    # 저장된 checkpoint를 다시 불러오는 함수
    def restore(self):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model_state_dict = torch.load(str(self.model_dir) + '/model_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].model.load_state_dict(policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
                if not self.use_render:
                    policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
                    self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)




    #여기 아래는 log에 상태 기록하는 것들임
    def log_train(self, train_infos, total_num_steps): 
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log(self, infos: Dict[str, Any], step):
        if self.use_wandb:
            wandb.log(infos, step=step)
        else:
            [self.writter.log(k, v, step) for k, v in infos.items()]

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def log_system(self):
        # RRAM
        mem = psutil.virtual_memory()
        total_mem = float(mem.total) / 1024 / 1024 / 1024
        used_mem = float(mem.used) / 1024 / 1024 / 1024
        if used_mem/total_mem > 0.95:
            slack = slackweb.Slack(url=webhook_url)
            host_name = socket.gethostname()
            slack.notify(text="Host {}: occupied memory is *{:.2f}*%!".format(host_name, used_mem/total_mem*100))
