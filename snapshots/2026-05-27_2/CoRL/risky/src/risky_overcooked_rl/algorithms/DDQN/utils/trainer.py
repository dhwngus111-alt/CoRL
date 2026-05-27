"""
전체 흐름에서의 단계:
    Risky Overcooked DDQN baseline의 공통 trainer 초기화/학습/평가/저장 단계.

호출 위치:
    non-curriculum DDQN trainer로 직접 사용되거나,
    CurriculumTrainer가 상속해서 공통 rollout, logging, checkpoint 로직을 재사용할 때 사용된다.

전체 역할:
    config를 runtime 값으로 보정하고 Overcooked env, DDQN agent model, logger를 초기화한다.
    각 rollout에서 replay memory 저장, model update, target update, test 평가, checkpoint/save 흐름을 관리한다.
"""

"""
Non-curriculum Trainer for Risky Overcooked RL
- Mostly deprecated and used as base class for CurriculumTrainer()
"""


import matplotlib.pyplot as plt
import numpy as np
from risky_overcooked_rl.utils.rl_logger_v2 import RLLogger_V2
from risky_overcooked_rl.utils.visualization import TrajectoryVisualizer, TrajectoryHeatmap
from risky_overcooked_rl.algorithms.DDQN import get_absolute_save_dir
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld,SoupState, ObjectState
from risky_overcooked_py.mdp.actions import Action
from itertools import count
from risky_overcooked_rl.utils.state_utils import FeasibleActionManager
import torch


import random
from datetime import datetime
debug = False
from collections import deque

class Trainer:
    # CurriculumTrainer가 상속해서 공통 학습/평가/저장 로직을 재사용하는 기본 trainer.
    def __init__(self,model_object,master_config):

        # Update configs with runtime values --------------
        # device: 모델과 tensor를 올릴 장치를 정하고, runtime config에도 같은 값을 기록한다.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # master_config: logger/save metadata가 실제 runtime 값과 맞도록 초기화 중에 보정된다.
        master_config['agents']['model']['device'] = self.device
        master_config['agents']['type'] = model_object.__name__
        master_config['trainer']['obs_shape'] = None # defined later
        self.timestamp = datetime.now().strftime("%m_%d_%Y-%H_%M")
        master_config['save']['date'] = self.timestamp
        self.master_config = master_config

        # Parse Sub Configurations -----------------------
        env_config = master_config['env']
        trainer_config = master_config['trainer']
        agents_config = master_config['agents']
        logger_config = master_config['logger']
        save_config = master_config['save']


        # Set random seeds -----------------
        # seed: numpy/torch/python random을 같은 seed로 맞춰 rollout 재현성을 높인다.
        seed = trainer_config['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        # Parse Trainer Configuration -------------------

        self.ITERATIONS = trainer_config['ITERATIONS']
        self.EXTRA_ITERATIONS = trainer_config['EXTRA_ITERATIONS']
        self.warmup_transitions = trainer_config['warmup_transitions']
        self.N_tests = trainer_config['N_tests']
        self.test_interval =  trainer_config['test_interval']
        enable_feasible_actions = trainer_config['feasible_actions']
        n_actions = trainer_config['joint_action_shape']

        # Instantiate MDP and Environment ----------------
        # env_config 값으로 layout, slip 확률, horizon, time cost를 실제 Overcooked env에 반영한다.
        layout = env_config['LAYOUT']
        time_cost = env_config['time_cost']
        p_slip = env_config['p_slip']
        horizon = env_config['HORIZON']
        neglect_boarders = env_config['neglect_boarders']
        overwrite = {}
        if p_slip != 'default' and p_slip != 'def':
            overwrite['p_slip'] = p_slip
        overwrite['neglect_boarders'] = neglect_boarders

        # self.mdp는 
        # layout의 구조, agent의 시작 위치, object들의 위치, action list, action 시 상태 어떻게 바뀌는지, reward shaping, 등등이 존재한다.
        self.mdp = OvercookedGridworld.from_layout_name(layout,**overwrite)  # trainer를 초기화할 때 MDP 즉, overcooked의 구조와 규칙을 불러옴 
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon,time_cost=time_cost) # 실제 episode를 굴리는 실행 wrapper
        obs_shape = self.mdp.get_lossless_encoding_vector_shape()
        master_config['trainer']['obs_shape'] = obs_shape

        self.LAYOUT = layout
        self.shared_rew = env_config['shared_rew']

        # Instantiate trainer configurations and Variables-------------
        # schedules: epsilon/random-start/reward-shaping 값을 iteration별 array로 미리 만들어 둔다.
        self.init_sched(trainer_config['schedules'])
        # feasible_action: 불가능한 joint action을 sampling 후보에서 제외할지 관리한다.
        self.feasible_action = FeasibleActionManager(self.env, enable=enable_feasible_actions)

        # Initialize Agent's policy and target networks ----------------
        loads = save_config['loads']
        self.rationality = agents_config['equilibrium']['rationality']
        self.test_rationality = agents_config['equilibrium']['rationality']
        self.cpt_params = agents_config['cpt']

        # model_config = agents_config['model']
        # model_config['rationality'] = agents_config['rationality'] # inherit rationality (minor change recommended)
        # Checkpointing/Saving utils ----------------
        self.checkpoint_score = -999
        self.checkpoint_mem = save_config['checkpoint_mem']
        # self.has_checkpointed = False
        self.train_rewards = deque(maxlen=self.checkpoint_mem)
        self.test_rewards = deque(maxlen=self.checkpoint_mem)
        self.fname_ext = save_config['fname_ext']
        self.save_dir = save_config['save_dir']
        self.wait_for_close = save_config['wait_for_close']
        self.auto_save = save_config['auto_save']
        self.save_with_heatmap = save_config['save_with_heatmap']
        self.state_history = None

        # Load/Instantiate Model ------------------------------
        # loads 값에 따라 rational baseline, latest checkpoint, 지정 파일, 또는 fresh model을 선택한다.
        if loads == 'rational':
            rational_fname = f"{layout}_pslip{str(p_slip).replace('.', '')}__rational__"
            # self.model = model_object.from_file(obs_shape, n_actions, config,rational_fname)
            self.model = model_object.from_file(obs_shape, n_actions, agents_config, rational_fname)
        elif loads == '':
            self.model = model_object(obs_shape, n_actions, agents_config)
        elif loads.lower() == 'latest':
            loads = self.get_fname(with_ext=False,with_timestamp=False)  # get the model (no date)
            self.model = model_object.from_file(obs_shape, n_actions, agents_config, loads)
        else:
            self.model = model_object.from_file(obs_shape, n_actions, agents_config, loads)
            # raise ValueError(f"Invalid load option: {config['loads']}")




        # Initiate Logger and Managers ----------------
        self._init_logger_()
        self.enable_report = logger_config['enable_report']
        if self.enable_report: self.print_config(master_config)



    def _init_logger_(self):
        # RLLogger: 학습 곡선, checkpoint 상태, preview/heatmap/save 버튼을 한 번에 관리한다.
        self.logger = RLLogger_V2(num_iters=self.ITERATIONS,wait_for_close = self.wait_for_close)

        self.logger.add_lineplot('test_reward', xlabel='', ylabel='$R_{test}$', filter_window=10, xtick=False)
        self.logger.add_lineplot('train_reward', xlabel='', ylabel='$R_{train}$', filter_window=50,xtick=False)
        self.logger.add_lineplot('loss', xlabel='iter', ylabel='$Loss$', filter_window=10)

        def get_curriculum():
            return self.curriculum.curr_curriculum_name if hasattr(self, 'curriculum') else None
        self.logger.add_annotation('test_reward', get_curriculum)

        # checkpoint watcher는 test reward plot을 기준으로 checkpoint callback을 호출한다.
        self.logger.add_checkpoint_watcher('test_reward', draw_on=['test_reward', 'train_reward', 'loss'], callback=self.checkpoint_callback)
        self.logger.add_settings(self.get_logger_display_data(self.master_config))

        # Create Watchers for training params
        # status callback들은 logger가 화면을 그릴 때 현재 trainer 상태를 즉시 읽는다.
        self.rshape_scale = self.rshape_sched[0]  # initial rshape scale
        self.epsilon = self.epsilon_sched[0]  # initial epsilon
        self.iteration = 0
        def get_rshape():  return self.rshape_scale
        def get_epsilon(): return self.epsilon
        def get_prog(): return f'{round(self.iteration/self.ITERATIONS,2)*100}%'
        def get_qval_range():
            return self.model.qval_range# if hasattr(self.model, 'qval_range') else None

        self.risk_taken = deque(maxlen=10)  # to track risk taken by agents
        def get_risk_taken():
            return np.mean(self.risk_taken).round(1) if len(self.risk_taken) > 0 else 0

        def extend_iteration(args):
            button =  self.logger.items[f'Extend Iters'].button
            self.EXTRA_ITERATIONS += 1000
            button.label.set_text(f'+{self.EXTRA_ITERATIONS} it')
        self.logger.add_status('$\epsilon$', callback=get_epsilon)
        self.logger.add_status('$r_{s}$',callback=get_rshape) # reward shaping scale
        # self.logger.add_status('Prog', callback=get_prog)
        self.logger.add_status('$Q\in$', callback=get_qval_range)
        # self.logger.add_status('Cur', callback=get_curriculum)
        self.logger.add_status('$N_\\rho$', callback=get_risk_taken)

        self.traj_visualizer = TrajectoryVisualizer(self.env)
        self.traj_heatmap = TrajectoryHeatmap(self.env)
        self.logger.add_button('Preview', callback=self.traj_visualizer.preview_qued_trajectory)
        self.logger.add_button('Heatmap', callback=self.traj_heatmap.preview)
        self.logger.add_button('Save ', callback=self.save)
        self.logger.add_button(f'Extend Iters', callback=extend_iteration)
        self.logger.items[f'Extend Iters'].button.label.set_text(f'+{self.EXTRA_ITERATIONS} it')

        # self.logger.add_toggle_button('wait_for_close', label='Wait For Close')
        # print()


    def get_fname(self, with_ext=True, with_timestamp=True):
        # CPT 파라미터가 완전 rational이면 파일명에 rational tag를 쓰고, 아니면 CPT 값을 붙인다.
        if (self.cpt_params['b'] == 0
                and self.cpt_params['lam'] == 1.0
                and self.cpt_params['eta_p'] == 1.0
                and self.cpt_params['eta_n'] == 1.0
                and self.cpt_params['delta_p'] == 1.0
                and self.cpt_params['delta_n'] == 1.0):
            h = f"{self.LAYOUT}" \
                f"_pslip{str(self.mdp.p_slip).replace('.', '')}" \
                f"__rational"
        else:
            h = f"{self.LAYOUT}" \
                f"_pslip{str(self.mdp.p_slip).replace('.', '')}" \
                f"__b{str(self.cpt_params['b']).replace('.', '')}" \
                f"_lam{str(self.cpt_params['lam']).replace('.', '')}" \
                f"_etap{str(self.cpt_params['eta_p']).replace('.', '')}" \
                f"_etan{str(self.cpt_params['eta_n']).replace('.', '')}" \
                f"_deltap{str(self.cpt_params['delta_p']).replace('.', '')}" \
                f"_deltan{str(self.cpt_params['delta_n']).replace('.', '')}"

        if with_timestamp:
            h += f"__{self.timestamp}"
        if with_ext:
            h = self.fname_ext + h
        return h

    @property
    def fname(self):
        return self.get_fname()


    def get_logger_display_data(self,master_config):
        # logger settings panel에 보여줄 주요 config/runtime 값을 사람이 읽기 좋은 label로 모은다.
        data = {}

        data['ALGORITHM'] = master_config['ALGORITHM']
        data['fname'] = self.fname
        if master_config['logger']['note'] != '':
            data['note'] = master_config['logger']['note']

        data['ENVIRONMENT'] = '================================'
        data['layout'] = f"{master_config['env']['LAYOUT']}"+" $p_{slip}$ = " + f"{self.mdp.p_slip}"
        # data['p_slip'] = self.mdp.p_slip #master_config['env']['p_slip']
        data['time_cost'] = self.env.time_cost  # master_config['env']['p_slip']

        # data['shared_rew'] = master_config['env']['shared_rew']
        data['neglect boarder'] = master_config['env']['neglect_boarders']

        data['TRAINER'] = '####################################'
        data['ITERATIONS'] = master_config['trainer']['ITERATIONS']
        data['OBS Shape'] = master_config['trainer']['obs_shape']
        # data['warmup_transitions'] = master_config['trainer']['warmup_transitions']
        # data['N_tests'] = master_config['trainer']['N_tests']
        # data['test_interval'] = master_config['trainer']['test_interval']
        # data['shared_rew'] = master_config['trainer']['shared_rew']
        # data['feasible_actions'] = master_config['trainer']['feasible_actions']
        # data['Auto Save'] = master_config['save']['auto_save']

        # data['SCHEDULES'] = '================================'
        data['epsilon'] = list(master_config['trainer']['schedules']['epsilon_sched'].values())
        # data['random start'] = list(master_config['trainer']['schedules']['rand_start_sched'].values())
        data['rew shaping'] = list(master_config['trainer']['schedules']['rshape_sched'].values())

        data['AGENTS'] = '####################################'
        data['type'] = master_config['agents']['type']
        # data['device'] = master_config['agents']['model']['device']

        # data['rationality'] = master_config['agents']['equilibrium']['rationality']
        # data['lr'] = master_config['agents']['model']['lr']
        # data['gamma'] = master_config['agents']['model']['gamma']
        # data['tau'] = master_config['agents']['model']['tau']
        #
        data[''] = f"$\lambda$={master_config['agents']['equilibrium']['rationality']}\t" \
                   f"$\\alpha$={master_config['agents']['model']['lr']}\t" \
                   f"$\gamma$={master_config['agents']['model']['gamma']}\t" \
                   f"$\\tau$={master_config['agents']['model']['tau']}"\


        data['Mem Size'] = master_config['agents']['model']['replay_memory_size']
        data['Minibatch Size'] = master_config['agents']['model']['minibatch_size']
        data['NN Shape'] = f"{self.model.model.size_hidden_layers}" \
                           f"x{self.model.model.num_hidden_layers}" \
                           f" with {self.model.model.activation_function_name} activation"
                           # f" with {master_config['agents']['model']['activation']} activation"
        data['Clip Grad'] = master_config['agents']['model']['clip_grad']
        # data['CPT'] = master_config['agents']['cpt']
        data['CPT'] =  f"$b$={master_config['agents']['cpt']['b']}, " \
                   f"$\ell$={master_config['agents']['cpt']['lam']}, " \
                   f"$\eta_p$={master_config['agents']['cpt']['eta_p']}, " \
                   f"$\eta_n$={master_config['agents']['cpt']['eta_n']}, " \
                   f"$\delta_p$={master_config['agents']['cpt']['delta_p']}, " \
                   f"$\delta_n$={master_config['agents']['cpt']['delta_n']}"
        data['MeanValRef'] = f"{master_config['agents']['cpt']['mean_value_ref']}"
        return data

    def print_config(self,config):
        for key, val in config.items():
            print(f'{key}={val}')
    def init_sched(self,schedules,eps_decay = 1,rshape_decay=1):
        # epsilon/random-start는 exponential schedule, reward shaping은 linear schedule로 생성한다.
        def exp_decay_sched(schedule,total_iterations):
            """ nonlinear time transformation where higher decay param ==> steeper decay"""
            START = schedule['start']
            END = schedule['end']
            DUR = schedule['duration']
            DECAY = schedule['decay']
            if DUR <= 1: # duration given in percent
                DUR = int(total_iterations * DUR)

            iters = np.arange(0, total_iterations)
            if START == END:  return np.ones(total_iterations) * START
            _sched = START * (END / START) ** ((iters/ DUR) ** (1 / DECAY))
            _sched = np.clip(_sched, END, None)
            return _sched

        def linear_decay_sched(schedule,total_iterations):
            START = schedule['start']
            END = schedule['end']
            DUR = schedule['duration']
            if DUR <= 1:  # duration given in percent
                DUR = int(total_iterations * DUR)

            _sched = np.hstack([np.linspace(START, END, DUR), END * np.ones(total_iterations - DUR)])
            return _sched

        # Scale Starting points
        schedules['rshape_sched']['start'] = schedules['rshape_sched']['start'] * rshape_decay
        schedules['epsilon_sched']['start'] = schedules['epsilon_sched']['start'] * eps_decay

        # Define schedules
        self.epsilon_sched = exp_decay_sched(schedules['epsilon_sched'], self.ITERATIONS)
        self.random_start_sched = exp_decay_sched(schedules['rand_start_sched'],self.ITERATIONS)
        self.rshape_sched = linear_decay_sched(schedules['rshape_sched'],self.ITERATIONS)

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # plt.plot( self.epsilon_sched )
        # plt.xlabel('Iterations')
        # plt.ylabel('Epsilon')
        # plt.title('Epsilon Schedule')
        # plt.grid()
        # plt.show()


    def run(self):
        # 기본 training loop: rollout으로 replay memory를 채우고 update한 뒤 주기적으로 test/checkpoint한다.
        train_rewards = []
        train_losses = []
        # Main training Loop
        for it in range(self.ITERATIONS):
            self.logger.spin()

            # Training Step ##########################################
            # Set Iteration parameters

            self._p_rand_start = self.random_start_sched[it]

            # Perform Rollout
            self.logger.start_iteration()

            # training_rollout은 한 episode를 실행하고 sparse/shaped reward와 loss/event 통계를 반환한다.
            cum_reward, cum_shaped_rewards,rollout_info =\
                self.training_rollout(it,rationality=self.rationality,
                                      epsilon = self.epsilon_sched[it],
                                      rshape_scale= self.rshape_sched[it],
                                      p_rand_start=self.random_start_sched[it])

            if it>1: self.model.scheduler.step() # updates learning rate scheduler
            self.model.update_target()  # performs soft update of target network
            self.logger.end_iteration()

            # slips = rollout_info['onion_slips'] + rollout_info['dish_slips'] + rollout_info['soup_slips']
            # risk/handoff metric은 env event_infos에서 누적된 행동 이벤트를 report용으로 요약한다.
            risks = rollout_info['onion_risked'] + rollout_info['dish_risked'] + rollout_info['soup_risked']
            handoffs = rollout_info['onion_handoff'] + rollout_info['dish_handoff'] + rollout_info['soup_handoff']
            if self.enable_report:
                print(f"Iteration {it} "
                      f"| train reward:{round(cum_reward, 3)} "
                      f"| shaped reward:{np.round(cum_shaped_rewards, 3)} "
                      f"| loss:{round(rollout_info['mean_loss'], 3)} "
                      # f"| slips:{slips} "
                      f"| risks:{risks} "
                      f"| handoffs:{handoffs} "
                      f" |"
                      f"| mem:{self.model.memory_len} "
                      f"| rshape:{round(self.rshape_sched[it], 3)} "
                      # f"| rat:{round(self.rationality_sched[it], 3)}"
                      f"| eps:{round(self.epsilon_sched[it], 3)} "
                      f"| LR={round(self.model.optimizer.param_groups[0]['lr'], 4)}"
                      f"| rstart={round(self.random_start_sched[it], 3)}"
                      )

            train_rewards.append(cum_reward + cum_shaped_rewards)
            train_losses.append(rollout_info['mean_loss'])

            # Testing Step ##########################################
            # time4test = (it % self.test_interval == 0 and it > 2)
            time4test = (it % self.test_interval == 0)
            if time4test:

                # Rollout test episodes ----------------------
                # test rollout은 epsilon=0 기본값으로 policy 평가를 수행하고 학습용 memory는 건드리지 않는다.
                test_rewards = []
                test_shaped_rewards = []
                for test in range(self.N_tests):
                    test_reward, test_shaped_reward, state_history, action_history, aprob_history =\
                        self.test_rollout(rationality=self.test_rationality)
                    test_rewards.append(test_reward)
                    test_shaped_rewards.append(test_shaped_reward)
                    # if not self.has_checkpointed:
                    #     self.traj_visualizer.que_trajectory(state_history)
                    #     self.traj_heatmap.que_trajectory(state_history)

                # Checkpointing ----------------------
                self.test_rewards.append(np.mean(test_rewards))  # for checkpointing
                self.train_rewards.append(np.mean(train_rewards))  # for checkpointing
                self.checkpoint(it,state_history)
                # if self.checkpoint(it):  # check if should checkpoint
                #     self.traj_visualizer.que_trajectory(state_history) # load preview of checkpointed trajectory
                #     self.traj_heatmap.que_trajectory(state_history)
                # Logging ----------------------
                self.logger.log(test_reward=[it, np.mean(test_rewards)],
                           train_reward=[it, np.mean(train_rewards)],
                           loss=[it, np.mean(train_losses)])
                self.logger.draw()
                if self.enable_report:
                    print(f"\nTest: | nTests= {self.N_tests} "
                          f"| Ave Reward = {np.mean(test_rewards)} "
                          f"| Ave Shaped Reward = {np.mean(test_shaped_rewards)}"
                          # f"\n{action_history}\n"#, f"{aprob_history[0]}\n"
                          )

                train_rewards = []
                train_losses = []
        self.logger.wait_for_close(enable=self.wait_for_close)
        # self.logger.wait_for_close(enable=True)
        if self.auto_save: self.save()

    def warmup(self,rshape_scale=1, epsilon=1):
        """Loads random transitions into memory"""

        # warmup rollout은 random start state에서 transition을 만들어 replay memory에 미리 넣는다.
        while len(self.model._memory) > self.warmup_transitions:
            self.logger.spin()
            self.env.state = self.mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.1)()

            old_state = self.env.state.deepcopy()
            obs = self.mdp.get_lossless_encoding_vector_astensor(self.env.state, device=self.device).unsqueeze(0)
            feasible_JAs = self.feasible_action.get_feasible_joint_actions(self.env.state, as_joint_idx=True)
            joint_action, joint_action_idx, action_probs = self.model.choose_joint_action(obs,
                                                                                          epsilon=epsilon,
                                                                                          feasible_JAs=feasible_JAs)
            next_state, reward, done, info = self.env.step(joint_action, get_mdp_info=True)
            # 현재 state와 action에서 가능한 다음 상태 후보를 계산한다.
            next_state_prospects = self.mdp.one_step_lookahead(old_state,  # must be called after step....
                                                               joint_action=Action.ALL_JOINT_ACTIONS[joint_action_idx],
                                                               as_tensor=True, device=self.device)

            # Track reward traces
            shaped_rewards = rshape_scale * np.array(info["shaped_r_by_agent"])
            total_rewards = np.array([reward + shaped_rewards]).flatten()

            # Store in memory ----------------
            self.model._memory.double_push(state=obs,
                                           action=joint_action_idx,
                                           rewards=total_rewards,
                                           next_prospects=next_state_prospects,
                                           done=done)



    ################################################################
    # Train/Test Rollouts   ########################################
    ################################################################
    def training_rollout(self,it,rationality,epsilon,rshape_scale,p_rand_start=0):

        # rollout 시작 전에 현재 rationality와 env 초기 상태를 설정한다.
        self.model.rationality = rationality
        self.env.reset()

        # Random start state if specified
        # if it / self.ITERATIONS < self.perc_random_start:
        if np.random.sample() < p_rand_start:
            self.env.state = self.random_start_state()

        losses = []
        cum_reward = 0
        cum_shaped_reward = np.zeros(2)

        rollout_info = {
            'onion_risked': np.zeros([1, 2]),
            'onion_pickup': np.zeros([1, 2]),
            'onion_drop': np.zeros([1, 2]),
            'dish_risked': np.zeros([1, 2]),
            'dish_pickup': np.zeros([1, 2]),
            'dish_drop': np.zeros([1, 2]),
            'soup_pickup': np.zeros([1, 2]),
            'soup_delivery': np.zeros([1, 2]),

            'soup_risked': np.zeros([1, 2]),
            'onion_slip': np.zeros([1, 2]),
            'dish_slip': np.zeros([1, 2]),
            'soup_slip': np.zeros([1, 2]),
            'onion_handoff': np.zeros([1, 2]),
            'dish_handoff': np.zeros([1, 2]),
            'soup_handoff': np.zeros([1, 2]),
            'mean_loss': 0
        }

        # episode가 done 될 때까지 joint action 선택, env step, replay 저장, model update를 반복한다.
        for t in count():
            obs = self.mdp.get_lossless_encoding_vector_astensor(self.env.state,device=self.device).unsqueeze(0)
            feasible_JAs = self.feasible_action.get_feasible_joint_actions(self.env.state,as_joint_idx=True)
            joint_action, joint_action_idx, action_probs = self.model.choose_joint_action(obs,
                                                                                          epsilon=epsilon,
                                                                                          feasible_JAs = feasible_JAs)
            next_state_prospects = self.mdp.one_step_lookahead(self.env.state.deepcopy(),
                                                               joint_action=Action.ALL_JOINT_ACTIONS[joint_action_idx],
                                                               as_tensor=True, device=self.device)
            next_state, reward, done, info = self.env.step(joint_action,get_mdp_info=True)

            for key in rollout_info.keys():
                if key not in ['mean_loss']:
                    rollout_info[key] += np.array(info['mdp_info']['event_infos'][key])

            # Track reward traces
            shaped_rewards = rshape_scale * np.array(info["shaped_r_by_agent"])
            if self.shared_rew: shaped_rewards = np.mean(shaped_rewards)*np.ones(2)
            total_rewards =  np.array([reward + shaped_rewards]).flatten()
            cum_reward += reward
            cum_shaped_reward += shaped_rewards

            # Store in memory ----------------
            self.model.memory_double_push(state=obs,
                                        action=joint_action_idx,
                                        rewards = total_rewards,
                                        next_prospects=next_state_prospects,
                                        done = done)
            # Update model ----------------
            loss = self.model.update()
            if loss is not None: losses.append(loss)
            if done:  break
            self.env.state = next_state
        rollout_info['mean_loss'] = np.mean(losses)
        return cum_reward, cum_shaped_reward, rollout_info



    def test_rollout(self,rationality,epsilon=0,rshape_scale=1,get_info = False):

        # get_info=True이면 training rollout과 같은 event counter를 평가 episode에서도 수집한다.
        if get_info:
            rollout_info = {
                'onion_risked': np.zeros([1, 2]),
                'onion_pickup': np.zeros([1, 2]),
                'onion_drop': np.zeros([1, 2]),
                'dish_risked': np.zeros([1, 2]),
                'dish_pickup': np.zeros([1, 2]),
                'dish_drop': np.zeros([1, 2]),
                'soup_pickup': np.zeros([1, 2]),
                'soup_delivery': np.zeros([1, 2]),

                'soup_risked': np.zeros([1, 2]),
                'onion_slip': np.zeros([1, 2]),
                'dish_slip': np.zeros([1, 2]),
                'soup_slip': np.zeros([1, 2]),
                'onion_handoff': np.zeros([1, 2]),
                'dish_handoff': np.zeros([1, 2]),
                'soup_handoff': np.zeros([1, 2]),

            }
        # evaluation 동안에는 policy/target network를 eval mode로 바꾸고, 끝나면 train mode로 복원한다.
        self.model.model.eval()
        self.model.target.eval()
        self.model.rationality = rationality
        self.env.reset()

        test_reward = 0
        test_shaped_reward = 0
        state_history = [self.env.state.deepcopy()]
        action_history = []
        aprob_history = []

        for t in count():
            obs = self.mdp.get_lossless_encoding_vector_astensor(self.env.state,device=self.device).unsqueeze(0)
            joint_action, joint_action_idx, action_probs = self.model.choose_joint_action(obs, epsilon=epsilon)
            next_state, reward, done, info = self.env.step(joint_action) # 이게 실제 rollout

            # Track reward traces
            test_reward += reward
            test_shaped_reward += rshape_scale*np.mean(info["shaped_r_by_agent"])*np.ones(2)

            # Track state-action history
            action_history.append(joint_action_idx)
            aprob_history.append(action_probs)
            state_history.append(next_state.deepcopy())

            if get_info:
                for key in rollout_info.keys():
                    if key not in ['mean_loss']:
                        rollout_info[key] += np.array(info['mdp_info']['event_infos'][key])

            if done:  break
            self.env.state = next_state
        self.model.model.train()
        self.model.target.train()
        if get_info: return test_reward, test_shaped_reward, state_history, action_history, aprob_history, rollout_info
        else: return test_reward, test_shaped_reward, state_history, action_history, aprob_history

    ################################################################
    # State Randomizer #############################################
    ################################################################
    def random_start_state(self):
        # random-start curriculum이 아닐 때도 사용할 수 있는 일반 random state generator.
        state = self.add_random_start_loc()
        state = self.add_random_start_pot_state(state)
        state = self.add_random_held_obj(state)
        # state = self.add_random_counter_state(state)
        return state
    def add_random_start_loc(self):
        random_state = self.mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.0)()
        return random_state
    def add_random_start_pot_state(self,state,rnd_obj_prob_thresh=0.5):
        # 빈 pot마다 확률적으로 partially/fully filled soup 상태를 넣는다.
        pots = self.mdp.get_pot_states(state)["empty"]
        for pot_loc in pots:
            p = np.random.rand()
            if p < rnd_obj_prob_thresh:
                n = int(np.random.randint(low=1, high=3))
                cooking_tick = np.random.randint(0, 20) if n == 3 else -1
                # cooking_tick = 0 if n == 3 else -1
                state.objects[pot_loc] = SoupState.get_soup(
                    pot_loc,
                    num_onions=n,
                    num_tomatoes=0,
                    cooking_tick=cooking_tick,
                )
        return state
    def add_random_held_obj(self,state,rnd_obj_prob_thresh=0.5):
        # For each player, add a random object with prob rnd_obj_prob_thresh
        # player별로 onion/dish/soup 중 하나를 확률적으로 들고 시작하게 만든다.
        for player in state.players:
            p = np.random.rand()
            if p < rnd_obj_prob_thresh:
                # Different objects have different probabilities
                obj = np.random.choice(["onion", "dish", "soup"], p=[0.6, 0.2, 0.2])
                self.add_held_obj(player, obj)
        return state
    def add_random_counter_state(self, state, rnd_obj_prob_thresh=0.025):
        # reachable counter 위에 낮은 확률로 object를 배치한다.
        counters = self.mdp.reachable_counters
        for counter_loc in counters:
            p = np.random.rand()
            if p < rnd_obj_prob_thresh:
                obj = np.random.choice(["onion", "dish", "soup"], p=[0.6, 0.2, 0.2])
                if obj == "soup":
                    state.add_object(SoupState.get_soup(counter_loc, num_onions=3, num_tomatoes=0, finished=True))
                else:
                    state.add_object(ObjectState(obj, counter_loc))
        return state
    def add_held_obj(self,player,obj):
        if obj == "soup": player.set_object(SoupState.get_soup(player.position, num_onions=3, num_tomatoes=0, finished=True))
        else: player.set_object(ObjectState(obj, player.position))
        return player

    ################################################################
    # Save Utils       #############################################
    ################################################################
    def checkpoint_callback(self):
        # logger checkpoint watcher에서 호출되며, 현재 checkpoint model과 trajectory preview를 갱신한다.
        self.model.update_checkpoint()
        if self.state_history is not None:
            self.traj_visualizer.que_trajectory(self.state_history)
            self.traj_heatmap.que_trajectory(self.state_history)


    def checkpoint(self,it, state_history): #TODO: Delete since LoggerV2

        score = np.mean(self.test_rewards)
        if score > self.checkpoint_score:
            if self.enable_report:
                print(f'\nCheckpointing model at iteration {it} with score {score}...\n')
            self.model.update_checkpoint()
            self.logger.update_checkpiont_line(it)
            self.checkpoint_score = score
            self.has_checkpointed = True

            self.traj_visualizer.que_trajectory(state_history)
            self.traj_heatmap.que_trajectory(state_history)
            return True
        #########################################
        # if len(self.train_rewards) == self.checkpoint_mem:
        #     ave_train = np.mean(self.train_rewards)
        #     ave_test = np.mean(self.test_rewards)
        #     # score = (ave_train + ave_test)/2
        #     score = ave_test
        #     if score > self.min_checkpoint_score and score > self.checkpoint_score:
        #         print(f'\nCheckpointing model at iteration {it} with score {score}...\n')
        #         self.model.update_checkpoint()
        #         self.logger.update_checkpiont_line(it)
        #         self.checkpoint_score = score
        #         self.has_checkpointed = True
        #         return True
        # return False


    def package_model_info(self,rational=False):
        # saved model metadata에 들어갈 layout/slip/CPT 파라미터를 구성한다.
        model_info = {
            'timestamp': self.timestamp,
            'layout': self.LAYOUT,
            'p_slip': self.mdp.p_slip,
            'b': self.cpt_params['b'] if not rational else 0.0,
            'lam': self.cpt_params['lam'] if not rational else 1.0,
            'eta_p': self.cpt_params['eta_p'] if not rational else 1.0,
            'eta_n': self.cpt_params['eta_n'] if not rational else 1.0,
            'delta_p': self.cpt_params['delta_p'] if not rational else 1.0,
            'delta_n': self.cpt_params['delta_n'] if not rational else 1.0,
        }
        return model_info

    def save(self,*args, save_model=True, save_fig=True):
        # find saved models absolute dir -------------


        # model weight, training figure, optional heatmap을 같은 fname 기준으로 저장한다.
        print(f'\n\nSaving model to {self.fname}')
        dir = get_absolute_save_dir(path=self.save_dir)
        img_fname = f"{dir}{self.fname}.png"

        if save_model:
            torch.save(self.model.checkpoint_model.state_dict(), dir + f"{self.fname}.pt")

        if save_fig:
            print('\tSaving training fig to ', img_fname)
            self.logger.save_fig(img_fname)

        if self.save_with_heatmap:
            print('\tSaving heatmap to ', img_fname)
            if (self.traj_heatmap.qued_trajectory is None
                    and self.state_history is not None):
                self.traj_heatmap.que_trajectory(self.state_history)

            elif self.traj_heatmap.qued_trajectory is None and self.state_history is None:
                print("\tNo heatmap found during saving. Skipping...")
                print(f'\tfinished {self.fname}\n\n')
                return
            fig = plt.figure(figsize=(6.5,5.25),constrained_layout=True)
            train_fig, hm_fig = fig.subfigures(2,1,height_ratios=(2,1))




            # add training img
            train_ax = train_fig.add_subplot(111)
            img = plt.imread(img_fname)
            train_ax.imshow(img)
            train_ax.set_xticks([])
            train_ax.set_yticks([])
            train_ax.set_title(self.fname,fontsize=6)

            # add heatmaps
            ax_dict = {
                'onion': hm_fig.add_subplot(131),
                'dish': hm_fig.add_subplot(132),
                'soup': hm_fig.add_subplot(133)
            }
            self.traj_heatmap.custom_preview(ax_dict)
            for key, ax in ax_dict.items():
                # ax.axis('off')
                # ax.set_title(key)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(key)

            fig.savefig(img_fname)
            plt.close(fig)
        print(f'\tfinished {self.fname}\n\n')






if __name__ == "__main__":
    raise NotImplementedError
