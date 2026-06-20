#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path

import torch

from hsp.config import get_config

from hsp.envs.overcooked.Overcooked_Env import Overcooked
from hsp.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new
from hsp.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

def make_train_env(all_args, run_dir): # 여기서 환경 묶음
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked": # Overcooked인지 확인한다.
                if all_args.overcooked_version == "old": # 버전을 확인 
                    env = Overcooked(all_args, run_dir) #
                else:
                    env = Overcooked_new(all_args, run_dir)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000) # 각 env에 시드를 부여
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                env = Overcooked(all_args, run_dir)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument("--layout_name", type=str, default='cramped_room', help="Name of Submap, 40+ in choice. See /src/data/layouts/.")
    parser.add_argument('--num_agents', type=int, default=1, help="number of players")
    parser.add_argument("--initial_reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_horizon", type=int, default=2.5e6, help="Shaping factor of potential dense reward.")
    parser.add_argument("--use_phi", default=False, action='store_true', help="While existing other agent like planning or human model, use an index to fix the main RL-policy agent.")
    parser.add_argument("--use_hsp", default=False, action='store_true') 
    parser.add_argument("--random_index", default=False, action='store_true') 
    parser.add_argument("--w0", type=str, default="1,1,1,1", help="Weight vector of dense reward 0 in overcooked env.")
    parser.add_argument("--w1", type=str, default="1,1,1,1", help="Weight vector of dense reward 1 in overcooked env.") 
    parser.add_argument("--predict_other_shaped_info", default=False, action='store_true', help="Predict other agent's shaped info within a short horizon, default False")
    parser.add_argument("--predict_shaped_info_horizon", default=50, type=int, help="Horizon for shaped info target, default 50")
    parser.add_argument("--predict_shaped_info_event_count", default=10, type=int, help="Event count for shaped info target, default 10")
    parser.add_argument("--use_task_v_out", default=False, action='store_true')
    parser.add_argument("--random_start_prob", default=0., type=float, help="Probability to use a random start state, default 0.")
    parser.add_argument("--use_detailed_rew_shaping", default=False, action="store_true")
    parser.add_argument("--overcooked_version", default="old", type=str, choices=["new", "old"])
    all_args = parser.parse_known_args(args)[0]

    return all_args

# 여기가 진입점이다.
def main(args):
    # sh에서 설정들 넘긴다.
    parser = get_config()  # 여기서 공통 학습 옵션을 설정함
    all_args = parse_args(args, parser) # Overcook/HSP 전용 옵션 추가

    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "rmappg":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mappg":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    else:
        raise NotImplementedError

    # HSP Stage 1: hidden reward weight random search.
    # 논문에서 말하는 hidden utility Rw를 직접 신경망으로 학습하는 것이 아니라,
    # event-based feature들의 가중치 w를 여러 방식으로 랜덤 샘플링해서
    # 서로 다른 bias/preference를 가진 policy들을 만든다.
    if all_args.use_hsp:
        def parse_value(s):
            # w0/w1은 shell script에서 comma-separated string으로 넘어온다.
            # 예:
            #   "0"             -> 그대로 "0"
            #   "1"             -> 그대로 "1"
            #   "r5"            -> [-5, 5] 구간에서 uniform random sample
            #   "r[-10:10:3]"   -> linspace(-10, 10, 3) = [-10, 0, 10] 중 하나 sample
            #
            # 즉 'r'로 시작하는 항목만 랜덤 샘플링 대상이고,
            # 나머지 숫자 문자열은 고정 reward weight로 남긴다.
            if s.startswith('r'):
                if '[' in s:
                    # "r[l:r:n]" 형태: l부터 r까지 n개 후보를 만들고 그중 하나를 고른다.
                    s = s[2:-1]
                    l, r, n = s.split(':')
                    l, r, n = float(l), float(r), int(n)
                    return np.random.choice(np.linspace(l, r, n))  # 여기서 랜덤으로 하나 고름
                else:
                    # "rv" 형태: -v부터 v까지 연속 구간에서 하나를 뽑는다.
                    v = float(s[1:])
                    return np.random.uniform(-v, v)
            return s

        # w0: agent0이 최적화할 hidden utility 쪽 reward weight.
        # HSP S1에서는 이 w0를 seed/run마다 다르게 뽑아서
        # onion 선호, tomato 선호, dish 선호 같은 biased policy 후보를 만든다.
        w0 = []
        for s in all_args.w0.split(','):
            w0.append(parse_value(s))

        # env 쪽 코드는 w0를 다시 comma-separated string으로 기대하므로
        # 샘플링된 리스트를 "v1,v2,v3,..." 형태로 되돌린다.
        all_args.w0 = ""
        for s in w0:
            all_args.w0 += str(s) + ","
        all_args.w0 = all_args.w0[:-1]

        # w1: agent1 쪽 reward weight.
        # 현재 HSP S1 shell script에서는 대부분 마지막 order reward만 1인 형태라
        # biased partner를 받쳐주는 task-reward 기준 상대 policy에 가깝다.
        # 단, 코드 자체는 w1에도 'r[...]'를 넣으면 동일하게 random search가 가능하다.
        w1 = []
        for s in all_args.w1.split(','):
            w1.append(parse_value(s))

        # w1도 env가 읽을 수 있도록 다시 string으로 변환한다.
        all_args.w1 = ""
        for s in w1:
            all_args.w1 += str(s) + ","
        all_args.w1 = all_args.w1[:-1]

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.layout_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.wandb_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.layout_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True,
                         tags=all_args.wandb_tags)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args, run_dir) # 환경 묶음
    eval_envs = make_eval_env(all_args, run_dir) if all_args.use_eval else None
    num_agents = all_args.num_agents # 2 player

    config = { # Runner에게 넘겨줄 학습에 필요한 재료 박스
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents, # 2
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from hsp.runner.shared.overcooked_runner import OvercookedRunner as Runner
    else:
        from hsp.runner.separated.overcooked_runner import OvercookedRunner as Runner

    runner = Runner(config)  # OvercookedRunner 클래스 객체 만듦 # Runner를 선택한다.  => Policy/trainer/buffer
    runner.run()             # OvercookedRunner.rum()이 실행된다.                     => OvercookedRunner.rum()으로 실제 rollout + PPO 학습 루프를 시작한다.
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])