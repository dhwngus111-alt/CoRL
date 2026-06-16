#!/usr/bin/env python3
"""Risky Overcooked 환경 실행 예제.

논문 단계:
    MARSRL/DDQN 학습이나 평가 본 단계가 아니라, 학습 전에 layout, MDP, Env,
    env.step() 흐름이 정상 동작하는지 확인하는 환경 스모크 테스트 단계이다.

호출 위치:
    사용자가 CLI에서 직접 실행한다.

전체 역할:
    지정한 layout을 로드하고 OvercookedGridworld(MDP)와 OvercookedEnv를 만든 뒤,
    random agent 2명으로 한 episode rollout을 수행하고 reward/stat 정보를 출력한다.

Example:
    python examples/random_agent_example.py --layout risky_mixed_coordination --horizon 50
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from risky_overcooked_py.agents.agent import AgentPair, RandomAgent  # noqa: E402
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv  # noqa: E402
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld  # noqa: E402


def run_random_rollout(layout, horizon, seed, p_slip=None, all_actions=True):
    """입력 옵션으로 한 episode를 random policy로 실행하고 요약 결과를 반환한다."""

    # seed: Python random과 NumPy random을 같은 값으로 고정해 재현 가능한 rollout을 만든다.
    random.seed(seed)
    np.random.seed(seed)

    # mdp_kwargs: layout 기본 설정 중 덮어쓸 MDP 옵션 묶음.
    # p_slip: puddle에 물건을 들고 들어갔을 때 미끄러질 확률 override 값이다.
    mdp_kwargs = {}
    if p_slip is not None:
        mdp_kwargs["p_slip"] = p_slip

    # layout: src/risky_overcooked_py/data/layouts/{layout}.layout 파일 이름.
    # mdp: layout 기반의 상태전이/보상 규칙을 담는 OvercookedGridworld 모델.
    mdp = OvercookedGridworld.from_layout_name(layout, **mdp_kwargs)

    # horizon: episode가 강제로 종료되기 전까지 허용되는 최대 environment step 수.
    # env: 현재 state를 들고 있으며 env.step(joint_action)으로 실제 rollout을 진행하는 wrapper.
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)

    # agents: agent0, agent1 두 명의 RandomAgent를 하나의 pair로 묶은 객체.
    # all_actions=True면 이동/정지/상호작용 전체 action에서 샘플링하고,
    # False면 이동/정지 action만 샘플링한다.
    agents = AgentPair(
        RandomAgent(all_actions=all_actions),
        RandomAgent(all_actions=all_actions),
    )
    agents.set_mdp(env.mdp)

    env.reset(regen_mdp=False)
    agents.reset()

    done = False
    total_reward = 0.0
    steps = 0
    info = {}

    while not done:
        # joint_action_and_infos: 각 agent가 고른 action과 부가 정보의 쌍 2개.
        joint_action_and_infos = agents.joint_action(env.state)

        # joint_action: (agent0_action, agent1_action) 형태의 동시 행동.
        # joint_action_info: 각 agent action에 대한 metadata. RandomAgent에서는 거의 비어 있다.
        joint_action, joint_action_info = zip(*joint_action_and_infos)

        # env.step: joint_action을 현재 state에 적용하고 다음 state/reward/done/info를 반환한다.
        _next_state, reward, done, info = env.step(joint_action, joint_action_info)
        total_reward += reward
        steps += 1

    # info["episode"]: done=True가 된 마지막 step에서 누적 sparse/shaped reward 등 episode 통계를 담는다.
    episode = info.get("episode", {})
    return {
        "layout": layout,
        "horizon": horizon,
        "seed": seed,
        "p_slip": mdp.p_slip,
        "steps": steps,
        "total_reward": total_reward,
        "episode_sparse_reward": episode.get("ep_sparse_r"),
        "episode_shaped_reward": episode.get("ep_shaped_r"),
        "subgoals": sorted(mdp.terrain_pos_dict.get("G", []), key=lambda p: (p[1], p[0])),
        "waters": sorted(mdp.water_disable_timers.keys(), key=lambda p: (p[1], p[0])),
    }


def main():
    """CLI argument를 읽고 random rollout을 실행한 뒤 결과를 터미널에 출력한다."""

    parser = argparse.ArgumentParser(
        description="Run a random-agent rollout in a Risky Overcooked layout."
    )
    parser.add_argument("--layout", default="risky_mixed_coordination")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p-slip", type=float, default=None, help="Override layout p_slip")
    parser.add_argument(
        "--motion-only",
        action="store_true",
        help="Sample only movement/stay actions instead of all actions.",
    )
    args = parser.parse_args()

    result = run_random_rollout(
        layout=args.layout,
        horizon=args.horizon,
        seed=args.seed,
        p_slip=args.p_slip,
        all_actions=not args.motion_only,
    )

    print("=== Random Agent Rollout ===")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
