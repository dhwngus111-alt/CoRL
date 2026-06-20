#!/usr/bin/env python
"""Create Risked Overcooked policy configs and initial population YAML files."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transplant.bootstrap import POLICY_POOL_ROOT, TRANSPLANT_ROOT, ensure_paths

ensure_paths()

from hsp.config import get_config  # noqa: E402

from transplant.adapters.risky_overcooked_env import RiskyOvercooked  # noqa: E402
from transplant.common import RISKY_ENV_NAME, add_risky_overcooked_args, normalize_risky_args  # noqa: E402


def _parse_policy_args(layout: str, algorithm_name: str, episode_length: int):
    parser = get_config()
    add_risky_overcooked_args(parser, mode="hsp")
    args = [
        "--env_name",
        RISKY_ENV_NAME,
        "--algorithm_name",
        algorithm_name,
        "--experiment_name",
        "policy-config",
        "--layout_name",
        layout,
        "--num_agents",
        "2",
        "--episode_length",
        str(episode_length),
        "--overcooked_version",
        "risky",
        "--cnn_layers_params",
        "32,3,1,1 64,3,1,1 32,3,1,1",
        "--use_wandb",
    ]
    if algorithm_name == "mappo":
        args.append("--use_recurrent_policy")
    return normalize_risky_args(parser.parse_known_args(args)[0])


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def _policy_entry(layout: str, config_name: str, train: bool, actor_path: str | None = None):
    entry = {
        "policy_config_path": f"{layout}/policy_config/{config_name}_policy_config.pkl",
        "featurize_type": "ppo",
        "train": train,
    }
    if actor_path is not None:
        entry["model_path"] = {"actor": actor_path}
    return entry


def build(layout: str, episode_length: int, policy_pool_root: Path, mep_population_size: int) -> None:
    layout_root = policy_pool_root / layout
    for subdir in [
        "policy_config",
        "mep/s1",
        "mep/s2",
        "hsp/s1",
        "hsp/s2",
    ]:
        (layout_root / subdir).mkdir(parents=True, exist_ok=True)

    run_dir = TRANSPLANT_ROOT / "results" / "_policy_config"
    run_dir.mkdir(parents=True, exist_ok=True)

    for config_name, algorithm in [("mlp", "mappo"), ("rnn", "rmappo")]:
        all_args = _parse_policy_args(layout, algorithm, episode_length)
        env = RiskyOvercooked(all_args, run_dir)
        policy_config = (
            all_args,
            env.observation_space[0],
            env.share_observation_space[0],
            env.action_space[0],
        )
        config_path = layout_root / "policy_config" / f"{config_name}_policy_config.pkl"
        with config_path.open("wb") as file:
            pickle.dump(policy_config, file)
        env.close()
        print(f"wrote {config_path}")

    mep_s1_train = {
        "mep_adaptive": _policy_entry(layout, "rnn", train=False),
    }
    for idx in range(1, mep_population_size + 1):
        mep_s1_train[f"mep{idx}"] = _policy_entry(layout, "mlp", train=True)
    _write_yaml(layout_root / "mep/s1/train.yml", mep_s1_train)
    print(f"wrote {layout_root / 'mep/s1/train.yml'}")


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="risky_multipath_subgoal")
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--policy-pool", type=Path, default=POLICY_POOL_ROOT)
    parser.add_argument("--mep-population-size", type=int, default=12)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    build(args.layout, args.episode_length, args.policy_pool, args.mep_population_size)


if __name__ == "__main__":
    main()
