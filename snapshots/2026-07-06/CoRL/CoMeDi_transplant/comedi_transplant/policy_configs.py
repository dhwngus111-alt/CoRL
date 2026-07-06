"""Policy config and YAML helpers for CoMeDi_transplant."""

from __future__ import annotations

import pickle
from argparse import Namespace
from pathlib import Path

import yaml

from comedi_transplant.bootstrap import COMEDI_TRANSPLANT_ROOT, POLICY_POOL_ROOT, ensure_paths


ensure_paths()

from hsp.config import get_config  # noqa: E402
from transplant.adapters.risky_overcooked_env import RiskyOvercooked  # noqa: E402
from transplant.common import (  # noqa: E402
    RISKY_ENV_NAME,
    add_risky_overcooked_args,
    normalize_risky_args,
)


DEFAULT_LAYOUTS = (
    "risky_dualpath_subgoal",
    "risky_mixed_coordination_subgoal",
    "risky_multipath_subgoal",
)
DEFAULT_CNN_LAYERS = "32,3,1,1 64,3,1,1 32,3,1,1"


def parse_policy_args(layout: str, algorithm_name: str, episode_length: int) -> Namespace:
    parser = get_config()
    add_risky_overcooked_args(parser, mode="hsp")
    parser.set_defaults(
        env_name=RISKY_ENV_NAME,
        algorithm_name=algorithm_name,
        experiment_name="comedi-policy-config",
        layout_name=layout,
        num_agents=2,
        episode_length=episode_length,
        hidden_size=64,
        layer_N=2,
        activation_id=1,
        lr=1e-2,
        critic_lr=1e-2,
        ppo_epoch=10,
        entropy_coef=0.0,
        use_linear_lr_decay=True,
        cnn_layers_params=DEFAULT_CNN_LAYERS,
        use_wandb=False,
    )
    if algorithm_name == "mappo":
        parser.set_defaults(use_recurrent_policy=False, use_naive_recurrent_policy=False)
    return normalize_risky_args(parser.parse_known_args([])[0])


def build_policy_configs(
    layout: str,
    episode_length: int = 200,
    policy_pool_root: Path = POLICY_POOL_ROOT,
) -> None:
    """Write MLP/RNN HSP policy config pickles for a Risky layout."""

    layout_root = policy_pool_root / layout
    for subdir in (
        "policy_config",
        "comedi/s1",
        "comedi/s2",
    ):
        (layout_root / subdir).mkdir(parents=True, exist_ok=True)

    run_dir = COMEDI_TRANSPLANT_ROOT / "results" / "_policy_config"
    run_dir.mkdir(parents=True, exist_ok=True)

    for config_name, algorithm in (("mlp", "mappo"), ("rnn", "rmappo")):
        all_args = parse_policy_args(layout, algorithm, episode_length)
        env = RiskyOvercooked(all_args, run_dir)
        try:
            policy_config = (
                all_args,
                env.observation_space[0],
                env.share_observation_space[0],
                env.action_space[0],
            )
            config_path = layout_root / "policy_config" / f"{config_name}_policy_config.pkl"
            with config_path.open("wb") as file:
                pickle.dump(policy_config, file)
        finally:
            env.close()


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def policy_entry(
    layout: str,
    config_name: str,
    train: bool,
    actor_path: str | None = None,
    critic_path: str | None = None,
) -> dict:
    entry = {
        "policy_config_path": f"{layout}/policy_config/{config_name}_policy_config.pkl",
        "featurize_type": "ppo",
        "train": train,
    }
    model_path = {}
    if actor_path is not None:
        model_path["actor"] = actor_path
    if critic_path is not None:
        model_path["critic"] = critic_path
    if model_path:
        entry["model_path"] = model_path
    return entry


def write_population_yamls(layout: str, population_size: int) -> None:
    """Write CoMeDi train/eval YAMLs after convention checkpoints exist."""

    s1_root = POLICY_POOL_ROOT / layout / "comedi" / "s1"
    s2_root = POLICY_POOL_ROOT / layout / "comedi" / "s2"

    s1_train = {}
    s1_eval = {}
    s2_train = {
        "comedi_adaptive": policy_entry(layout, "rnn", train=True),
    }
    s2_eval = {}
    for idx in range(1, population_size + 1):
        name = f"comedi{idx}"
        actor_rel = f"{layout}/comedi/s1/{name}_actor.pt"
        s1_train[name] = policy_entry(layout, "mlp", train=True)
        fixed_entry = policy_entry(layout, "mlp", train=False, actor_path=actor_rel)
        s1_eval[name] = fixed_entry
        s2_train[name] = dict(fixed_entry)
        s2_eval[name] = dict(fixed_entry)

    write_yaml(s1_root / "train.yml", s1_train)
    write_yaml(s1_root / "eval.yml", s1_eval)
    write_yaml(s2_root / "train.yml", s2_train)
    write_yaml(s2_root / "eval.yml", s2_eval)


def update_adaptive_eval_yaml(layout: str, actor_rel_path: str) -> None:
    """Add the trained adaptive checkpoint to the CoMeDi S2 eval YAML."""

    s2_root = POLICY_POOL_ROOT / layout / "comedi" / "s2"
    eval_path = s2_root / "eval.yml"
    data = yaml.safe_load(eval_path.open()) if eval_path.exists() else {}
    data["comedi_adaptive"] = policy_entry(
        layout,
        "rnn",
        train=False,
        actor_path=actor_rel_path,
    )
    write_yaml(eval_path, data)


def update_adaptive_train_yaml(
    layout: str,
    actor_rel_path: str | None = None,
    critic_rel_path: str | None = None,
) -> None:
    """Register an optional warm-start checkpoint for CoMeDi S2 training."""

    s2_root = POLICY_POOL_ROOT / layout / "comedi" / "s2"
    train_path = s2_root / "train.yml"
    data = yaml.safe_load(train_path.open()) if train_path.exists() else {}
    data["comedi_adaptive"] = policy_entry(
        layout,
        "rnn",
        train=True,
        actor_path=actor_rel_path,
        critic_path=critic_rel_path,
    )
    write_yaml(train_path, data)
