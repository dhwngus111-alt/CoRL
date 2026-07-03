#!/usr/bin/env python
"""Train a Risky-native adaptive agent against a CoMeDi convention pool."""

from __future__ import annotations

import argparse
import copy
import re
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import yaml

from comedi_transplant.bootstrap import POLICY_POOL_ROOT, ensure_paths
from comedi_transplant.policy_configs import (
    DEFAULT_CNN_LAYERS,
    build_policy_configs,
    update_adaptive_eval_yaml,
    update_adaptive_train_yaml,
    write_population_yamls,
)


ensure_paths()

from hsp.config import get_config  # noqa: E402
from transplant.common import (  # noqa: E402
    add_risky_overcooked_args,
    finish_logging,
    init_wandb,
    make_run_dir,
    normalize_risky_args,
    set_process_title,
    set_seeds,
    setup_device,
)
from transplant.train_risky_adaptive import (  # noqa: E402
    make_eval_env,
    make_final_render_env,
    make_train_env,
)
from transplant.train_risky_hsp import train_hsp_s1  # noqa: E402


def _add_comedi_adaptive_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--comedi_population_size", type=int, default=8)
    parser.add_argument("--comedi_adaptive_sp_warmup_steps", type=int, default=200_000)
    parser.add_argument("--comedi_adaptive_sp_ppo_epoch", type=int, default=100)
    parser.add_argument("--comedi_adaptive_sp_lr", type=float, default=1e-2)
    parser.add_argument("--comedi_adaptive_sp_entropy_coef", type=float, default=1e-3)
    parser.add_argument("--comedi_adaptive_agent_name", type=str, default="comedi_adaptive")
    parser.add_argument("--comedi_skip_warmup", action="store_true")
    parser.add_argument("--comedi_skip_policy_config", action="store_true")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parse_argv = list(argv)
    if "--adaptive_agent_name" not in parse_argv:
        parse_argv.extend(["--adaptive_agent_name", "comedi_adaptive"])
    parser = get_config()
    add_risky_overcooked_args(parser, mode="adaptive")
    _add_comedi_adaptive_args(parser)
    parser.set_defaults(
        algorithm_name="adaptive",
        experiment_name="comedi-S2",
        adaptive_agent_name="comedi_adaptive",
        n_rollout_threads=50,
        episode_length=200,
        ppo_epoch=15,
        num_mini_batch=1,
        hidden_size=64,
        layer_N=2,
        activation_id=1,
        cnn_layers_params=DEFAULT_CNN_LAYERS,
        stage=2,
        use_agent_policy_id=True,
    )
    all_args = normalize_risky_args(parser.parse_known_args(parse_argv)[0])
    if "--comedi_adaptive_agent_name" not in parse_argv:
        all_args.comedi_adaptive_agent_name = all_args.adaptive_agent_name
    if all_args.algorithm_name != "adaptive":
        raise ValueError("CoMeDi adaptive training requires --algorithm_name adaptive")
    return all_args


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"_periodic_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _latest_checkpoint(directory: Path, pattern: str) -> Path | None:
    candidates = [path for path in directory.glob(pattern) if path.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: (_checkpoint_step(path), path.stat().st_mtime_ns))
    return candidates[-1]


def _copy_checkpoint(src: Path | None, dst: Path) -> Path | None:
    if src is None or not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _ensure_population_yamls(layout: str, population_size: int, episode_length: int) -> None:
    if not (POLICY_POOL_ROOT / layout / "policy_config" / "rnn_policy_config.pkl").exists():
        build_policy_configs(layout, episode_length, POLICY_POOL_ROOT)
    write_population_yamls(layout, population_size)


def _run_sp_warmup(all_args) -> tuple[Path | None, Path | None]:
    if all_args.comedi_skip_warmup:
        return None, None

    warmup_args = copy.deepcopy(all_args)
    warmup_args.algorithm_name = "rmappo"
    warmup_args.experiment_name = "comedi-adaptive-SP"
    warmup_args.wandb_stage_name = "comedi_adaptive_sp"
    if not getattr(warmup_args, "wandb_run_name", ""):
        warmup_args.wandb_run_name = (
            f"comedi_adaptive_sp_{warmup_args.layout_name}_seed{warmup_args.seed}"
        )
    warmup_args.num_env_steps = int(all_args.comedi_adaptive_sp_warmup_steps)
    warmup_args.ppo_epoch = int(all_args.comedi_adaptive_sp_ppo_epoch)
    warmup_args.lr = float(all_args.comedi_adaptive_sp_lr)
    warmup_args.critic_lr = float(all_args.comedi_adaptive_sp_lr)
    warmup_args.entropy_coef = float(all_args.comedi_adaptive_sp_entropy_coef)
    warmup_args.use_recurrent_policy = True
    warmup_args.use_naive_recurrent_policy = False

    print(
        "=== CoMeDi adaptive SP warmup: "
        f"{warmup_args.layout_name}, steps={warmup_args.num_env_steps} ==="
    )
    runner = train_hsp_s1(warmup_args)
    save_dir = Path(runner.save_dir)
    actor_src = _latest_checkpoint(save_dir, "actor_periodic_*.pt")
    critic_src = _latest_checkpoint(save_dir, "critic_periodic_*.pt")
    if actor_src is None:
        raise FileNotFoundError(f"warmup actor checkpoint not found in {save_dir}")

    s2_root = POLICY_POOL_ROOT / all_args.layout_name / "comedi" / "s2"
    actor_dst = _copy_checkpoint(actor_src, s2_root / "comedi_adaptive_warmup_actor.pt")
    critic_dst = _copy_checkpoint(critic_src, s2_root / "comedi_adaptive_warmup_critic.pt")
    actor_rel = f"{all_args.layout_name}/comedi/s2/{actor_dst.name}" if actor_dst else None
    critic_rel = f"{all_args.layout_name}/comedi/s2/{critic_dst.name}" if critic_dst else None
    update_adaptive_train_yaml(all_args.layout_name, actor_rel, critic_rel)
    return actor_dst, critic_dst


def _run_stage2(all_args):
    all_args.algorithm_name = "adaptive"
    all_args.stage = 2
    all_args.adaptive_agent_name = all_args.comedi_adaptive_agent_name
    all_args.population_size = int(all_args.comedi_population_size)
    all_args.population_yaml_path = str(
        POLICY_POOL_ROOT / all_args.layout_name / "comedi" / "s2" / "train.yml"
    )
    all_args.use_agent_policy_id = True
    all_args.wandb_job_type = "training"
    if not getattr(all_args, "wandb_stage_name", ""):
        all_args.wandb_stage_name = "comedi_adaptive_s2"
    if not getattr(all_args, "wandb_run_name", ""):
        all_args.wandb_run_name = (
            f"comedi_adaptive_s2_{all_args.layout_name}_seed{all_args.seed}"
        )

    device = setup_device(all_args)
    run_dir = make_run_dir(all_args)
    set_process_title(all_args)
    set_seeds(all_args.seed)

    envs = make_train_env(all_args, run_dir)
    eval_envs = make_eval_env(all_args, run_dir) if all_args.use_eval else None
    run = init_wandb(all_args, run_dir)
    render_run_dir = run_dir
    if all_args.use_wandb and run is not None and getattr(run, "dir", None):
        render_run_dir = Path(run.dir)
    final_render_envs = make_final_render_env(all_args, render_run_dir)
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "final_render_envs": final_render_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    from transplant.runners.risky_overcooked_runner import RiskyOvercookedRunner as Runner

    runner = Runner(config)
    try:
        population_config = yaml.load(open(all_args.population_yaml_path), yaml.Loader)
        override_policy_config = {}
        agent_name = all_args.adaptive_agent_name
        override_policy_config[agent_name] = (
            Namespace(
                env_name=all_args.env_name,
                use_agent_policy_id=all_args.use_agent_policy_id,
                predict_other_shaped_info=False,
                predict_shaped_info_horizon=all_args.predict_shaped_info_horizon,
                predict_shaped_info_event_count=all_args.predict_shaped_info_event_count,
                shaped_info_coef=all_args.shaped_info_coef,
                policy_group_normalization=all_args.policy_group_normalization,
                num_v_out=all_args.num_v_out,
                use_task_v_out=all_args.use_task_v_out,
                use_policy_vhead=all_args.use_policy_vhead,
            ),
            *runner.policy_config[1:],
        )
        for policy_name in population_config:
            if policy_name != agent_name:
                override_policy_config[policy_name] = (
                    None,
                    None,
                    runner.policy_config[2],
                    None,
                )

        runner.policy.load_population(
            all_args.population_yaml_path,
            evaluation=False,
            override_policy_config=override_policy_config,
        )
        runner.trainer.init_population()
        runner.train_mep()
        return runner
    finally:
        envs.close()
        if all_args.use_eval and eval_envs is not envs:
            eval_envs.close()
        if final_render_envs is not None:
            final_render_envs.close()
        finish_logging(all_args, run, runner)


def _register_final_adaptive(layout: str, runner, adaptive_name: str) -> Path:
    adaptive_dir = Path(runner.save_dir) / adaptive_name
    actor_src = adaptive_dir / "actor_best_r.pt"
    if not actor_src.exists():
        actor_src = _latest_checkpoint(adaptive_dir, "actor_periodic_*.pt")
    if actor_src is None or not actor_src.exists():
        raise FileNotFoundError(f"adaptive actor checkpoint not found in {adaptive_dir}")

    s2_root = POLICY_POOL_ROOT / layout / "comedi" / "s2"
    actor_dst = _copy_checkpoint(actor_src, s2_root / "comedi_adaptive_actor.pt")
    actor_rel = f"{layout}/comedi/s2/{actor_dst.name}"
    update_adaptive_eval_yaml(layout, actor_rel)

    critic_src = adaptive_dir / "critic_best_r.pt"
    if not critic_src.exists():
        critic_src = _latest_checkpoint(adaptive_dir, "critic_periodic_*.pt")
    _copy_checkpoint(critic_src, s2_root / "comedi_adaptive_critic.pt")
    return actor_dst


def train_adaptive(all_args) -> None:
    if not all_args.comedi_skip_policy_config:
        build_policy_configs(all_args.layout_name, all_args.episode_length, POLICY_POOL_ROOT)
    _ensure_population_yamls(
        all_args.layout_name,
        int(all_args.comedi_population_size),
        int(all_args.episode_length),
    )
    _run_sp_warmup(all_args)
    runner = _run_stage2(all_args)
    actor_path = _register_final_adaptive(
        all_args.layout_name,
        runner,
        all_args.comedi_adaptive_agent_name,
    )
    print(f"registered CoMeDi adaptive eval actor: {actor_path}")


def main(argv: list[str] | None = None) -> None:
    all_args = parse_args(sys.argv[1:] if argv is None else argv)
    train_adaptive(all_args)


if __name__ == "__main__":
    main()
