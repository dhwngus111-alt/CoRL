"""Shared helpers for transplant train/eval entrypoints."""

from __future__ import annotations

import os
import socket
from pathlib import Path

import numpy as np
import torch

from transplant.bootstrap import POLICY_POOL_ROOT, TRANSPLANT_ROOT, ensure_paths


ensure_paths()

from transplant.hsp_hidden_utility import HIDDEN_UTILITY_KEYS  # noqa: E402


RISKY_ENV_NAME = "RiskyOvercooked"
DEFAULT_LAYOUT = "risky_dualpath_subgoal"
DEFAULT_EPISODE_LENGTH = 200
DEFAULT_SUBGOAL_DISABLE_STEPS = 60
DEFAULT_P_SLIP = 0.4
DEFAULT_TIME_COST = -0.3
DEFAULT_DISTANCE_SHAPING_REW = 0.3
DEFAULT_SUBGOAL_PRESS_REW = 2.0
HIDDEN_UTILITY_DIM = len(HIDDEN_UTILITY_KEYS)


def normalize_risky_args(all_args):
    all_args.env_name = RISKY_ENV_NAME
    all_args.overcooked_version = "risky"
    if getattr(all_args, "predict_other_shaped_info", False):
        print("Risky transplant disables HSP Overcooked shaped-info prediction.")
        all_args.predict_other_shaped_info = False
    return all_args


def add_risky_overcooked_args(parser, mode: str):
    parser.add_argument("--layout_name", type=str, default=DEFAULT_LAYOUT)
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--wandb_stage_name", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_run_id", type=str, default="")
    parser.add_argument("--wandb_group_name", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_prefix", type=str, default="")
    parser.add_argument("--initial_reward_shaping_factor", type=float, default=1.0)
    parser.add_argument("--reward_shaping_factor", type=float, default=1.0)
    parser.add_argument("--reward_shaping_horizon", type=int, default=0)
    parser.add_argument("--use_phi", default=False, action="store_true")
    parser.add_argument("--use_hsp", default=False, action="store_true")
    parser.add_argument("--random_index", default=False, action="store_true")
    parser.add_argument("--w0", type=str, default=",".join(["0"] * HIDDEN_UTILITY_DIM + ["1"]))
    parser.add_argument("--w1", type=str, default=",".join(["0"] * HIDDEN_UTILITY_DIM + ["1"]))
    parser.add_argument("--predict_other_shaped_info", default=False, action="store_true")
    parser.add_argument("--predict_shaped_info_horizon", default=50, type=int)
    parser.add_argument("--predict_shaped_info_event_count", default=10, type=int)
    parser.add_argument("--use_task_v_out", default=False, action="store_true")
    parser.add_argument("--random_start_prob", default=0.0, type=float)
    parser.add_argument("--use_detailed_rew_shaping", default=False, action="store_true")
    parser.add_argument("--overcooked_version", default="risky", choices=["old", "new", "risky"])
    parser.add_argument("--p_slip", type=float, default=DEFAULT_P_SLIP)
    parser.add_argument("--time_cost", type=float, default=DEFAULT_TIME_COST)
    parser.add_argument("--neglect_boarders", default=False, action="store_true")
    parser.add_argument("--subgoal_disable_steps", type=int, default=DEFAULT_SUBGOAL_DISABLE_STEPS)
    parser.add_argument("--placement_in_pot_rew", type=float, default=3.0)
    parser.add_argument("--dish_pickup_reward", type=float, default=1.0)
    parser.add_argument("--soup_pickup_reward", type=float, default=3.0)
    parser.add_argument("--distance_shaping_rew", type=float, default=DEFAULT_DISTANCE_SHAPING_REW)
    parser.add_argument("--subgoal_press_rew", type=float, default=DEFAULT_SUBGOAL_PRESS_REW)
    parser.add_argument("--handoff_shaping", default=False, action="store_true")
    parser.add_argument("--competitive_onion_supply", type=int, default=None)

    if mode == "hsp":
        parser.add_argument("--hsp_final_gif_episodes", type=int, default=0)

    if mode in {"adaptive", "eval", "event_eval"}:
        parser.add_argument("--use_agent_policy_id", default=False, action="store_true")
        parser.add_argument("--population_yaml_path", type=str)

    if mode == "adaptive":
        parser.add_argument("--shaped_info_coef", default=0.5, type=float)
        parser.add_argument("--policy_group_normalization", default=False, action="store_true")
        parser.add_argument("--use_advantage_prioritized_sampling", default=False, action="store_true")
        parser.add_argument("--uniform_preference", default=False, action="store_true")
        parser.add_argument("--uniform_sampling_repeat", default=0, type=int)
        parser.add_argument("--stage", type=int, default=1)
        parser.add_argument("--mep_final_gif_episodes", type=int, default=0)
        parser.add_argument("--mep_use_prioritized_sampling", default=False, action="store_true")
        parser.add_argument("--mep_prioritized_alpha", type=float, default=3.0)
        parser.add_argument("--mep_entropy_alpha", type=float, default=0.01)
        parser.add_argument("--population_size", type=int, default=12)
        parser.add_argument("--adaptive_agent_name", type=str, required=True)
        parser.add_argument("--adaptive_checkpoint_interval_steps", type=int, default=0)
        parser.add_argument("--adaptive_checkpoint_dir_name", type=str, default="milestone_checkpoints")
        parser.add_argument("--train_env_batch", type=int, default=1)
        parser.add_argument("--eval_env_batch", type=int, default=1)
        parser.add_argument("--use_policy_in_env", default=True, action="store_false")
        parser.add_argument("--eval_policy", default="", type=str)

    if mode == "eval":
        parser.add_argument("--store_traj", default=False, action="store_true")
        parser.add_argument("--agent0_policy_name", type=str, required=True)
        parser.add_argument("--agent1_policy_name", type=str, required=True)
        parser.add_argument("--adaptive_policy_name", type=str, default="hsp_adaptive")
        parser.add_argument("--metrics_output_path", type=Path, default=None)
        parser.add_argument("--event_table_output_path", type=Path, default=None)
        parser.add_argument("--score_table_output_path", type=Path, default=None)
        parser.add_argument("--gif_manifest_output_path", type=Path, default=None)
        parser.add_argument("--render_eval_gif_episodes", type=int, default=0)
        parser.add_argument("--render_gif_subdir", type=str, default="")
        parser.add_argument("--gif_panel_section", type=str, default="final_eval_gifs")
        parser.add_argument("--stream_eval_gifs_to_wandb", default=False, action="store_true")
        parser.add_argument("--stream_gif_wandb_project", type=str, default="")
        parser.add_argument("--stream_gif_wandb_entity", type=str, default="")
        parser.add_argument("--stream_gif_wandb_mode", type=str, default="online")
        parser.add_argument("--stream_gif_wandb_run_name", type=str, default="")
        parser.add_argument("--stream_gif_wandb_run_id", type=str, default="")
        parser.add_argument("--log_eval_scalars", default=False, action="store_true")

    return parser


def parse_random_weight_string(weight: str) -> str:
    def parse_value(item: str):
        if item.startswith("r"):
            if "[" in item:
                spec = item[2:-1]
                left, right, num = spec.split(":")
                return np.random.choice(np.linspace(float(left), float(right), int(num)))
            span = float(item[1:])
            return np.random.uniform(-span, span)
        return item

    return ",".join(str(parse_value(item)) for item in weight.split(","))


def sample_hsp_weights(all_args):
    if getattr(all_args, "use_hsp", False):
        all_args.w0 = parse_random_weight_string(all_args.w0)
        all_args.w1 = parse_random_weight_string(all_args.w1)
    return all_args


def setup_device(all_args):
    torch.set_num_threads(all_args.n_training_threads)
    if not all_args.cuda:
        return torch.device("cpu")

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as exc:
        raise RuntimeError(
            "CUDA was requested but CUDA initialization raised an exception; "
            f"CUDA_VISIBLE_DEVICES={visible_devices!r}. Refusing to fall back to CPU."
        ) from exc

    if not cuda_available:
        raise RuntimeError(
            "CUDA was requested but PyTorch could not initialize a GPU; "
            f"CUDA_VISIBLE_DEVICES={visible_devices!r}. Refusing to fall back to CPU. "
            "Check nvidia-smi/driver health, or pass --cuda only when CPU execution "
            "is intentional."
        )

    device = torch.device("cuda:0")
    try:
        probe = torch.empty(1, device=device)
        del probe
    except Exception as exc:
        raise RuntimeError(
            "CUDA was requested and detected, but a CUDA tensor allocation failed; "
            f"CUDA_VISIBLE_DEVICES={visible_devices!r}. Refusing to fall back to CPU."
        ) from exc

    if all_args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return device


def set_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def make_results_root(all_args) -> Path:
    run_dir = (
        TRANSPLANT_ROOT
        / "results"
        / all_args.env_name
        / all_args.layout_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_numbered_run_dir(all_args) -> Path:
    run_dir = make_results_root(all_args)
    existing = [
        int(folder.name.split("run")[1])
        for folder in run_dir.iterdir()
        if folder.is_dir() and folder.name.startswith("run") and folder.name[3:].isdigit()
    ]
    curr_run = f"run{max(existing) + 1}" if existing else "run1"
    run_dir = run_dir / curr_run
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_run_dir(all_args) -> Path:
    run_dir = make_results_root(all_args)
    if all_args.use_wandb:
        return run_dir
    return make_numbered_run_dir(all_args)


def init_wandb(all_args, run_dir: Path):
    if not all_args.use_wandb:
        try:
            import wandb

            return wandb.init(mode="disabled", dir=str(run_dir), reinit=True)
        except Exception:
            return None
    import wandb

    run_name = str(getattr(all_args, "wandb_run_name", "") or "").strip()
    if not run_name:
        base_run_name = f"{all_args.algorithm_name}_{all_args.experiment_name}_seed{all_args.seed}"
        stage_name = str(getattr(all_args, "wandb_stage_name", "") or "").strip()
        run_prefix = str(
            getattr(all_args, "wandb_run_prefix", "") or f"HSP_{all_args.layout_name}"
        ).strip()
        if stage_name:
            run_name = f"{stage_name}_{run_prefix}_{base_run_name}" if run_prefix else f"{stage_name}_{base_run_name}"
        else:
            run_name = base_run_name
        if run_prefix and not stage_name and not run_name.startswith(f"{run_prefix}_"):
            run_name = f"{run_prefix}_{run_name}"
    group_name = str(getattr(all_args, "wandb_group_name", "") or f"HSP_{all_args.layout_name}")
    default_project = f"{all_args.env_name}_{all_args.layout_name}"
    project_name = str(getattr(all_args, "wandb_project", "") or default_project)
    run_id = str(getattr(all_args, "wandb_run_id", "") or "").strip()

    init_kwargs = dict(
        config=all_args,
        project=project_name,
        entity=all_args.wandb_name,
        notes=socket.gethostname(),
        name=run_name,
        group=group_name,
        dir=str(run_dir),
        job_type=getattr(all_args, "wandb_job_type", "training"),
        reinit=True,
        tags=getattr(all_args, "wandb_tags", []),
        settings=wandb.Settings(
            start_method=os.environ.get("WANDB_START_METHOD", "thread"),
            x_disable_stats=True,
            x_disable_machine_info=True,
        ),
    )
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "allow"

    return wandb.init(**init_kwargs)


def set_process_title(all_args):
    try:
        import setproctitle

        setproctitle.setproctitle(
            f"{all_args.algorithm_name}-{all_args.env_name}-{all_args.experiment_name}@{all_args.user_name}"
        )
    except Exception:
        pass


def finish_logging(all_args, run, runner):
    if hasattr(runner, "finalize_stage2_logging"):
        try:
            runner.finalize_stage2_logging()
        except Exception as exc:
            print(f"warning: failed to finalize stage2 logging: {exc}")
    if all_args.use_wandb:
        run.finish()
    elif hasattr(runner, "writter"):
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()
        if run is not None:
            run.finish()


def ensure_policy_pool_env():
    os.environ.setdefault("POLICY_POOL", str(POLICY_POOL_ROOT))
