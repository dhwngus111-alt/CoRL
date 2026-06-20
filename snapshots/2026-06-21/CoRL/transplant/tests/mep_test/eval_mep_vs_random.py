#!/usr/bin/env python
"""Evaluate 12 final MEP policies against a uniform-random partner."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import re
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from statistics import fmean

import numpy as np
import yaml


TEST_ROOT = Path(__file__).resolve().parent
CORL_ROOT = TEST_ROOT.parents[2]
sys.path.insert(0, str(CORL_ROOT))
os.environ.setdefault("POLICY_POOL", str(TEST_ROOT / "policy_pool"))

from transplant.bootstrap import TRANSPLANT_ROOT, ensure_paths  # noqa: E402

ensure_paths()

from hsp.config import get_config  # noqa: E402
from transplant.common import (  # noqa: E402
    add_risky_overcooked_args,
    normalize_risky_args,
    setup_device,
)
from transplant.eval_risky_hsp import make_eval_env  # noqa: E402
from transplant.runners.risky_overcooked_runner import (  # noqa: E402
    RiskyOvercookedRunner,
)


LAYOUT = "risky_multipath_subgoal"
MEP_COUNT = 12
EPISODES_PER_POLICY = 5
FINAL_ACTOR_VERSION = 20_000_000
ACTION_COUNT = 6
DEFAULT_WANDB_PROJECT = "mep_random_test"
ACTOR_RE = re.compile(r"actor_periodic_(\d+)\.pt$")


class UniformRandomEvalPolicy:
    """Minimal PolicyPool-compatible uniform random policy."""

    def __init__(self, seed: int, action_count: int = ACTION_COUNT):
        self.seed = int(seed)
        self.action_count = int(action_count)
        self._control_agents: list[tuple[int, int]] = []
        self._rng_by_agent: dict[tuple[int, int], np.random.Generator] = {}

    @property
    def control_agents(self) -> list[tuple[int, int]]:
        return self._control_agents

    def reset(self, num_envs: int, num_agents: int) -> None:
        del num_envs, num_agents
        self._control_agents = []
        self._rng_by_agent = {}

    def register_control_agent(self, env_idx: int, agent_idx: int) -> None:
        key = (int(env_idx), int(agent_idx))
        if key in self._rng_by_agent:
            return
        self._control_agents.append(key)
        agent_seed = self.seed + key[0] * 10_007 + key[1] * 1_009
        self._rng_by_agent[key] = np.random.default_rng(agent_seed)

    def step(self, obs, agents, **kwargs):
        del obs, kwargs
        return np.asarray(
            [[self._rng_by_agent[tuple(agent)].integers(0, self.action_count)] for agent in agents],
            dtype=np.int64,
        )

    def prep_rollout(self) -> None:
        pass

    def to(self, device):
        del device
        return self


def mep_role_for_episode(episode_index: int) -> str:
    """Use a reproducible 3:2 MEP agent-position split over five episodes."""

    return "agent0" if int(episode_index) % 2 == 0 else "agent1"


def _policy_config_args(path: Path):
    with path.open("rb") as file:
        payload = pickle.load(file)
    return payload[0] if isinstance(payload, (tuple, list)) else payload


def _candidate_policy_config(actor: Path, search_root: Path) -> Path | None:
    for parent in actor.parents:
        candidate = parent / "policy_config.pkl"
        if candidate.exists():
            return candidate
        if parent == search_root:
            break
    return None


def _metadata_matches(actor: Path, search_root: Path) -> bool:
    config_path = _candidate_policy_config(actor, search_root)
    if config_path is None:
        return False
    args = _policy_config_args(config_path)
    checks = {
        "layout_name": LAYOUT,
        "episode_length": 200,
        "p_slip": 0.4,
        "time_cost": -0.3,
    }
    for name, expected in checks.items():
        actual = getattr(args, name, None)
        if isinstance(expected, float):
            try:
                if abs(float(actual) - expected) > 1e-8:
                    return False
            except (TypeError, ValueError):
                return False
        elif actual != expected:
            return False
    return True


def _select_final_actor(search_root: Path, policy_name: str, expected_version: int) -> Path:
    candidates = []
    for actor in search_root.rglob(f"{policy_name}/actor_periodic_{expected_version}.pt"):
        match = ACTOR_RE.fullmatch(actor.name)
        if match and int(match.group(1)) == expected_version and _metadata_matches(actor, search_root):
            candidates.append(actor)
    if not candidates:
        raise FileNotFoundError(
            f"no metadata-matching {policy_name} final actor version {expected_version} "
            f"under {search_root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def prepare_test_policy_pool(
    *,
    source_results_root: Path,
    source_policy_config: Path,
    policy_pool_root: Path,
    run_config_dir: Path,
    expected_version: int = FINAL_ACTOR_VERSION,
) -> Path:
    """Copy 12 validated final actors into the isolated test policy pool."""

    search_root = source_results_root / "RiskyOvercooked" / LAYOUT / "mep" / "mep-S1"
    if not search_root.is_dir():
        raise FileNotFoundError(search_root)

    source_args = _policy_config_args(source_policy_config)
    if getattr(source_args, "layout_name", None) != LAYOUT:
        raise ValueError(f"policy config layout is not {LAYOUT}: {source_policy_config}")
    if int(getattr(source_args, "episode_length", -1)) != 200:
        raise ValueError(f"policy config episode_length is not 200: {source_policy_config}")

    layout_root = policy_pool_root / LAYOUT
    config_dir = layout_root / "policy_config"
    actor_dir = layout_root / "mep" / "s1"
    config_dir.mkdir(parents=True, exist_ok=True)
    actor_dir.mkdir(parents=True, exist_ok=True)
    run_config_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_policy_config, config_dir / "mlp_policy_config.pkl")

    population = {}
    for policy_idx in range(1, MEP_COUNT + 1):
        policy_name = f"mep{policy_idx}"
        source_actor = _select_final_actor(search_root, policy_name, expected_version)
        destination = actor_dir / f"{policy_name}_final_actor.pt"
        shutil.copy2(source_actor, destination)
        population[policy_name] = {
            "policy_config_path": f"{LAYOUT}/policy_config/mlp_policy_config.pkl",
            "featurize_type": "ppo",
            "train": False,
            "model_path": {"actor": f"{LAYOUT}/mep/s1/{destination.name}"},
        }

    population_path = actor_dir / "final_eval.yml"
    with population_path.open("w") as file:
        yaml.safe_dump(population, file, sort_keys=False)
    shutil.copy2(population_path, run_config_dir / "mep_final_eval.yml")
    return population_path


def select_eval_device(all_args) -> tuple[object, str, str | None]:
    """Prefer CUDA, but permit CPU fallback only for this evaluation."""

    try:
        device = setup_device(all_args)
        return device, device.type, None
    except RuntimeError as exc:
        if not all_args.allow_cpu_fallback:
            raise
        warning = str(exc)
        warnings.warn(f"CUDA evaluation unavailable; falling back to CPU: {warning}")
        all_args.cuda = False
        device = setup_device(all_args)
        return device, "cpu", warning


def _role_map(policy_name: str, episode_count: int) -> tuple[dict, list[str]]:
    roles = [mep_role_for_episode(index) for index in range(episode_count)]
    mapping = {}
    for env_idx, role in enumerate(roles):
        if role == "agent0":
            mapping[(env_idx, 0)] = policy_name
            mapping[(env_idx, 1)] = "random"
        else:
            mapping[(env_idx, 0)] = "random"
            mapping[(env_idx, 1)] = policy_name
    return mapping, roles


def _gif_paths_by_env(eval_envs, expected_count: int) -> list[Path]:
    raw_paths = eval_envs.get_last_render_gif_paths()
    if len(raw_paths) != expected_count:
        raise RuntimeError(f"expected GIF results from {expected_count} envs, got {len(raw_paths)}")
    selected = []
    for env_idx, env_paths in enumerate(raw_paths):
        paths = env_paths if isinstance(env_paths, (list, tuple)) else [env_paths]
        paths = [Path(path) for path in paths if path and Path(path).is_file()]
        if len(paths) != 1:
            raise RuntimeError(f"expected exactly one GIF for env {env_idx}, got {len(paths)}")
        selected.append(paths[0])
    return selected


def build_episode_rows(
    *, policy_name: str, roles: list[str], eval_info: dict, gif_paths: list[Path], device_used: str
) -> list[dict]:
    sparse = list(eval_info["eval_ep_sparse_r"])
    sparse0 = list(eval_info["eval_ep_sparse_r_by_agent0"])
    sparse1 = list(eval_info["eval_ep_sparse_r_by_agent1"])
    delivery0 = list(eval_info["eval_ep_delivery_by_agent0"])
    delivery1 = list(eval_info["eval_ep_delivery_by_agent1"])
    lengths = {len(sparse), len(sparse0), len(sparse1), len(delivery0), len(delivery1), len(roles), len(gif_paths)}
    if lengths != {EPISODES_PER_POLICY}:
        raise RuntimeError(f"expected exactly {EPISODES_PER_POLICY} episode results, lengths={sorted(lengths)}")

    rows = []
    for index, role in enumerate(roles):
        mep_delivery = delivery0[index] if role == "agent0" else delivery1[index]
        random_delivery = delivery1[index] if role == "agent0" else delivery0[index]
        rows.append(
            {
                "policy": policy_name,
                "episode": index + 1,
                "mep_role": role,
                "team_sparse_reward": float(sparse[index]),
                "agent0_sparse_reward": float(sparse0[index]),
                "agent1_sparse_reward": float(sparse1[index]),
                "team_deliveries": float(delivery0[index] + delivery1[index]),
                "mep_delivery_credit": float(mep_delivery),
                "random_delivery_credit": float(random_delivery),
                "device": device_used,
                "gif_path": str(gif_paths[index]),
            }
        )
    return rows


def summarize_policy(rows: list[dict]) -> dict:
    policy_names = {row["policy"] for row in rows}
    if len(policy_names) != 1 or len(rows) != EPISODES_PER_POLICY:
        raise ValueError("summary requires five rows from exactly one policy")
    return {
        "policy": rows[0]["policy"],
        "mean_sparse_reward": fmean(row["team_sparse_reward"] for row in rows),
        "mean_deliveries": fmean(row["team_deliveries"] for row in rows),
        "mean_mep_delivery_credit": fmean(row["mep_delivery_credit"] for row in rows),
        "mean_random_delivery_credit": fmean(row["random_delivery_credit"] for row in rows),
        "device": rows[0]["device"],
    }


def _wandb_payload(policy_name: str, rows: list[dict], summary: dict, wandb) -> dict:
    table_columns = list(rows[0])
    payload = {
        f"{policy_name}/summary/mean_sparse_reward": summary["mean_sparse_reward"],
        f"{policy_name}/summary/mean_deliveries": summary["mean_deliveries"],
        f"{policy_name}/summary/mep_delivery_credit": summary["mean_mep_delivery_credit"],
        f"{policy_name}/summary/random_delivery_credit": summary["mean_random_delivery_credit"],
        f"{policy_name}/episodes/table": wandb.Table(
            columns=table_columns,
            data=[[row[column] for column in table_columns] for row in rows],
        ),
    }
    for row in rows:
        payload[f"{policy_name}/gifs/episode_{int(row['episode']):02d}"] = wandb.Video(
            row["gif_path"], format="gif"
        )
    return payload


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _copy_episode_gifs(source_paths: list[Path], output_root: Path, policy_name: str, roles: list[str]) -> list[Path]:
    policy_dir = output_root / "gifs" / policy_name
    policy_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, (source, role) in enumerate(zip(source_paths, roles), start=1):
        destination = policy_dir / f"episode_{index:02d}_{role}.gif"
        shutil.copy2(source, destination)
        outputs.append(destination)
    return outputs


def parse_args(argv=None):
    parser = get_config()
    add_risky_overcooked_args(parser, mode="hsp")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--policy-pool-root", type=Path, default=TEST_ROOT / "policy_pool")
    parser.add_argument("--source-results-root", type=Path, default=TRANSPLANT_ROOT / "results")
    parser.add_argument(
        "--source-policy-config",
        type=Path,
        default=TRANSPLANT_ROOT / "policy_pool" / LAYOUT / "policy_config" / "mlp_policy_config.pkl",
    )
    parser.add_argument("--expected-final-version", type=int, default=FINAL_ACTOR_VERSION)
    parser.add_argument("--base-eval-seed", type=int, default=20260621)
    parser.add_argument("--allow-cpu-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    args = normalize_risky_args(parser.parse_known_args(argv)[0])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_root = args.output_root or TEST_ROOT / "outputs" / f"run_{timestamp}"
    args.algorithm_name = "population"
    args.experiment_name = "mep-vs-random"
    args.layout_name = LAYOUT
    args.num_agents = 2
    args.episode_length = 200
    args.p_slip = 0.4
    args.subgoal_disable_steps = 60
    args.time_cost = 0.0
    args.n_rollout_threads = EPISODES_PER_POLICY
    args.n_eval_rollout_threads = EPISODES_PER_POLICY
    args.eval_episodes = EPISODES_PER_POLICY
    args.eval_stochastic = False
    args.render_eval_gif_episodes = EPISODES_PER_POLICY
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    for subdir in ("config", "csv", "gifs", "wandb"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    population_yaml = prepare_test_policy_pool(
        source_results_root=args.source_results_root,
        source_policy_config=args.source_policy_config,
        policy_pool_root=args.policy_pool_root,
        run_config_dir=output_root / "config",
        expected_version=args.expected_final_version,
    )
    device, device_used, cuda_error = select_eval_device(args)

    import wandb

    run_name = args.wandb_run_name or f"mep_vs_random_{LAYOUT}_{output_root.name}"
    run = wandb.init(
        project=args.wandb_project or DEFAULT_WANDB_PROJECT,
        entity=args.wandb_name,
        name=run_name,
        group=args.wandb_group_name or f"MEP_eval_{LAYOUT}",
        job_type="evaluation",
        mode=args.wandb_mode,
        dir=str(output_root / "wandb"),
        config={
            "layout": LAYOUT,
            "mep_policy_count": MEP_COUNT,
            "episodes_per_policy": EPISODES_PER_POLICY,
            "episode_length": 200,
            "p_slip": 0.4,
            "subgoal_disable_steps": 60,
            "time_cost": 0.0,
            "device_used": device_used,
            "cuda_fallback_reason": cuda_error,
            "population_yaml": str(population_yaml),
        },
        settings=wandb.Settings(
            start_method=os.environ.get("WANDB_START_METHOD", "thread"),
            x_disable_stats=True,
            x_disable_machine_info=True,
        ),
    )

    envs = None
    eval_envs = None
    runner = None
    episode_rows: list[dict] = []
    summary_rows: list[dict] = []
    gif_manifest: list[dict] = []
    try:
        envs = make_eval_env(args, output_root, render_eval_gifs=False)
        runner = RiskyOvercookedRunner(
            {
                "all_args": args,
                "envs": envs,
                "eval_envs": None,
                "num_agents": 2,
                "device": device,
                "run_dir": output_root,
            }
        )
        featurize_type = runner.policy.load_population(str(population_yaml), evaluation=True)
        runner.policy.register_policy(
            "random",
            UniformRandomEvalPolicy(args.base_eval_seed),
            None,
            False,
            ["random", {"type": "uniform_random", "id": 0.0}],
        )

        for policy_idx in range(1, MEP_COUNT + 1):
            policy_name = f"mep{policy_idx}"
            mapping, roles = _role_map(policy_name, EPISODES_PER_POLICY)
            runner.policy.set_map_ea2p(mapping, load_unused_to_cpu=True)
            args.render_gif_subdir = policy_name
            render_root = output_root / ".render_tmp" / policy_name
            eval_envs = make_eval_env(args, render_root, render_eval_gifs=True)
            runner.eval_envs = eval_envs
            eval_envs.reset_featurize_type(
                [
                    (
                        featurize_type.get(mapping[(env_idx, 0)], "ppo"),
                        featurize_type.get(mapping[(env_idx, 1)], "ppo"),
                    )
                    for env_idx in range(EPISODES_PER_POLICY)
                ]
            )

            eval_info = runner.evaluate_one_episode_with_multi_policy(
                runner.policy.policy_pool, mapping
            )
            raw_gif_paths = _gif_paths_by_env(eval_envs, EPISODES_PER_POLICY)
            copied_gifs = _copy_episode_gifs(raw_gif_paths, output_root, policy_name, roles)
            eval_envs.close()
            eval_envs = None
            shutil.rmtree(render_root, ignore_errors=True)

            rows = build_episode_rows(
                policy_name=policy_name,
                roles=roles,
                eval_info=eval_info,
                gif_paths=copied_gifs,
                device_used=device_used,
            )
            summary = summarize_policy(rows)
            episode_rows.extend(rows)
            summary_rows.append(summary)
            gif_manifest.extend(
                {
                    "policy": row["policy"],
                    "episode": row["episode"],
                    "mep_role": row["mep_role"],
                    "gif_path": row["gif_path"],
                }
                for row in rows
            )
            _write_csv(output_root / "csv" / "episodes.csv", episode_rows)
            _write_csv(output_root / "csv" / "policy_summary.csv", summary_rows)
            _write_csv(output_root / "csv" / "gif_manifest.csv", gif_manifest)
            run.log(_wandb_payload(policy_name, rows, summary, wandb))
            print(
                f"{policy_name}: mean_sparse={summary['mean_sparse_reward']:.3f}, "
                f"mean_deliveries={summary['mean_deliveries']:.3f}, device={device_used}"
            )
    finally:
        if eval_envs is not None:
            eval_envs.close()
        if envs is not None:
            envs.close()
        run.finish()
        shutil.rmtree(output_root / ".render_tmp", ignore_errors=True)

    if len(episode_rows) != MEP_COUNT * EPISODES_PER_POLICY:
        raise RuntimeError(f"expected 60 episode rows, got {len(episode_rows)}")
    if len(summary_rows) != MEP_COUNT:
        raise RuntimeError(f"expected 12 policy summaries, got {len(summary_rows)}")
    print(f"completed MEP-vs-random evaluation: {output_root}")


if __name__ == "__main__":
    main(sys.argv[1:])
