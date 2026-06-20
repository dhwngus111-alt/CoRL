#!/usr/bin/env python
"""Evaluate two policies from a transplant population YAML on Risky Overcooked."""

from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transplant.bootstrap import ensure_paths

ensure_paths()

from hsp.config import get_config  # noqa: E402
from hsp.envs.env_wrappers import ChooseDummyVecEnv, ChooseSubprocVecEnv  # noqa: E402

from transplant.adapters.risky_overcooked_env import CATEGORY_KEYS, RiskyOvercooked  # noqa: E402
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


def _render_gif_rank_enabled(all_args, rank: int | None) -> bool:
    gif_episodes = int(getattr(all_args, "render_eval_gif_episodes", 0) or 0)
    if gif_episodes <= 0 or rank is None:
        return False
    thread_count = int(getattr(all_args, "n_eval_rollout_threads", 1) or 1)
    gif_episodes = min(gif_episodes, thread_count)
    return rank >= thread_count - gif_episodes


def make_eval_env(all_args, run_dir, *, render_eval_gifs: bool = False):
    def get_env_fn(rank):
        def init_env():
            env = RiskyOvercooked(all_args, run_dir, featurize_type=("ppo", "ppo"), rank=rank)
            if render_eval_gifs and _render_gif_rank_enabled(all_args, rank):
                env.use_render = True
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ChooseDummyVecEnv([get_env_fn(0)])
    return ChooseSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def _collect_eval_gif_paths(eval_envs, max_paths: int) -> list[Path]:
    if max_paths <= 0 or not hasattr(eval_envs, "get_last_render_gif_paths"):
        return []
    paths = []
    try:
        raw_paths = eval_envs.get_last_render_gif_paths()
    except Exception as exc:
        print(f"warning: failed to collect eval GIF paths: {exc}")
        return []
    for raw_path in raw_paths:
        if isinstance(raw_path, (list, tuple)):
            paths.extend(Path(path) for path in raw_path if path)
        elif raw_path:
            paths.append(Path(raw_path))
    existing_paths = [path for path in paths if path.exists()]
    existing_paths.sort(key=lambda path: path.stat().st_mtime_ns)
    return existing_paths[-max_paths:]


def parse_args(args, parser):
    add_risky_overcooked_args(parser, mode="eval")
    return normalize_risky_args(parser.parse_known_args(args)[0])


def _metric_scalar(eval_infos: dict, key: str) -> float | None:
    values = eval_infos.get(key)
    if not values:
        return None
    return float(values[0])


def _write_raw_metrics(eval_infos: dict, output_path: Path | None) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as file:
        file.write(f"{eval_infos}\n")


def _print_eval_summary(all_args, eval_infos: dict) -> None:
    pair_prefix = f"{all_args.agent0_policy_name}-{all_args.agent1_policy_name}"
    fields = [
        ("sparse", f"{pair_prefix}-eval_ep_sparse_r"),
        ("shaped", f"{pair_prefix}-eval_ep_shaped_r"),
        ("agent0_sparse", f"{pair_prefix}-eval_ep_sparse_r_by_agent0"),
        ("agent1_sparse", f"{pair_prefix}-eval_ep_sparse_r_by_agent1"),
    ]
    parts = []
    for label, key in fields:
        value = _metric_scalar(eval_infos, key)
        if value is not None:
            parts.append(f"{label}={value:g}")
    print(f"eval summary {pair_prefix}: " + ", ".join(parts))


def _adaptive_event_rows(all_args, eval_infos: dict) -> list[dict]:
    agent0 = all_args.agent0_policy_name
    agent1 = all_args.agent1_policy_name
    if agent0 == "hsp_adaptive":
        adaptive_agent_idx = 0
        partner = agent1
    elif agent1 == "hsp_adaptive":
        adaptive_agent_idx = 1
        partner = agent0
    else:
        return []

    pair_prefix = f"{agent0}-{agent1}"
    rows = []
    for event_name in CATEGORY_KEYS:
        key = f"{pair_prefix}-eval_ep_{event_name}_by_agent{adaptive_agent_idx}"
        value = _metric_scalar(eval_infos, key)
        if value is None:
            continue
        rows.append(
            {
                "partner": partner,
                "partner_group": "mep" if partner.startswith("mep") else "hsp",
                "direction": f"{agent0}_vs_{agent1}",
                "adaptive_agent": f"agent{adaptive_agent_idx}",
                "event": event_name,
                "count": value,
            }
        )
    return rows


def _write_event_table_rows(rows: list[dict], output_path: Path | None) -> None:
    if output_path is None or not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _score_row(all_args, eval_infos: dict) -> dict | None:
    agent0 = all_args.agent0_policy_name
    agent1 = all_args.agent1_policy_name
    if agent0 == "hsp_adaptive":
        partner = agent1
        adaptive_agent = "agent0"
    elif agent1 == "hsp_adaptive":
        partner = agent0
        adaptive_agent = "agent1"
    else:
        return None

    pair_prefix = f"{agent0}-{agent1}"
    score = _metric_scalar(eval_infos, f"{pair_prefix}-eval_ep_sparse_r")
    if score is None:
        return None
    return {
        "partner": partner,
        "partner_group": "mep" if partner.startswith("mep") else "hsp",
        "direction": f"{agent0}_vs_{agent1}",
        "adaptive_agent": adaptive_agent,
        "agent0_policy": agent0,
        "agent1_policy": agent1,
        "episode_sparse_score": score,
        "agent0_sparse": _metric_scalar(eval_infos, f"{pair_prefix}-eval_ep_sparse_r_by_agent0"),
        "agent1_sparse": _metric_scalar(eval_infos, f"{pair_prefix}-eval_ep_sparse_r_by_agent1"),
    }


def _write_score_row(row: dict | None, output_path: Path | None) -> None:
    if output_path is None or row is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _gif_manifest_rows(all_args, gif_paths: list[Path]) -> list[dict]:
    if not gif_paths:
        return []
    agent0 = all_args.agent0_policy_name
    agent1 = all_args.agent1_policy_name
    if agent0 == "hsp_adaptive":
        partner = agent1
        adaptive_agent = "agent0"
    elif agent1 == "hsp_adaptive":
        partner = agent0
        adaptive_agent = "agent1"
    else:
        return []

    direction = f"{agent0}_vs_{agent1}"
    gif_label = f"agent0={agent0} | agent1={agent1}"
    return [
        {
            "partner": partner,
            "direction": direction,
            "agent0_policy": agent0,
            "agent1_policy": agent1,
            "adaptive_agent": adaptive_agent,
            "gif_label": gif_label,
            "gif_path": str(path),
        }
        for path in gif_paths
    ]


def _write_gif_manifest_rows(rows: list[dict], output_path: Path | None) -> None:
    if output_path is None or not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _wandb_key_component(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)
    return value.strip("_") or "unknown"


def _gif_panel_name(row: dict) -> str | None:
    agent0 = row.get("agent0_policy", "")
    agent1 = row.get("agent1_policy", "")
    adaptive_agent = row.get("adaptive_agent", "")
    if adaptive_agent == "agent0":
        partner_agent = "agent1"
    elif adaptive_agent == "agent1":
        partner_agent = "agent0"
    elif agent0 == "hsp_adaptive":
        adaptive_agent = "agent0"
        partner_agent = "agent1"
    elif agent1 == "hsp_adaptive":
        adaptive_agent = "agent1"
        partner_agent = "agent0"
    else:
        return None

    return "__".join(
        [
            f"adaptive_{_wandb_key_component(adaptive_agent)}",
            f"partner_{_wandb_key_component(partner_agent)}",
            f"agent0_{_wandb_key_component(agent0)}",
            f"agent1_{_wandb_key_component(agent1)}",
        ]
    )


def _make_gif_panel_payload(gif_rows: list[dict], wandb) -> dict:
    payload = {}
    for row in gif_rows:
        partner = row.get("partner", "")
        gif_path = Path(row.get("gif_path", ""))
        if not partner or not gif_path.is_file():
            continue
        panel_name = _gif_panel_name(row)
        if panel_name is None:
            continue
        payload[f"final_eval_gifs/{panel_name}"] = wandb.Video(str(gif_path), format="gif")
    return payload


def _stream_gif_rows_to_wandb(all_args, gif_rows: list[dict]) -> None:
    if not getattr(all_args, "stream_eval_gifs_to_wandb", False) or not gif_rows:
        return
    run_id = str(getattr(all_args, "stream_gif_wandb_run_id", "") or "").strip()
    if not run_id:
        print("warning: streaming eval GIF upload skipped because stream_gif_wandb_run_id is empty")
        return

    import wandb

    payload = _make_gif_panel_payload(gif_rows, wandb)
    if not payload:
        return

    project = str(getattr(all_args, "stream_gif_wandb_project", "") or all_args.env_name)
    entity = str(getattr(all_args, "stream_gif_wandb_entity", "") or getattr(all_args, "wandb_name", ""))
    run_name = str(getattr(all_args, "stream_gif_wandb_run_name", "") or run_id)
    mode = str(getattr(all_args, "stream_gif_wandb_mode", "") or "online")
    pair_prefix = f"{all_args.agent0_policy_name}-{all_args.agent1_policy_name}"
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        id=run_id,
        resume="allow",
        group=getattr(all_args, "wandb_group_name", "") or f"HSP_{all_args.layout_name}",
        job_type="evaluation_gif_stream",
        mode=mode,
        reinit=True,
        config={
            "layout": all_args.layout_name,
            "pair": pair_prefix,
            "agent0_policy": all_args.agent0_policy_name,
            "agent1_policy": all_args.agent1_policy_name,
        },
        settings=wandb.Settings(
            start_method=os.environ.get("WANDB_START_METHOD", "thread"),
            x_disable_stats=True,
            x_disable_machine_info=True,
        ),
    )
    run.log(payload)
    run.finish()
    print(f"streamed {len(payload)} eval GIF(s) to W&B run {run_id}")


def _log_eval_metrics_to_wandb(all_args, eval_infos: dict) -> None:
    if not all_args.use_wandb:
        return
    import wandb

    pair_prefix = f"{all_args.agent0_policy_name}-{all_args.agent1_policy_name}"
    metrics = {
        f"final_eval/{pair_prefix}/sparse": _metric_scalar(eval_infos, f"{pair_prefix}-eval_ep_sparse_r"),
        f"final_eval/{pair_prefix}/shaped": _metric_scalar(eval_infos, f"{pair_prefix}-eval_ep_shaped_r"),
        f"final_eval/{pair_prefix}/agent0_sparse": _metric_scalar(
            eval_infos, f"{pair_prefix}-eval_ep_sparse_r_by_agent0"
        ),
        f"final_eval/{pair_prefix}/agent1_sparse": _metric_scalar(
            eval_infos, f"{pair_prefix}-eval_ep_sparse_r_by_agent1"
        ),
    }
    metrics = {key: value for key, value in metrics.items() if value is not None}
    if getattr(all_args, "log_eval_scalars", False):
        for key, values in eval_infos.items():
            if values:
                metrics[f"final_eval_raw/{pair_prefix}/{key}"] = float(values[0])
    if metrics:
        wandb.log(metrics, step=all_args.eval_episodes)

    rows = _adaptive_event_rows(all_args, eval_infos)
    if rows:
        table = wandb.Table(columns=list(rows[0]), data=[[row[column] for column in rows[0]] for row in rows])
        wandb.log({f"final_eval/{pair_prefix}/adaptive_event_counts": table}, step=all_args.eval_episodes)


def evaluate_policy_pair(all_args, run_dir=None):
    """Evaluate the currently configured agent0/agent1 pair without owning logging lifecycle."""
    device = setup_device(all_args)
    run_dir = make_run_dir(all_args) if run_dir is None else run_dir
    set_process_title(all_args)
    set_seeds(all_args.seed)

    envs = None
    eval_envs = None
    runner = None
    try:
        envs = make_eval_env(all_args, run_dir)
        eval_envs = make_eval_env(all_args, run_dir, render_eval_gifs=True)
        config = {
            "all_args": all_args,
            "envs": envs,
            "eval_envs": eval_envs,
            "num_agents": all_args.num_agents,
            "device": device,
            "run_dir": run_dir,
        }

        if all_args.share_policy:
            from transplant.runners.risky_overcooked_runner import RiskyOvercookedRunner as Runner
        else:
            from hsp.runner.separated.overcooked_runner import MPERunner as Runner

        runner = Runner(config)

        featurize_type = runner.policy.load_population(all_args.population_yaml_path, evaluation=True)

        map_ea2p = {}
        for env_idx in range(all_args.n_eval_rollout_threads):
            map_ea2p[(env_idx, 0)] = all_args.agent0_policy_name
            map_ea2p[(env_idx, 1)] = all_args.agent1_policy_name
        runner.policy.set_map_ea2p(map_ea2p)

        agent0_featurize_type = featurize_type.get(all_args.agent0_policy_name, "ppo")
        agent1_featurize_type = featurize_type.get(all_args.agent1_policy_name, "ppo")
        eval_envs.reset_featurize_type(
            [(agent0_featurize_type, agent1_featurize_type) for _ in range(all_args.n_eval_rollout_threads)]
        )

        eval_infos = runner.evaluate_with_multi_policy()
        runner.eval_gif_paths = _collect_eval_gif_paths(
            eval_envs, int(getattr(all_args, "render_eval_gif_episodes", 0) or 0)
        )
        return eval_infos, runner
    finally:
        if envs is not None:
            envs.close()
        if eval_envs is not None:
            eval_envs.close()


def record_eval_outputs(
    all_args, eval_infos: dict, *, gif_paths: list[Path] | None = None, log_to_wandb: bool = True
) -> None:
    event_rows = _adaptive_event_rows(all_args, eval_infos)
    score_row = _score_row(all_args, eval_infos)
    gif_rows = _gif_manifest_rows(all_args, gif_paths or [])
    _write_raw_metrics(eval_infos, all_args.metrics_output_path)
    _write_event_table_rows(event_rows, all_args.event_table_output_path)
    _write_score_row(score_row, all_args.score_table_output_path)
    _write_gif_manifest_rows(gif_rows, all_args.gif_manifest_output_path)
    _stream_gif_rows_to_wandb(all_args, gif_rows)
    _print_eval_summary(all_args, eval_infos)
    if log_to_wandb:
        _log_eval_metrics_to_wandb(all_args, eval_infos)


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    assert all_args.algorithm_name == "population"
    all_args.time_cost = 0.0
    all_args.wandb_job_type = "evaluation"

    run_dir = make_run_dir(all_args)
    run = init_wandb(all_args, run_dir)
    runner = None
    try:
        eval_infos, runner = evaluate_policy_pair(all_args, run_dir=run_dir)
        record_eval_outputs(
            all_args,
            eval_infos,
            gif_paths=getattr(runner, "eval_gif_paths", []),
        )
    finally:
        if runner is not None:
            finish_logging(all_args, run, runner)
        elif run is not None:
            run.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
