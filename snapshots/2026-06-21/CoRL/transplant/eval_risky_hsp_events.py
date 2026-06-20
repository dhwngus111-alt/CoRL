#!/usr/bin/env python
"""Evaluate all HSP S1 biased pairs into one W&B run."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transplant.bootstrap import POLICY_POOL_ROOT, TRANSPLANT_ROOT, ensure_paths

ensure_paths()

from hsp.config import get_config  # noqa: E402

from transplant.adapters.risky_overcooked_env import CATEGORY_KEYS  # noqa: E402
from transplant.common import (  # noqa: E402
    add_risky_overcooked_args,
    finish_logging,
    init_wandb,
    make_results_root,
    normalize_risky_args,
)
from transplant.eval_risky_hsp import (  # noqa: E402
    _metric_scalar,
    evaluate_policy_pair,
    record_eval_outputs,
)


def parse_args(args, parser):
    add_risky_overcooked_args(parser, mode="event_eval")
    parser.add_argument("--seed_max", type=int, default=int(os.environ.get("SEED_MAX", 36)))
    parser.add_argument("--wandb_bundle_run_name", type=str, default="")
    parser.add_argument("--metrics_output_dir", type=Path, default=None)
    parser.add_argument(
        "--event_gif_episodes",
        type=int,
        default=int(os.environ.get("EVENT_EVAL_GIF_EPISODES", 3)),
    )
    return normalize_risky_args(parser.parse_known_args(args)[0])


def _default_run_name(all_args) -> str:
    stage_name = str(getattr(all_args, "wandb_stage_name", "") or "").strip()
    prefix = f"{stage_name}_" if stage_name else ""
    run_name = f"{prefix}{all_args.algorithm_name}_{all_args.experiment_name}_seeds1-{all_args.seed_max}"
    run_prefix = str(
        getattr(all_args, "wandb_run_prefix", "") or f"HSP_{all_args.layout_name}"
    ).strip()
    if stage_name and run_prefix:
        return (
            f"{stage_name}_{run_prefix}_{all_args.algorithm_name}_"
            f"{all_args.experiment_name}_seeds1-{all_args.seed_max}"
        )
    return f"{run_prefix}_{run_name}" if run_prefix else run_name


def _actor_root(layout: str) -> Path:
    policy_pool = Path(os.environ.get("POLICY_POOL", str(POLICY_POOL_ROOT)))
    return policy_pool / layout / "hsp" / "s1"


def _pair_rows(pair_prefix: str, seed: int, direction: str, eval_infos: dict) -> tuple[dict, list[dict]]:
    summary = {
        "seed": seed,
        "direction": direction,
        "pair": pair_prefix,
        "sparse": _metric_scalar(eval_infos, f"{pair_prefix}-eval_ep_sparse_r"),
        "shaped": _metric_scalar(eval_infos, f"{pair_prefix}-eval_ep_shaped_r"),
        "agent0_sparse": _metric_scalar(eval_infos, f"{pair_prefix}-eval_ep_sparse_r_by_agent0"),
        "agent1_sparse": _metric_scalar(eval_infos, f"{pair_prefix}-eval_ep_sparse_r_by_agent1"),
    }
    event_rows = []
    for event_name in CATEGORY_KEYS:
        agent0_count = _metric_scalar(eval_infos, f"{pair_prefix}-eval_ep_{event_name}_by_agent0")
        agent1_count = _metric_scalar(eval_infos, f"{pair_prefix}-eval_ep_{event_name}_by_agent1")
        if agent0_count is None and agent1_count is None:
            continue
        event_rows.append(
            {
                "seed": seed,
                "direction": direction,
                "pair": pair_prefix,
                "event": event_name,
                "agent0_count": agent0_count,
                "agent1_count": agent1_count,
            }
        )
    return summary, event_rows


def _table(rows: list[dict], columns: list[str] | None = None):
    import wandb

    columns = columns or list(rows[0])
    return wandb.Table(columns=columns, data=[[row.get(column) for column in columns] for row in rows])


def _wandb_pair_section(pair_prefix: str) -> str:
    # W&B workspaces group panels by the first slash-separated path component.
    return f"event_eval_{pair_prefix.replace('/', '_')}"


def _log_pair_to_wandb(
    all_args,
    pair_index: int,
    summary: dict,
    event_rows: list[dict],
    gif_paths: list[Path],
) -> None:
    if not all_args.use_wandb:
        return
    import wandb

    pair_prefix = summary["pair"]
    pair_section = _wandb_pair_section(pair_prefix)
    payload = {
        f"{pair_section}/sparse": summary["sparse"],
        f"{pair_section}/shaped": summary["shaped"],
        f"{pair_section}/agent0_sparse": summary["agent0_sparse"],
        f"{pair_section}/agent1_sparse": summary["agent1_sparse"],
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    if event_rows:
        pair_columns = ["pair", "event", "agent0_count", "agent1_count"]
        payload[f"{pair_section}/event_counts"] = _table(event_rows, columns=pair_columns)
    for gif_index, gif_path in enumerate(gif_paths, start=1):
        if gif_path.exists():
            payload[f"{pair_section}/eval_gif_{gif_index}"] = wandb.Video(
                str(gif_path), format="gif"
            )
    if payload:
        wandb.log(payload, step=pair_index)


def _log_summary_tables_to_wandb(all_args, pair_index: int, summaries: list[dict], event_rows: list[dict]) -> None:
    if not all_args.use_wandb:
        return
    import wandb

    payload = {}
    if summaries:
        payload["event_eval_summary/pair_summary"] = _table(summaries)
    if event_rows:
        payload["event_eval_summary/event_counts_long"] = _table(event_rows)
    if payload:
        wandb.log(payload, step=pair_index)


def _write_header(logfile: Path, header: str, *, append: bool) -> None:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with logfile.open(mode) as file:
        file.write(f"{header}\n")
    print(header)


def _directions(seed: int) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    return (
        (f"hsp{seed}_w0", f"hsp{seed}_w1", "w0-w1"),
        (f"hsp{seed}_w1", f"hsp{seed}_w0", "w1-w0"),
    )


def main(args=None):
    parser = get_config()
    base_args = parse_args(sys.argv[1:] if args is None else args, parser)
    assert base_args.algorithm_name == "population"

    base_args.time_cost = 0.0
    base_args.wandb_job_type = "evaluation"
    base_args.wandb_run_name = base_args.wandb_bundle_run_name or _default_run_name(base_args)
    base_args.wandb_group_name = base_args.wandb_group_name or f"HSP_{base_args.layout_name}"
    base_args.wandb_tags = ["hsp_s1_event_eval", f"seeds_1_{base_args.seed_max}"]

    if base_args.metrics_output_dir is None:
        base_args.metrics_output_dir = TRANSPLANT_ROOT / "biased_eval" / base_args.layout_name

    run_dir = make_results_root(base_args)
    bundle_run = init_wandb(base_args, run_dir) if base_args.use_wandb else None
    actor_root = _actor_root(base_args.layout_name)
    pair_summaries = []
    all_event_rows = []
    pair_index = 0

    try:
        for seed in range(1, int(base_args.seed_max) + 1):
            if not (actor_root / f"hsp{seed}_w0_actor.pt").exists():
                print(f"skip hsp{seed}: missing w0 actor")
                continue
            if not (actor_root / f"hsp{seed}_w1_actor.pt").exists():
                print(f"skip hsp{seed}: missing w1 actor")
                continue

            logfile = base_args.metrics_output_dir / f"eval_hsp{seed}.txt"
            for direction_index, (agent0, agent1, direction) in enumerate(_directions(seed)):
                pair_index += 1
                pair_args = copy.deepcopy(base_args)
                pair_args.seed = seed
                pair_args.agent0_policy_name = agent0
                pair_args.agent1_policy_name = agent1
                pair_args.metrics_output_path = logfile
                pair_args.event_table_output_path = None
                pair_args.log_eval_scalars = False
                pair_args.render_gif_subdir = f"{agent0}-{agent1}"
                if base_args.use_wandb:
                    pair_args.render_eval_gif_episodes = min(
                        max(0, int(base_args.event_gif_episodes)),
                        int(pair_args.eval_episodes),
                    )
                else:
                    pair_args.render_eval_gif_episodes = 0

                header = f"=== eval {agent0} vs {agent1} ==="
                _write_header(logfile, header, append=direction_index != 0)
                eval_infos, runner = evaluate_policy_pair(pair_args, run_dir=run_dir)
                try:
                    record_eval_outputs(pair_args, eval_infos, log_to_wandb=False)
                    summary, event_rows = _pair_rows(f"{agent0}-{agent1}", seed, direction, eval_infos)
                    pair_summaries.append(summary)
                    all_event_rows.extend(event_rows)
                    gif_paths = getattr(runner, "eval_gif_paths", [])
                    _log_pair_to_wandb(base_args, pair_index, summary, event_rows, gif_paths)
                finally:
                    if not pair_args.use_wandb:
                        finish_logging(pair_args, None, runner)
        _log_summary_tables_to_wandb(base_args, pair_index + 1, pair_summaries, all_event_rows)
    finally:
        if bundle_run is not None:
            bundle_run.finish()


if __name__ == "__main__":
    main()
