#!/usr/bin/env python
"""Evaluate a CoMeDi adaptive agent against each learned convention."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

from comedi_transplant.bootstrap import COMEDI_TRANSPLANT_ROOT, POLICY_POOL_ROOT, ensure_paths
from comedi_transplant.policy_configs import DEFAULT_CNN_LAYERS


ensure_paths()

from hsp.config import get_config  # noqa: E402
from transplant.common import (  # noqa: E402
    add_risky_overcooked_args,
    finish_logging,
    init_wandb,
    make_run_dir,
    normalize_risky_args,
)
from transplant.eval_risky_hsp import (  # noqa: E402
    evaluate_policy_pair,
    record_eval_outputs,
)


def _add_comedi_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--comedi_adaptive_agent_name", type=str, default="comedi_adaptive")
    parser.add_argument("--partner_policy", type=str, default="")
    parser.add_argument("--partner_group", type=str, default="all")
    parser.add_argument("--final_eval_dir", type=Path, default=None)
    parser.add_argument("--comedi_eval_time_cost", type=float, default=0.0)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parse_argv = list(argv)
    if "--agent0_policy_name" not in parse_argv:
        parse_argv.extend(["--agent0_policy_name", "__placeholder_agent0__"])
    if "--agent1_policy_name" not in parse_argv:
        parse_argv.extend(["--agent1_policy_name", "__placeholder_agent1__"])
    parser = get_config()
    add_risky_overcooked_args(parser, mode="eval")
    _add_comedi_eval_args(parser)
    parser.set_defaults(
        algorithm_name="population",
        experiment_name="comedi-final-eval",
        n_eval_rollout_threads=1,
        episode_length=200,
        eval_episodes=32,
        hidden_size=64,
        layer_N=2,
        activation_id=1,
        cnn_layers_params=DEFAULT_CNN_LAYERS,
        adaptive_policy_name="comedi_adaptive",
    )
    all_args = normalize_risky_args(parser.parse_known_args(parse_argv)[0])
    if all_args.algorithm_name != "population":
        raise ValueError("CoMeDi eval requires --algorithm_name population")
    all_args.time_cost = float(all_args.comedi_eval_time_cost)
    all_args.adaptive_policy_name = all_args.comedi_adaptive_agent_name
    if not all_args.population_yaml_path:
        all_args.population_yaml_path = str(
            POLICY_POOL_ROOT / all_args.layout_name / "comedi" / "s2" / "eval.yml"
        )
    return all_args


def _partners(all_args) -> list[str]:
    data = yaml.safe_load(open(all_args.population_yaml_path)) or {}
    adaptive_name = all_args.comedi_adaptive_agent_name
    if adaptive_name not in data:
        raise ValueError(f"{adaptive_name} not found in {all_args.population_yaml_path}")
    if all_args.partner_policy:
        if all_args.partner_policy not in data:
            raise ValueError(
                f"{all_args.partner_policy} not found in {all_args.population_yaml_path}"
            )
        return [all_args.partner_policy]

    partners = [name for name in data if name != adaptive_name]
    if all_args.partner_group != "all":
        partners = [name for name in partners if name.startswith(all_args.partner_group)]
    if not partners:
        raise ValueError(f"no partner policies found in {all_args.population_yaml_path}")
    return partners


def _eval_paths(all_args) -> tuple[Path, Path, Path, Path]:
    final_eval_dir = all_args.final_eval_dir or (
        COMEDI_TRANSPLANT_ROOT / "final_eval" / all_args.layout_name
    )
    final_eval_dir.mkdir(parents=True, exist_ok=True)
    return (
        final_eval_dir,
        final_eval_dir / "comedi_adaptive_event_counts_long.csv",
        final_eval_dir / "comedi_adaptive_sparse_scores.csv",
        final_eval_dir / "comedi_adaptive_pair_gifs.csv",
    )


def _evaluate_one(base_args, agent0: str, agent1: str, final_eval_dir: Path):
    pair_args = copy.deepcopy(base_args)
    pair_args.agent0_policy_name = agent0
    pair_args.agent1_policy_name = agent1
    pair_args.experiment_name = f"final-{agent0}-{agent1}"
    pair_args.wandb_stage_name = getattr(pair_args, "wandb_stage_name", "") or "comedi_final_eval"
    pair_args.wandb_run_name = (
        getattr(pair_args, "wandb_run_name", "")
        or f"comedi_eval_{pair_args.layout_name}_{agent0}_vs_{agent1}_seed{pair_args.seed}"
    )
    pair_args.render_gif_subdir = f"{agent0}_vs_{agent1}"
    run_dir = make_run_dir(pair_args)
    run = init_wandb(pair_args, run_dir)
    runner = None
    try:
        eval_infos, runner = evaluate_policy_pair(pair_args, run_dir=run_dir)
        record_eval_outputs(
            pair_args,
            eval_infos,
            gif_paths=getattr(runner, "eval_gif_paths", []),
            log_to_wandb=pair_args.use_wandb,
        )
        with (final_eval_dir / f"{agent0}_vs_{agent1}.txt").open("w") as file:
            file.write(f"{eval_infos}\n")
    finally:
        if runner is not None:
            finish_logging(pair_args, run, runner)
        elif run is not None:
            run.finish()


def evaluate_all(all_args) -> None:
    final_eval_dir, event_csv, score_csv, gif_csv = _eval_paths(all_args)
    event_csv.write_text("")
    score_csv.write_text("")
    gif_csv.write_text("")
    all_args.event_table_output_path = event_csv
    all_args.score_table_output_path = score_csv
    all_args.gif_manifest_output_path = gif_csv

    partners = _partners(all_args)
    adaptive_name = all_args.comedi_adaptive_agent_name
    print(f"CoMeDi final eval partners ({len(partners)}): {' '.join(partners)}")
    for partner in partners:
        _evaluate_one(all_args, adaptive_name, partner, final_eval_dir)
        _evaluate_one(all_args, partner, adaptive_name, final_eval_dir)


def main(argv: list[str] | None = None) -> None:
    all_args = parse_args(sys.argv[1:] if argv is None else argv)
    evaluate_all(all_args)


if __name__ == "__main__":
    main()
