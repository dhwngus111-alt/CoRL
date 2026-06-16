#!/usr/bin/env python
"""Train all HSP S1 seeds into one W&B run while keeping local runs split."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transplant.bootstrap import ensure_paths

ensure_paths()

from hsp.config import get_config  # noqa: E402

from transplant.common import (  # noqa: E402
    add_risky_overcooked_args,
    init_wandb,
    make_numbered_run_dir,
    make_results_root,
    normalize_risky_args,
)
from transplant.train_risky_hsp import train_hsp_s1, validate_hsp_args  # noqa: E402


def parse_args(args, parser):
    add_risky_overcooked_args(parser, mode="hsp")
    parser.add_argument("--seed_max", type=int, default=int(os.environ.get("SEED_MAX", 36)))
    parser.add_argument("--wandb_bundle_run_name", type=str, default="")
    return normalize_risky_args(parser.parse_known_args(args)[0])


def _default_run_name(all_args) -> str:
    stage_name = str(getattr(all_args, "wandb_stage_name", "") or "").strip()
    prefix = f"{stage_name}_" if stage_name else ""
    return f"{prefix}{all_args.algorithm_name}_{all_args.experiment_name}_seeds1-{all_args.seed_max}"


def main(args=None):
    parser = get_config()
    base_args = parse_args(sys.argv[1:] if args is None else args, parser)
    validate_hsp_args(base_args)

    base_args.wandb_job_type = "training"
    base_args.wandb_run_name = base_args.wandb_bundle_run_name or _default_run_name(base_args)
    base_args.wandb_group_name = base_args.layout_name
    base_args.wandb_tags = ["hsp_s1_bundle", f"seeds_1_{base_args.seed_max}"]

    bundle_run_dir = make_results_root(base_args)
    bundle_run = init_wandb(base_args, bundle_run_dir) if base_args.use_wandb else None

    try:
        for seed in range(1, int(base_args.seed_max) + 1):
            print(f"=== HSP S1 seed {seed}/{base_args.seed_max} ===")
            seed_args = copy.deepcopy(base_args)
            seed_args.seed = seed
            seed_args.hsp_wandb_seed_label = f"seed{seed:02d}"

            local_run_dir = make_numbered_run_dir(seed_args)
            train_hsp_s1(
                seed_args,
                run_dir=local_run_dir,
                run=bundle_run,
                finish_wandb=not seed_args.use_wandb,
                local_run_dir=local_run_dir if seed_args.use_wandb else None,
            )
    finally:
        if bundle_run is not None:
            bundle_run.finish()


if __name__ == "__main__":
    main()
