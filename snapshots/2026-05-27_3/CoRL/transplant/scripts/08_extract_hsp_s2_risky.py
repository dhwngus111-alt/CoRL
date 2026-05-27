#!/usr/bin/env python
"""Extract the HSP S2 adaptive checkpoint and create eval YAML."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transplant.bootstrap import POLICY_POOL_ROOT, TRANSPLANT_ROOT, ensure_paths

ensure_paths()

from transplant.common import RISKY_ENV_NAME  # noqa: E402


VERSION_RE = re.compile(r"actor_periodic_(\d+)\.pt$")


def _version(path: Path) -> int:
    match = VERSION_RE.search(path.name)
    if match is None:
        return -1
    return int(match.group(1))


def _checkpoint_sort_key(path: Path) -> tuple[int, float]:
    return (_version(path), path.stat().st_mtime)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as file:
        return yaml.safe_load(file) or {}


def _log_to_wandb(args, src_actor: Path, dest_actor: Path, eval_yaml_path: Path) -> None:
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or f"08_extract_hsp-S2_{args.layout}",
        group=args.layout,
        job_type="artifact",
        mode=args.wandb_mode,
        config={
            "layout": args.layout,
            "source_actor": str(src_actor),
            "dest_actor": str(dest_actor),
            "eval_yaml": str(eval_yaml_path),
            "checkpoint_step": _version(src_actor),
        },
        settings=wandb.Settings(
            start_method=os.environ.get("WANDB_START_METHOD", "thread"),
            x_disable_stats=True,
            x_disable_machine_info=True,
        ),
    )
    artifact = wandb.Artifact(
        name=f"hsp-s2-adaptive-{args.layout}",
        type="model",
        metadata={
            "layout": args.layout,
            "source_actor": str(src_actor),
            "checkpoint_step": _version(src_actor),
        },
    )
    artifact.add_file(str(dest_actor), name=f"{args.layout}/hsp/s2/hsp_adaptive.pt")
    artifact.add_file(str(eval_yaml_path), name=f"{args.layout}/hsp/s2/eval.yml")
    train_yaml_path = dest_actor.parent / "train.yml"
    if train_yaml_path.exists():
        artifact.add_file(str(train_yaml_path), name=f"{args.layout}/hsp/s2/train.yml")
    run.log({"extract/checkpoint_step": _version(src_actor)})
    run.log_artifact(artifact)
    run.finish()
    print(f"wandb artifact uploaded: {artifact.name}")


def extract(args) -> None:
    layout = args.layout
    results_root = args.results_root
    policy_pool_root = args.policy_pool
    search_root = results_root / RISKY_ENV_NAME / layout / "adaptive" / "hsp-S2"
    actors = sorted(search_root.rglob("hsp_adaptive/actor_periodic_*.pt"), key=_checkpoint_sort_key)
    if not actors:
        raise FileNotFoundError(f"No hsp_adaptive actor checkpoints found under {search_root}")

    dest_root = policy_pool_root / layout / "hsp" / "s2"
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / "hsp_adaptive.pt"
    if actors[-1].resolve() != dest.resolve():
        shutil.copy2(actors[-1], dest)
    print(f"{actors[-1]} -> {dest}")

    train_yaml = _load_yaml(dest_root / "train.yml")
    eval_yaml = {}
    for name, entry in train_yaml.items():
        copied = dict(entry)
        copied["train"] = False
        if name == "hsp_adaptive":
            copied["model_path"] = {"actor": f"{layout}/hsp/s2/hsp_adaptive.pt"}
        eval_yaml[name] = copied

    eval_yaml_path = dest_root / "eval.yml"
    with eval_yaml_path.open("w") as file:
        yaml.safe_dump(eval_yaml, file, sort_keys=False)
    print(f"wrote {eval_yaml_path}")

    if args.use_wandb:
        _log_to_wandb(args, actors[-1], dest, eval_yaml_path)


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="risky_multipath")
    parser.add_argument("--results-root", type=Path, default=TRANSPLANT_ROOT / "results")
    parser.add_argument("--policy-pool", type=Path, default=POLICY_POOL_ROOT)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=RISKY_ENV_NAME)
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get(
            "WANDB_ENTITY",
            "dhwngus41-daegu-gyeongbuk-institute-of-science-technology",
        ),
    )
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    extract(args)


if __name__ == "__main__":
    main()
