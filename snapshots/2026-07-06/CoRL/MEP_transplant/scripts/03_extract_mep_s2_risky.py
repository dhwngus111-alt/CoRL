#!/usr/bin/env python
"""Extract the MEP S2 adaptive checkpoint and create eval YAML."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import yaml


MEP_ROOT = Path(__file__).resolve().parents[1]
CORL_ROOT = MEP_ROOT.parent
os.environ["TRANSPLANT_OUTPUT_ROOT"] = os.environ.get("MEP_TRANSPLANT_ROOT", str(MEP_ROOT))
os.environ.setdefault("POLICY_POOL", str(MEP_ROOT / "policy_pool"))
sys.path.insert(0, str(CORL_ROOT))

from transplant.bootstrap import POLICY_POOL_ROOT, TRANSPLANT_ROOT, ensure_paths  # noqa: E402

ensure_paths()

from transplant.common import RISKY_ENV_NAME  # noqa: E402


VERSION_RE = re.compile(r"actor_periodic_(\d+)\.pt$")


def _version(path: Path) -> int:
    match = VERSION_RE.fullmatch(path.name)
    return int(match.group(1)) if match else -1


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as file:
        return yaml.safe_load(file) or {}


def _select_actor(search_root: Path) -> Path:
    actors = sorted(
        search_root.rglob("mep_adaptive/actor_periodic_*.pt"),
        key=lambda path: (_version(path), path.stat().st_mtime_ns),
    )
    if not actors:
        raise FileNotFoundError(f"No mep_adaptive actor checkpoints found under {search_root}")
    return actors[-1]


def extract(args: argparse.Namespace) -> None:
    search_root = args.results_root / RISKY_ENV_NAME / args.layout / "mep" / "mep-S2"
    source_actor = _select_actor(search_root)

    dest_root = args.policy_pool / args.layout / "mep" / "s2"
    dest_root.mkdir(parents=True, exist_ok=True)
    dest_actor = dest_root / "mep_adaptive.pt"
    shutil.copy2(source_actor, dest_actor)
    print(f"{source_actor} -> {dest_actor}")

    train_yaml = _load_yaml(dest_root / "train.yml")
    eval_yaml = {}
    for name, entry in train_yaml.items():
        copied = dict(entry)
        copied["train"] = False
        if name == "mep_adaptive":
            copied["model_path"] = {"actor": f"{args.layout}/mep/s2/mep_adaptive.pt"}
        eval_yaml[name] = copied

    eval_yaml_path = dest_root / "eval.yml"
    with eval_yaml_path.open("w") as file:
        yaml.safe_dump(eval_yaml, file, sort_keys=False)
    print(f"wrote {eval_yaml_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default=os.environ.get("LAYOUT", "risky_dualpath_subgoal"))
    parser.add_argument("--results-root", type=Path, default=TRANSPLANT_ROOT / "results")
    parser.add_argument("--policy-pool", type=Path, default=POLICY_POOL_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    extract(parse_args(argv))


if __name__ == "__main__":
    main()
