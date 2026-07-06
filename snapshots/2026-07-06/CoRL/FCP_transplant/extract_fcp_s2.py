#!/usr/bin/env python
"""Extract the trained FCP Stage-2 adaptive checkpoint and write eval YAML.

Mirror of ``transplant/scripts/08_extract_hsp_s2_risky.py`` for FCP: take the
latest ``fcp_adaptive/actor_periodic_*.pt`` from the Stage-2 run, copy it to
``<pool>/<layout>/fcp/s2/fcp_adaptive.pt``, and derive ``fcp/s2/eval.yml`` from
``fcp/s2/train.yml`` (every entry frozen; the adaptive agent gets a model_path).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FCP_transplant.bootstrap import POLICY_POOL_ROOT, OUTPUT_ROOT, ensure_paths

ensure_paths()

from transplant.common import RISKY_ENV_NAME  # noqa: E402

VERSION_RE = re.compile(r"actor_periodic_(\d+)\.pt$")


def _version(path: Path) -> int:
    match = VERSION_RE.search(path.name)
    return int(match.group(1)) if match else -1


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as file:
        return yaml.safe_load(file) or {}


def extract(layout: str, results_root: Path, policy_pool_root: Path,
            adaptive_agent_name: str) -> None:
    search_root = results_root / RISKY_ENV_NAME / layout / "adaptive" / "fcp-S2"
    actors = sorted(
        search_root.rglob(f"{adaptive_agent_name}/actor_periodic_*.pt"),
        key=lambda p: (_version(p), p.stat().st_mtime),
    )
    if not actors:
        raise FileNotFoundError(
            f"No {adaptive_agent_name} actor checkpoints found under {search_root}"
        )

    dest_root = policy_pool_root / layout / "fcp" / "s2"
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / f"{adaptive_agent_name}.pt"
    if actors[-1].resolve() != dest.resolve():
        shutil.copy2(actors[-1], dest)
    print(f"{actors[-1]} -> {dest}")

    train_yaml = _load_yaml(dest_root / "train.yml")
    if not train_yaml:
        raise FileNotFoundError(f"missing {dest_root / 'train.yml'}; run S1 extraction first")

    eval_yaml = {}
    for name, entry in train_yaml.items():
        copied = dict(entry)
        copied["train"] = False
        if name == adaptive_agent_name:
            copied["model_path"] = {"actor": f"{layout}/fcp/s2/{adaptive_agent_name}.pt"}
        eval_yaml[name] = copied

    eval_path = dest_root / "eval.yml"
    with eval_path.open("w") as file:
        yaml.safe_dump(eval_yaml, file, sort_keys=False)
    print(f"wrote {eval_path}")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", default="risky_dualpath_subgoal")
    parser.add_argument("--results-root", type=Path, default=OUTPUT_ROOT / "results")
    parser.add_argument("--policy-pool", type=Path, default=POLICY_POOL_ROOT)
    parser.add_argument("--adaptive-agent-name", default="fcp_adaptive")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    extract(args.layout, args.results_root, args.policy_pool, args.adaptive_agent_name)


if __name__ == "__main__":
    main()
