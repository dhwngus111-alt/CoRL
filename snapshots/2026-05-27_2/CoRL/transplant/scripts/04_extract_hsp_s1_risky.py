#!/usr/bin/env python
"""Extract local HSP S1 checkpoints into transplant/policy_pool."""

from __future__ import annotations

import argparse
import pickle
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


VERSION_RE = re.compile(r"actor_agent([01])_periodic_(\d+)\.pt$")


def _actor_version(path: Path) -> int:
    match = VERSION_RE.search(path.name)
    if match is None:
        return -1
    return int(match.group(2))


def _seed_from_run(run_dir: Path, fallback: int) -> int:
    for config_path in [run_dir / "policy_config.pkl", run_dir / "files" / "policy_config.pkl"]:
        if config_path.exists():
            try:
                with config_path.open("rb") as file:
                    all_args = pickle.load(file)[0]
                return int(getattr(all_args, "seed", fallback))
            except Exception as exc:
                print(f"warning: failed reading {config_path}: {exc}")
    return fallback


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    print(f"{src} -> {dst}")


def _run_dirs(search_root: Path) -> list[Path]:
    candidates = [path for path in search_root.glob("run*") if path.is_dir()]
    wandb_root = search_root / "wandb"
    if wandb_root.exists():
        candidates.extend(path for path in wandb_root.glob("run-*") if path.is_dir())
    return sorted(set(candidates), key=lambda path: (path.stat().st_mtime, str(path)))


def extract(layout: str, results_root: Path, policy_pool_root: Path) -> None:
    search_root = results_root / RISKY_ENV_NAME / layout / "mappo" / "hsp-S1"
    dest_root = policy_pool_root / layout / "hsp" / "s1"
    eval_yaml = {}

    run_dirs = _run_dirs(search_root)
    for fallback_idx, run_dir in enumerate(run_dirs, start=1):
        seed = _seed_from_run(run_dir, fallback=fallback_idx)
        for agent_idx, tag in [(0, "w0"), (1, "w1")]:
            actors = sorted(
                run_dir.rglob(f"actor_agent{agent_idx}_periodic_*.pt"),
                key=_actor_version,
            )
            if not actors:
                print(f"warning: no actor_agent{agent_idx} checkpoint under {run_dir}")
                continue
            dst = dest_root / f"hsp{seed}_{tag}_actor.pt"
            _copy(actors[-1], dst)
            eval_yaml[f"hsp{seed}_{tag}"] = {
                "policy_config_path": f"{layout}/policy_config/mlp_policy_config.pkl",
                "featurize_type": "ppo",
                "train": False,
                "model_path": {"actor": f"{layout}/hsp/s1/hsp{seed}_{tag}_actor.pt"},
            }

    if eval_yaml:
        with (dest_root / "eval.yml").open("w") as file:
            yaml.safe_dump(eval_yaml, file, sort_keys=False)
        print(f"wrote {dest_root / 'eval.yml'}")


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="risky_multipath")
    parser.add_argument("--results-root", type=Path, default=TRANSPLANT_ROOT / "results")
    parser.add_argument("--policy-pool", type=Path, default=POLICY_POOL_ROOT)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    extract(args.layout, args.results_root, args.policy_pool)


if __name__ == "__main__":
    main()
