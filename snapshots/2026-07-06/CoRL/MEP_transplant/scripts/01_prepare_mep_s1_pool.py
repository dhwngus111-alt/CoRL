#!/usr/bin/env python
"""Prepare the MEP Stage-1 policy pool for Risky Overcooked MEP S2.

The HSP/transplant pipeline trained a larger MEP population.  Following the MEP
paper baseline, this script takes only the first five source policies
(mep1..mep5) and copies their init/mid/final checkpoints into the
MEP_transplant output tree for a 15-policy Stage-2 partner pool.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml


MEP_ROOT = Path(__file__).resolve().parents[1]
CORL_ROOT = MEP_ROOT.parent
os.environ["TRANSPLANT_OUTPUT_ROOT"] = os.environ.get("MEP_TRANSPLANT_ROOT", str(MEP_ROOT))
os.environ.setdefault("POLICY_POOL", str(MEP_ROOT / "policy_pool"))
sys.path.insert(0, str(CORL_ROOT))

from transplant.bootstrap import POLICY_POOL_ROOT, ensure_paths  # noqa: E402

ensure_paths()

from transplant.build_policy_configs import build as build_policy_configs  # noqa: E402
from transplant.common import DEFAULT_EPISODE_LENGTH  # noqa: E402


CHECKPOINT_TAGS = {"init": 1, "mid": 2, "final": 3}


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    print(f"{src} -> {dst}")


def _source_pool_checkpoints(
    layout: str,
    source_policy_pool: Path,
    policy_name: str,
) -> dict[str, Path] | None:
    source_s1 = source_policy_pool / layout / "mep" / "s1"
    selected = {
        tag: source_s1 / f"{policy_name}_{tag}_actor.pt"
        for tag in CHECKPOINT_TAGS
    }
    if all(path.is_file() for path in selected.values()):
        return selected
    return None


def _source_pool_population(
    layout: str,
    source_policy_pool: Path,
    selected_policy_count: int,
) -> tuple[dict[str, dict[str, Path]] | None, list[str]]:
    population = {}
    missing = []
    for policy_idx in range(1, selected_policy_count + 1):
        policy_name = f"mep{policy_idx}"
        selected = _source_pool_checkpoints(layout, source_policy_pool, policy_name)
        if selected is None:
            missing.append(policy_name)
        else:
            population[policy_name] = selected
    if missing:
        return None, missing
    return population, []


def _policy_entry(layout: str, policy_name: str, tag: str) -> dict:
    return {
        "policy_config_path": f"{layout}/policy_config/mlp_policy_config.pkl",
        "featurize_type": "ppo",
        "train": False,
        "model_path": {"actor": f"{layout}/mep/s1/{policy_name}_{tag}_actor.pt"},
    }


def prepare(args: argparse.Namespace) -> None:
    layout = args.layout
    policy_pool_root = args.policy_pool.resolve()
    source_policy_pool = args.source_policy_pool.resolve()

    build_policy_configs(layout, args.episode_length, policy_pool_root, args.selected_policy_count)

    s1_root = policy_pool_root / layout / "mep" / "s1"
    s2_root = policy_pool_root / layout / "mep" / "s2"
    s1_root.mkdir(parents=True, exist_ok=True)
    s2_root.mkdir(parents=True, exist_ok=True)

    s1_eval = {}
    s2_train = {
        "mep_adaptive": {
            "policy_config_path": f"{layout}/policy_config/rnn_policy_config.pkl",
            "featurize_type": "ppo",
            "train": True,
        }
    }
    metadata = {
        "layout": layout,
        "source_policy_pool": str(source_policy_pool),
        "selected_policy_count": args.selected_policy_count,
        "selected_policy_names": [f"mep{idx}" for idx in range(1, args.selected_policy_count + 1)],
        "checkpoints_per_policy": len(CHECKPOINT_TAGS),
        "policies": {},
    }
    source_pool_population, missing_source_pool = _source_pool_population(
        layout,
        source_policy_pool,
        args.selected_policy_count,
    )
    if source_pool_population is None:
        missing = ", ".join(missing_source_pool)
        raise FileNotFoundError(
            f"Missing extracted MEP policies in {source_policy_pool / layout / 'mep' / 's1'}: {missing}. "
            "MEP_transplant expects the HSP/transplant pipeline to have already extracted "
            "init/middle/final checkpoints for mep1..mep5."
        )

    for policy_idx in range(1, args.selected_policy_count + 1):
        policy_name = f"mep{policy_idx}"
        selected = source_pool_population[policy_name]

        metadata["policies"][policy_name] = {}
        for tag, src in selected.items():
            dst = s1_root / f"{policy_name}_{tag}_actor.pt"
            _copy(src, dst)
            metadata["policies"][policy_name][tag] = {
                "source": str(src),
                "source_kind": "policy_pool",
            }
            entry_name = f"{policy_name}_{CHECKPOINT_TAGS[tag]}"
            entry = _policy_entry(layout, policy_name, tag)
            s1_eval[entry_name] = entry
            s2_train[entry_name] = dict(entry)

    for path, data in [
        (s1_root / "eval.yml", s1_eval),
        (s2_root / "train.yml", s2_train),
        (s1_root / "source_metadata.yml", metadata),
    ]:
        with path.open("w") as file:
            yaml.safe_dump(data, file, sort_keys=False)
        print(f"wrote {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default=os.environ.get("LAYOUT", "risky_dualpath_subgoal"))
    parser.add_argument(
        "--source-policy-pool",
        type=Path,
        default=Path(os.environ.get("MEP_SOURCE_POLICY_POOL", CORL_ROOT / "transplant" / "policy_pool")),
    )
    parser.add_argument("--policy-pool", type=Path, default=POLICY_POOL_ROOT)
    parser.add_argument("--episode-length", type=int, default=DEFAULT_EPISODE_LENGTH)
    selected_default = int(os.environ.get("MEP_SELECTED_POLICY_COUNT", os.environ.get("MEP_S1_POPULATION_SIZE", "5")))
    parser.add_argument("--selected-policy-count", type=int, default=selected_default)
    parser.add_argument("--mep-population-size", type=int, dest="selected_policy_count")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    prepare(parse_args(argv))


if __name__ == "__main__":
    main()
