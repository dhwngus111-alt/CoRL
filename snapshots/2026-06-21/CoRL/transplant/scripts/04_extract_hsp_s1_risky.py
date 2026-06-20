#!/usr/bin/env python
"""Extract local HSP S1 checkpoints into transplant/policy_pool."""

from __future__ import annotations

import argparse
import csv
import pickle
import re
import shutil
import sys
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path

import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transplant.bootstrap import POLICY_POOL_ROOT, TRANSPLANT_ROOT, ensure_paths

ensure_paths()

from transplant.common import RISKY_ENV_NAME  # noqa: E402


VERSION_RE = re.compile(r"actor_agent([01])_periodic_(\d+)\.pt$")


@dataclass
class ExtractOptions:
    layout: str
    results_root: Path
    policy_pool_root: Path
    condition: str
    episode_length: int | None
    subgoal_disable_steps: int | None
    num_env_steps: int | None
    require_final_version: int | None
    require_seeds: int | None
    dry_run: bool


@dataclass
class SelectedRun:
    run_dir: Path
    seed: int
    actors: dict[int, Path]
    risky_nonzero_count: int | None


def _actor_version(path: Path) -> int:
    match = VERSION_RE.search(path.name)
    if match is None:
        return -1
    return int(match.group(2))


def _load_policy_args(run_dir: Path):
    for config_path in [run_dir / "policy_config.pkl", run_dir / "files" / "policy_config.pkl"]:
        if config_path.exists():
            try:
                with config_path.open("rb") as file:
                    return pickle.load(file)[0]
            except Exception as exc:
                print(f"warning: failed reading {config_path}: {exc}")
    return None


def _seed_from_args(all_args, fallback: int) -> int:
    return int(getattr(all_args, "seed", fallback)) if all_args is not None else fallback


def _int_arg(all_args, name: str) -> int | None:
    value = getattr(all_args, name, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_value(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _nonzero_float(value: str | None) -> bool:
    try:
        return abs(float(str(value or "0").strip())) > 1e-12
    except ValueError:
        return False


def _risky_nonzero_count(run_dir: Path) -> int | None:
    selection_path = run_dir / "hidden_utility_selection.csv"
    if not selection_path.exists():
        return None

    count = 0
    with selection_path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row.get("group") not in {"risked_extra", "risky_multipath"}:
                continue
            if "w0_nonzero" in row:
                count += int(_bool_value(row.get("w0_nonzero")))
            else:
                count += int(_nonzero_float(row.get("w0")))
    return count


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


def _selected_actor(run_dir: Path, agent_idx: int, required_version: int | None) -> Path | None:
    actors = sorted(
        run_dir.rglob(f"actor_agent{agent_idx}_periodic_*.pt"),
        key=_actor_version,
    )
    if required_version is None:
        return actors[-1] if actors else None

    for actor in actors:
        if _actor_version(actor) == required_version:
            return actor
    return None


def _matches_metadata(run_dir: Path, all_args, options: ExtractOptions) -> tuple[bool, int | None]:
    if all_args is None:
        uses_metadata_filter = any(
            value is not None
            for value in [
                options.episode_length,
                options.subgoal_disable_steps,
                options.num_env_steps,
            ]
        )
        if uses_metadata_filter:
            print(f"skip {run_dir}: missing policy_config.pkl for metadata filters")
            return False, None
        return True, None

    layout_name = getattr(all_args, "layout_name", options.layout)
    if layout_name != options.layout:
        print(f"skip {run_dir}: layout_name={layout_name!r} does not match {options.layout!r}")
        return False, None

    checks = [
        ("episode_length", options.episode_length),
        ("subgoal_disable_steps", options.subgoal_disable_steps),
        ("num_env_steps", options.num_env_steps),
    ]
    for name, expected in checks:
        if expected is None:
            continue
        actual = _int_arg(all_args, name)
        if actual != expected:
            print(f"skip {run_dir}: {name}={actual!r}, expected {expected}")
            return False, None

    return True, None


def _matches_condition(run_dir: Path, condition: str) -> tuple[bool, int | None]:
    if condition == "all":
        return True, _risky_nonzero_count(run_dir)

    risky_nonzero_count = _risky_nonzero_count(run_dir)
    if risky_nonzero_count is None:
        print(f"skip {run_dir}: missing hidden_utility_selection.csv for condition={condition}")
        return False, None
    if condition == "risky-aware" and risky_nonzero_count <= 0:
        print(f"skip {run_dir}: risked extra hidden utility is all zero")
        return False, risky_nonzero_count
    if condition == "extra-zero" and risky_nonzero_count != 0:
        print(
            f"skip {run_dir}: risked extra hidden utility has "
            f"{risky_nonzero_count} non-zero w0 entries"
        )
        return False, risky_nonzero_count
    return True, risky_nonzero_count


def _select_runs(search_root: Path, options: ExtractOptions) -> list[SelectedRun]:
    selected = []

    run_dirs = _run_dirs(search_root)
    for fallback_idx, run_dir in enumerate(run_dirs, start=1):
        all_args = _load_policy_args(run_dir)
        metadata_ok, _ = _matches_metadata(run_dir, all_args, options)
        if not metadata_ok:
            continue
        condition_ok, risky_nonzero_count = _matches_condition(run_dir, options.condition)
        if not condition_ok:
            continue

        actors = {}
        for agent_idx in [0, 1]:
            actor = _selected_actor(run_dir, agent_idx, options.require_final_version)
            if actor is None:
                suffix = (
                    f" version {options.require_final_version}"
                    if options.require_final_version is not None
                    else ""
                )
                print(f"skip {run_dir}: missing actor_agent{agent_idx}{suffix}")
                actors = {}
                break
            actors[agent_idx] = actor
        if not actors:
            continue

        selected.append(
            SelectedRun(
                run_dir=run_dir,
                seed=_seed_from_args(all_args, fallback=fallback_idx),
                actors=actors,
                risky_nonzero_count=risky_nonzero_count,
            )
        )

    return selected


def _validate_required_seeds(selected: list[SelectedRun], required_seeds: int | None) -> None:
    if required_seeds is None:
        return

    by_seed = defaultdict(list)
    for run in selected:
        by_seed[run.seed].append(run.run_dir)

    expected = set(range(1, required_seeds + 1))
    actual = set(by_seed)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    duplicates = {seed: paths for seed, paths in by_seed.items() if len(paths) > 1}

    problems = []
    if missing:
        problems.append(f"missing seeds: {missing}")
    if extra:
        problems.append(f"unexpected seeds: {extra}")
    if duplicates:
        duplicate_text = {
            seed: [str(path) for path in paths] for seed, paths in sorted(duplicates.items())
        }
        problems.append(f"duplicate seeds: {duplicate_text}")
    if len(selected) != required_seeds:
        problems.append(f"selected {len(selected)} runs, expected {required_seeds}")

    if problems:
        raise SystemExit("error: invalid selected HSP S1 seed set. " + "; ".join(problems))


def _print_selection(selected: list[SelectedRun]) -> None:
    print(f"selected {len(selected)} HSP S1 runs")
    for run in sorted(selected, key=lambda item: item.seed):
        versions = {
            agent_idx: _actor_version(actor) for agent_idx, actor in sorted(run.actors.items())
        }
        print(
            f"seed {run.seed}: {run.run_dir} "
            f"versions={versions} risky_nonzero={run.risky_nonzero_count}"
        )


def extract(options: ExtractOptions) -> None:
    search_root = options.results_root / RISKY_ENV_NAME / options.layout / "mappo" / "hsp-S1"
    dest_root = options.policy_pool_root / options.layout / "hsp" / "s1"
    eval_yaml = {}

    selected = _select_runs(search_root, options)
    _validate_required_seeds(selected, options.require_seeds)
    _print_selection(selected)

    if options.dry_run:
        print("dry-run: no actors copied and no eval.yml written")
        return

    for run in selected:
        for agent_idx, tag in [(0, "w0"), (1, "w1")]:
            dst = dest_root / f"hsp{run.seed}_{tag}_actor.pt"
            _copy(run.actors[agent_idx], dst)
            eval_yaml[f"hsp{run.seed}_{tag}"] = {
                "policy_config_path": f"{options.layout}/policy_config/mlp_policy_config.pkl",
                "featurize_type": "ppo",
                "train": False,
                "model_path": {"actor": f"{options.layout}/hsp/s1/hsp{run.seed}_{tag}_actor.pt"},
            }

    if eval_yaml:
        with (dest_root / "eval.yml").open("w") as file:
            yaml.safe_dump(eval_yaml, file, sort_keys=False)
        print(f"wrote {dest_root / 'eval.yml'}")


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="risky_multipath_subgoal")
    parser.add_argument("--results-root", type=Path, default=TRANSPLANT_ROOT / "results")
    parser.add_argument("--policy-pool", type=Path, default=POLICY_POOL_ROOT)
    parser.add_argument(
        "--condition",
        choices=["risky-aware", "extra-zero", "all"],
        default="all",
        help="Filter by risked extra hidden utility weights.",
    )
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--subgoal-disable-steps", type=int, default=None)
    parser.add_argument("--num-env-steps", type=int, default=None)
    parser.add_argument("--require-final-version", type=int, default=None)
    parser.add_argument("--require-seeds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    extract(
        ExtractOptions(
            layout=args.layout,
            results_root=args.results_root,
            policy_pool_root=args.policy_pool,
            condition=args.condition,
            episode_length=args.episode_length,
            subgoal_disable_steps=args.subgoal_disable_steps,
            num_env_steps=args.num_env_steps,
            require_final_version=args.require_final_version,
            require_seeds=args.require_seeds,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
