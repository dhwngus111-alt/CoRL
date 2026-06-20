#!/usr/bin/env python
"""Extract local MEP S1 checkpoints into transplant/policy_pool."""

from __future__ import annotations

import argparse
import math
import pickle
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transplant.bootstrap import POLICY_POOL_ROOT, TRANSPLANT_ROOT, ensure_paths

ensure_paths()

from transplant.common import RISKY_ENV_NAME  # noqa: E402


VERSION_RE = re.compile(r"actor_periodic_(\d+)\.pt$")
MEP_SELF_PLAY_TAG = "{policy_name}-{policy_name}-ep_sparse_r"
CHECKPOINT_TAGS = {"init": 1, "mid": 2, "final": 3}


def _has_requirements(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            args.require_p_slip,
            args.require_episode_length,
            args.require_time_cost,
        )
    )


def _version(path: Path) -> int:
    match = VERSION_RE.search(path.name)
    if match is None:
        return -1
    return int(match.group(1))


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    print(f"{src} -> {dst}")


def _latest_by_version(paths: list[Path]) -> list[Path]:
    by_version = {}
    for path in paths:
        version = _version(path)
        if version < 0:
            continue
        current = by_version.get(version)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            by_version[version] = path
    return [by_version[version] for version in sorted(by_version)]


def _find_policy_config(actor: Path, search_root: Path) -> Path | None:
    for parent in actor.parents:
        candidate = parent / "policy_config.pkl"
        if candidate.exists():
            return candidate
        if parent == search_root:
            break
    return None


def _load_args_from_policy_config(path: Path):
    with path.open("rb") as file:
        payload = pickle.load(file)
    if isinstance(payload, (tuple, list)) and payload:
        return payload[0]
    return payload


def _metadata_value(metadata, name: str):
    if isinstance(metadata, dict):
        return metadata.get(name)
    return getattr(metadata, name, None)


def _format_expected(actual, expected) -> str:
    return f"expected {expected!r}, got {actual!r}"


def _validate_actor_metadata(actor: Path, search_root: Path, args: argparse.Namespace) -> None:
    config_path = _find_policy_config(actor, search_root)
    if config_path is None:
        raise SystemExit(f"error: no policy_config.pkl found for {actor}")

    metadata = _load_args_from_policy_config(config_path)
    checks = (
        ("p_slip", args.require_p_slip, float),
        ("episode_length", args.require_episode_length, int),
        ("time_cost", args.require_time_cost, float),
    )
    for name, expected, caster in checks:
        if expected is None:
            continue
        actual = _metadata_value(metadata, name)
        if actual is None:
            raise SystemExit(f"error: {config_path} does not define {name}")
        try:
            actual_value = caster(actual)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"error: invalid {name} in {config_path}: {actual!r}") from exc
        if caster is float:
            ok = abs(actual_value - expected) <= 1e-8
        else:
            ok = actual_value == expected
        if not ok:
            raise SystemExit(
                f"error: metadata mismatch for {actor}: {name} "
                f"{_format_expected(actual_value, expected)} in {config_path}"
            )


def _nearest_actor(actors: list[Path], target_version: int) -> Path:
    return min(actors, key=lambda path: (abs(_version(path) - target_version), _version(path)))


def _load_sparse_history(search_root: Path, policy_name: str) -> list[tuple[int, float]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return []

    tag = MEP_SELF_PLAY_TAG.format(policy_name=policy_name)
    events = []
    event_files = sorted(search_root.rglob("events.out.tfevents.*"), key=lambda path: path.stat().st_mtime)
    for event_file in event_files:
        try:
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
            if tag not in accumulator.Tags().get("scalars", []):
                continue
            events.extend((int(event.step), float(event.value)) for event in accumulator.Scalars(tag))
        except Exception:
            continue

    deduped = {}
    for step, value in events:
        if math.isnan(value):
            value = 0.0
        deduped[step] = value
    return sorted(deduped.items())


def _select_mid_actor(actors: list[Path], history: list[tuple[int, float]]) -> Path:
    if not history:
        return actors[len(actors) // 2]

    final_sparse_r = history[-1][1]
    mid_sparse_r = final_sparse_r / 2.0
    target_step = min(history, key=lambda item: abs(item[1] - mid_sparse_r))[0]
    max_history_step = max(step for step, _ in history)
    if max_history_step <= 0:
        return actors[len(actors) // 2]

    max_actor_version = max(_version(actor) for actor in actors)
    estimated_actor_version = int(target_step / max_history_step * (max_actor_version + 1))
    return _nearest_actor(actors, estimated_actor_version)


def _select_checkpoints(actors: list[Path], history: list[tuple[int, float]]) -> dict[str, Path]:
    init_actor = next((actor for actor in actors if _version(actor) == 0), actors[0])
    return {
        "init": init_actor,
        "mid": _select_mid_actor(actors, history),
        "final": actors[-1],
    }


def extract(
    layout: str,
    results_root: Path,
    policy_pool_root: Path,
    max_policies: int,
    args: argparse.Namespace | None = None,
) -> None:
    if args is None:
        args = argparse.Namespace(
            require_p_slip=None,
            require_episode_length=None,
            require_time_cost=None,
        )
    search_root = results_root / RISKY_ENV_NAME / layout / "mep" / "mep-S1"
    dest_root = policy_pool_root / layout / "mep" / "s1"
    grouped = defaultdict(list)
    validate_metadata = _has_requirements(args)
    for actor in search_root.rglob("mep*/actor_periodic_*.pt"):
        if validate_metadata:
            _validate_actor_metadata(actor, search_root, args)
        grouped[actor.parent.name].append(actor)

    eval_yaml = {}
    for policy_idx in range(1, max_policies + 1):
        policy_name = f"mep{policy_idx}"
        actors = _latest_by_version(grouped.get(policy_name, []))
        if not actors:
            raise FileNotFoundError(f"No actor checkpoints found for {policy_name} under {search_root}")
        history = _load_sparse_history(search_root, policy_name)
        selected = _select_checkpoints(actors, history)
        for tag, src in selected.items():
            dst = dest_root / f"{policy_name}_{tag}_actor.pt"
            _copy(src, dst)
            eval_yaml[f"{policy_name}_{CHECKPOINT_TAGS[tag]}"] = {
                "policy_config_path": f"{layout}/policy_config/mlp_policy_config.pkl",
                "featurize_type": "ppo",
                "train": False,
                "model_path": {"actor": f"{layout}/mep/s1/{policy_name}_{tag}_actor.pt"},
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
    parser.add_argument("--max-policies", type=int, default=6)
    parser.add_argument("--require-p-slip", type=float, default=None)
    parser.add_argument("--require-episode-length", type=int, default=None)
    parser.add_argument("--require-time-cost", type=float, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    extract(args.layout, args.results_root, args.policy_pool, args.max_policies, args)


if __name__ == "__main__":
    main()
