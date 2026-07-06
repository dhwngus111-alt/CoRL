#!/usr/bin/env python
"""Extract FCP Stage-1 self-play checkpoints into the FCP policy pool.

For each self-play seed we pick three checkpoints -- exactly HSP's FCP filter
(``extract_sp_S1_models.py``) and transplant's MEP filter
(``02_extract_mep_s1_risky.py``):

    init   the earliest checkpoint (version 0, ~randomly initialised)
    mid    the checkpoint whose episode sparse reward is closest to half of the
           final sparse reward (a "half-trained" partner)
    final  the last checkpoint (a fully-trained partner)

Stage-1 is SHARED-policy self-play (one policy plays both slots), so the runner
saves a single ``actor_periodic_*.pt`` per checkpoint step (not per-agent
``actor_agent{0,1}_periodic_*.pt``, which is the separated/HSP-S1 convention).
With N seeds this yields 3*N frozen partners (HSP FCP: 12 seeds -> 36).

Outputs, under ``<policy_pool>/<layout>/``:
    fcp/s1/sp{seed}_{init,mid,final}_actor.pt   copied checkpoints
    fcp/s1/eval.yml                             pool-only eval config
    fcp/s2/train.yml                            Stage-2 training config
        (fcp_adaptive: rnn, train=True) + (36 partners: mlp, train=False)
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FCP_transplant.bootstrap import POLICY_POOL_ROOT, OUTPUT_ROOT, ensure_paths

ensure_paths()

from transplant.common import RISKY_ENV_NAME  # noqa: E402

# Shared-policy self-play saves one actor per checkpoint step: actor_periodic_N.pt
# (the separated runner would instead save actor_agent{0,1}_periodic_N.pt).
VERSION_RE = re.compile(r"actor_periodic_(\d+)\.pt$")
CHECKPOINT_TAGS = {"init": 1, "mid": 2, "final": 3}
FCP_S1_SPARSE_HISTORY_FILENAME = "fcp_s1_sparse_history.csv"


def _version(path: Path) -> int:
    match = VERSION_RE.search(path.name)
    return int(match.group(1)) if match else -1


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


def _load_policy_args(run_dir: Path):
    for config_path in (run_dir / "policy_config.pkl", run_dir / "files" / "policy_config.pkl"):
        if config_path.exists():
            try:
                with config_path.open("rb") as file:
                    return pickle.load(file)[0]
            except Exception as exc:  # noqa: BLE001
                print(f"warning: failed reading {config_path}: {exc}")
    return None


def _periodic_actors(run_dir: Path) -> list[Path]:
    by_version = {}
    for path in run_dir.rglob("actor_periodic_*.pt"):
        version = _version(path)
        if version < 0:
            continue
        current = by_version.get(version)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            by_version[version] = path
    return [by_version[v] for v in sorted(by_version)]


def _dedupe_history(events: list[tuple[int, float]]) -> list[tuple[int, float]]:
    deduped: dict[int, float] = {}
    for step, value in events:
        deduped[step] = 0.0 if math.isnan(value) else value
    return sorted(deduped.items())


def _load_csv_sparse_history(run_dir: Path) -> list[tuple[int, float]]:
    """Episode sparse-reward curve from FCP S1's local canonical CSV."""
    history_path = run_dir / FCP_S1_SPARSE_HISTORY_FILENAME
    if not history_path.exists():
        return []

    events: list[tuple[int, float]] = []
    try:
        with history_path.open(newline="") as file:
            for row in csv.DictReader(file):
                try:
                    step = int(float(row["step"]))
                    value = float(row["ep_sparse_r"])
                except (KeyError, TypeError, ValueError):
                    continue
                events.append((step, value))
    except OSError as exc:
        print(f"warning: failed reading {history_path}: {exc}")
        return []
    return _dedupe_history(events)


def _load_tensorboard_sparse_history(run_dir: Path) -> list[tuple[int, float]]:
    """Episode sparse-reward curve for one self-play seed from TensorBoard."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:  # noqa: BLE001
        return []

    events: list[tuple[int, float]] = []
    for event_file in sorted(run_dir.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime):
        try:
            acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            acc.Reload()
            scalar_tags = acc.Tags().get("scalars", [])
            # Prefer the overall episode sparse reward; ignore per-agent splits.
            tag = None
            if "ep_sparse_r" in scalar_tags:
                tag = "ep_sparse_r"
            else:
                for candidate in scalar_tags:
                    if candidate.endswith("ep_sparse_r") and "by_agent" not in candidate:
                        tag = candidate
                        break
            if tag is None:
                continue
            events.extend((int(e.step), float(e.value)) for e in acc.Scalars(tag))
        except Exception:  # noqa: BLE001
            continue
    return _dedupe_history(events)


def _load_sparse_history(run_dir: Path) -> list[tuple[int, float]]:
    """Load FCP S1 ep_sparse_r history, preferring local CSV over legacy TensorBoard."""
    csv_history = _load_csv_sparse_history(run_dir)
    if csv_history:
        return csv_history
    return _load_tensorboard_sparse_history(run_dir)


def _nearest_actor(actors: list[Path], target_version: int) -> Path:
    return min(actors, key=lambda p: (abs(_version(p) - target_version), _version(p)))


def _interpolate_history(history: list[tuple[int, float]],
                         interval: int = 100) -> list[tuple[int, float]]:
    """Linearly densify the (step, reward) curve, as HSP's extract_sp does.

    HSP's ``extract_sp_S1_models.py`` resamples the reward curve at ``interval``
    (=100) step spacing before locating the half-final-reward point, so the
    "mid" checkpoint is chosen against a smooth curve rather than only the few
    logged points. We replicate that, but scale ``interval`` up for very wide
    step ranges so the dense curve stays bounded (~200k points max).
    """
    if len(history) < 2:
        return history
    span = history[-1][0] - history[0][0]
    if span > 0:
        interval = max(interval, span // 200_000 + 1)
    dense = [history[0]]
    for (l_s, l_er), (s, er) in zip(history, history[1:]):
        if s <= l_s:
            continue
        for w in range(l_s + 1, s, interval):
            dense.append((w, l_er + (er - l_er) * (w - l_s) / (s - l_s)))
        dense.append((s, er))
    return dense


def _select_mid(actors: list[Path], history: list[tuple[int, float]]) -> Path:
    if not history:
        return actors[len(actors) // 2]
    # Half of the final episode sparse reward, located on the interpolated curve.
    final_sparse_r = history[-1][1]
    mid_sparse_r = final_sparse_r / 2.0
    dense = _interpolate_history(history)
    target_step = min(dense, key=lambda item: abs(item[1] - mid_sparse_r))[0]
    max_step = max(step for step, _ in dense)
    if max_step <= 0:
        return actors[len(actors) // 2]
    max_version = max(_version(a) for a in actors)
    estimated = int(target_step / max_step * (max_version + 1))
    return _nearest_actor(actors, estimated)


def _select_checkpoints(actors: list[Path], history: list[tuple[int, float]]) -> dict[str, Path]:
    init_actor = next((a for a in actors if _version(a) == 0), actors[0])
    return {"init": init_actor, "mid": _select_mid(actors, history), "final": actors[-1]}


def _mlp_partner_entry(layout: str, seed: int, tag: str) -> dict:
    return {
        "policy_config_path": f"{layout}/policy_config/mlp_policy_config.pkl",
        "featurize_type": "ppo",
        "train": False,
        "model_path": {"actor": f"{layout}/fcp/s1/sp{seed}_{tag}_actor.pt"},
    }


def extract(layout: str, results_root: Path, policy_pool_root: Path,
            adaptive_agent_name: str, require_seeds: int | None,
            require_reward_history: bool = False) -> None:
    search_root = results_root / RISKY_ENV_NAME / layout / "mappo" / "fcp-S1"
    dest_s1 = policy_pool_root / layout / "fcp" / "s1"
    dest_s2 = policy_pool_root / layout / "fcp" / "s2"

    if not search_root.exists():
        raise FileNotFoundError(f"no FCP S1 results under {search_root}")

    # Group discovered run dirs by their training seed.
    by_seed: dict[int, Path] = {}
    for fallback_idx, run_dir in enumerate(_run_dirs(search_root), start=1):
        actors = _periodic_actors(run_dir)
        if not actors:
            continue
        all_args = _load_policy_args(run_dir)
        seed = int(getattr(all_args, "seed", fallback_idx)) if all_args is not None else fallback_idx
        # Keep the most recent run dir for a given seed if duplicated.
        prev = by_seed.get(seed)
        if prev is None or run_dir.stat().st_mtime > prev.stat().st_mtime:
            by_seed[seed] = run_dir

    if not by_seed:
        raise FileNotFoundError(
            f"no actor_periodic_*.pt checkpoints found under {search_root}. "
            "Stage 1 must run as SHARED self-play (do not pass --share_policy); "
            "the separated runner would save actor_agent{0,1}_periodic_*.pt instead."
        )

    if require_seeds is not None:
        expected = set(range(1, require_seeds + 1))
        missing = sorted(expected - set(by_seed))
        if missing:
            raise SystemExit(f"error: missing FCP S1 seeds {missing} under {search_root}")

    eval_yaml: dict[str, dict] = {}
    train_yaml: dict[str, dict] = {
        adaptive_agent_name: {
            "policy_config_path": f"{layout}/policy_config/rnn_policy_config.pkl",
            "featurize_type": "ppo",
            "train": True,
        }
    }

    for seed in sorted(by_seed):
        run_dir = by_seed[seed]
        actors = _periodic_actors(run_dir)
        history = _load_sparse_history(run_dir)
        if not history:
            message = (
                f"seed {seed}: no local fcp_s1_sparse_history.csv or TensorBoard "
                f"ep_sparse_r history under {run_dir}. The 'mid' checkpoint would fall "
                "back to the temporal-middle checkpoint, NOT the half-final-reward "
                "criterion. Run Stage 1 with a current FCP transplant so the local CSV "
                "records ep_sparse_r for the exact HSP-FCP selection."
            )
            if require_reward_history:
                raise SystemExit(f"error: {message}")
            print(f"WARNING {message}")
        selected = _select_checkpoints(actors, history)
        print(
            f"seed {seed}: {run_dir.name} "
            f"versions={{init:{_version(selected['init'])}, "
            f"mid:{_version(selected['mid'])}, final:{_version(selected['final'])}}}"
        )
        for tag, src in selected.items():
            dst = dest_s1 / f"sp{seed}_{tag}_actor.pt"
            _copy(src, dst)
            # Numeric partner keys sp{seed}_{1,2,3} match HSP's FCP pool naming;
            # the checkpoint FILE keeps the descriptive sp{seed}_{init,mid,final}
            # name (as in HSP's extract_sp_S1_models.py). Use the same numeric key
            # in both eval.yml and train.yml so the two pools stay consistent.
            partner_key = f"sp{seed}_{CHECKPOINT_TAGS[tag]}"
            eval_yaml[partner_key] = _mlp_partner_entry(layout, seed, tag)
            train_yaml[partner_key] = _mlp_partner_entry(layout, seed, tag)

    dest_s1.mkdir(parents=True, exist_ok=True)
    dest_s2.mkdir(parents=True, exist_ok=True)
    with (dest_s1 / "eval.yml").open("w") as file:
        yaml.safe_dump(eval_yaml, file, sort_keys=False)
    print(f"wrote {dest_s1 / 'eval.yml'}  ({len(eval_yaml)} partners)")
    with (dest_s2 / "train.yml").open("w") as file:
        yaml.safe_dump(train_yaml, file, sort_keys=False)
    partner_count = len(train_yaml) - 1
    print(f"wrote {dest_s2 / 'train.yml'}  (1 adaptive + {partner_count} frozen partners)")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", default="risky_dualpath_subgoal")
    parser.add_argument("--results-root", type=Path, default=OUTPUT_ROOT / "results")
    parser.add_argument("--policy-pool", type=Path, default=POLICY_POOL_ROOT)
    parser.add_argument("--adaptive-agent-name", default="fcp_adaptive")
    parser.add_argument("--require-seeds", type=int, default=None,
                        help="Fail unless seeds 1..N are all present (HSP FCP: 12).")
    parser.add_argument("--require-reward-history", action="store_true",
                        help="Fail (instead of temporal-middle fallback) if a seed has no "
                             "local CSV or TensorBoard ep_sparse_r history for the "
                             "half-final-reward 'mid' criterion.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    extract(args.layout, args.results_root, args.policy_pool,
            args.adaptive_agent_name, args.require_seeds, args.require_reward_history)


if __name__ == "__main__":
    main()
