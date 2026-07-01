#!/usr/bin/env python
"""Select diverse HSP S1 policies and create HSP S2 train YAML."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transplant.adapters.risky_overcooked_env import HIDDEN_UTILITY_KEYS
from transplant.bootstrap import POLICY_POOL_ROOT, TRANSPLANT_ROOT, ensure_paths

ensure_paths()


KEY_RE = re.compile(r"hsp(\d+)_w([01])-hsp\1_w([01])-eval_ep_(.+)_by_agent([01])")
MEP_POLICY_COUNT = 6
MEP_CHECKPOINT_TAGS = (1, 2, 3)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as file:
        return yaml.safe_load(file) or {}


def _parse_event_logs(eval_dir: Path) -> dict[int, np.ndarray]:
    accum = defaultdict(lambda: np.zeros(len(HIDDEN_UTILITY_KEYS), dtype=np.float32))
    counts = defaultdict(int)
    event_index = {key: idx for idx, key in enumerate(HIDDEN_UTILITY_KEYS)}

    for logfile in sorted(eval_dir.glob("eval*.txt")):
        for raw_line in logfile.read_text().splitlines():
            line = raw_line.strip()
            if not line.startswith("{"):
                continue
            try:
                metrics = eval(line, {"np": np, "__builtins__": {}})
            except Exception:
                continue
            for key, value in metrics.items():
                match = KEY_RE.match(key)
                if match is None:
                    continue
                run_id = int(match.group(1))
                first_w = match.group(2)
                event_name = match.group(4)
                agent_idx = match.group(5)
                if event_name not in event_index:
                    continue
                if first_w == "0" and agent_idx != "0":
                    continue
                if first_w == "1" and agent_idx != "1":
                    continue
                scalar = value[0] if isinstance(value, list) else value
                accum[run_id][event_index[event_name]] += float(scalar)
                counts[run_id] += 1

    return {run_id: vector / max(counts[run_id], 1) for run_id, vector in accum.items()}


def _select_diverse(run_vectors: dict[int, np.ndarray], k: int, seed: int) -> list[int]:
    runs = sorted(run_vectors)
    if len(runs) <= k:
        return runs
    rng = np.random.default_rng(seed)
    matrix = np.stack([run_vectors[run] for run in runs])
    matrix = matrix / (matrix.max(axis=0, keepdims=True) + 1e-3)
    selected = [int(rng.integers(0, len(runs)))]
    while len(selected) < k:
        scores = np.full(len(runs), -1e9, dtype=np.float32)
        for idx in range(len(runs)):
            if idx in selected:
                continue
            scores[idx] = sum(np.abs(matrix[idx] - matrix[j]).sum() for j in selected)
        selected.append(int(scores.argmax()))
    return sorted(runs[idx] for idx in selected)


def _validated_mep_entries(mep_eval: dict) -> dict:
    expected = [f"mep{policy_idx}_{tag}" for policy_idx in range(1, MEP_POLICY_COUNT + 1) for tag in MEP_CHECKPOINT_TAGS]
    missing = [name for name in expected if name not in mep_eval]
    if missing:
        raise SystemExit(
            "error: incomplete MEP S1 policy pool. Missing entries: "
            + ", ".join(missing)
            + ". Re-run 02_extract_mep_s1_risky.py."
        )

    filtered = {}
    for name in expected:
        entry = dict(mep_eval[name])
        if "model_path" not in entry or "actor" not in entry["model_path"]:
            raise SystemExit(f"error: MEP entry {name} does not define model_path.actor")
        entry["train"] = False
        filtered[name] = entry
    return filtered


def write_hsp_s2_yaml(layout: str, policy_pool_root: Path, selected_runs: list[int]) -> Path:
    layout_root = policy_pool_root / layout
    mep_eval = _load_yaml(layout_root / "mep/s1/eval.yml")
    mep_entries = _validated_mep_entries(mep_eval)
    data = {
        "hsp_adaptive": {
            "policy_config_path": f"{layout}/policy_config/rnn_policy_config.pkl",
            "featurize_type": "ppo",
            "train": True,
        }
    }
    data.update(mep_entries)
    for run_id in selected_runs:
        data[f"hsp{run_id}"] = {
            "policy_config_path": f"{layout}/policy_config/mlp_policy_config.pkl",
            "featurize_type": "ppo",
            "train": False,
            "model_path": {"actor": f"{layout}/hsp/s1/hsp{run_id}_w0_actor.pt"},
        }
    output_path = layout_root / "hsp/s2/train.yml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        yaml.safe_dump(data, file, sort_keys=False)
    return output_path


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="risky_dualpath_subgoal")
    parser.add_argument("--k", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--eval-result-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--policy-pool", type=Path, default=POLICY_POOL_ROOT)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.eval_result_dir is None:
        args.eval_result_dir = TRANSPLANT_ROOT / "biased_eval" / args.layout
    run_vectors = _parse_event_logs(args.eval_result_dir)
    if not run_vectors:
        raise SystemExit(
            "error: no event logs found for greedy selection. "
            "Run 05_eval_events_risky.sh with EVAL_EPISODES >= N_EVAL_ROLLOUT_THREADS first."
        )
    if len(run_vectors) < args.k:
        raise SystemExit(
            f"error: only {len(run_vectors)} event vectors found, but k={args.k}. "
            "Re-run event evaluation for all HSP S1 candidates before selecting policies."
        )
    selected = _select_diverse(run_vectors, args.k, args.seed)
    if len(selected) != args.k:
        raise SystemExit(f"error: expected {args.k} HSP policies, selected {len(selected)}")
    output_path = write_hsp_s2_yaml(args.layout, args.policy_pool, selected)
    print("selected runs:", selected)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
