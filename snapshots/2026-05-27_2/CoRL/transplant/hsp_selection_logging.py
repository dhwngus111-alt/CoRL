"""Logging helpers for fixed HSP Stage 1 hidden utility weights."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from transplant.hsp_hidden_utility import (
    HIDDEN_UTILITY_KEYS,
    RISKY_MULTIPATH_EVENT_KEYS,
)


SELECTION_COLUMNS = (
    "index",
    "key",
    "group",
    "w0",
    "w1",
    "w0_nonzero",
    "w1_nonzero",
)


def _parse_weight_values(weight: str | Iterable[float], name: str) -> list[float]:
    expected_len = len(HIDDEN_UTILITY_KEYS) + 1
    if isinstance(weight, str):
        values = [float(item.strip()) for item in weight.split(",") if item.strip()]
    else:
        values = [float(item) for item in weight]

    if len(values) != expected_len:
        raise ValueError(
            f"{name} must have {expected_len} values "
            f"({len(HIDDEN_UTILITY_KEYS)} hidden utility weights + sparse weight), "
            f"got {len(values)}"
        )
    return values


def _selection_group(key: str) -> str:
    if key == "sparse":
        return "sparse"
    if key in RISKY_MULTIPATH_EVENT_KEYS:
        return "risky_multipath"
    return "hsp_core"


def hidden_utility_selection_rows(w0: str | Iterable[float], w1: str | Iterable[float]) -> list[dict]:
    w0_values = _parse_weight_values(w0, "w0")
    w1_values = _parse_weight_values(w1, "w1")
    keys = list(HIDDEN_UTILITY_KEYS) + ["sparse"]

    rows = []
    for idx, key in enumerate(keys):
        w0_value = float(w0_values[idx])
        w1_value = float(w1_values[idx])
        rows.append(
            {
                "index": idx,
                "key": key,
                "group": _selection_group(key),
                "w0": w0_value,
                "w1": w1_value,
                "w0_nonzero": bool(w0_value != 0.0),
                "w1_nonzero": bool(w1_value != 0.0),
            }
        )
    return rows


def write_hidden_utility_selection_csv(rows: list[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SELECTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def log_hidden_utility_selection(
    all_args,
    run_dir: str | Path,
    prefix: str,
    use_wandb: bool,
) -> Path:
    rows = hidden_utility_selection_rows(all_args.w0, all_args.w1)
    csv_path = write_hidden_utility_selection_csv(
        rows,
        Path(run_dir) / "hidden_utility_selection.csv",
    )

    if use_wandb:
        try:
            import wandb

            table = wandb.Table(
                columns=list(SELECTION_COLUMNS),
                data=[[row[column] for column in SELECTION_COLUMNS] for row in rows],
            )
            wandb.log({f"{prefix}/hidden_utility_selection": table})
        except Exception as exc:
            print(f"warning: failed to log hidden utility selection table to wandb: {exc}")

    return csv_path
