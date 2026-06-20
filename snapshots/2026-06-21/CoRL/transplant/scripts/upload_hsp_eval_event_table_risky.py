#!/usr/bin/env python
"""Upload compact HSP final-eval event tables to W&B."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transplant.adapters.risky_overcooked_env import CATEGORY_KEYS
from transplant.bootstrap import TRANSPLANT_ROOT, ensure_paths
from transplant.common import RISKY_ENV_NAME


ensure_paths()


def _natural_partner_key(name: str) -> tuple[int, int, int]:
    mep_match = re.fullmatch(r"mep(\d+)_(\d+)", name)
    if mep_match:
        return (0, int(mep_match.group(1)), int(mep_match.group(2)))
    hsp_match = re.fullmatch(r"hsp(\d+)", name)
    if hsp_match:
        return (1, int(hsp_match.group(1)), 0)
    return (2, 0, 0)


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["count"] = float(row["count"])
    return rows


def _read_gif_rows(path: Path | None) -> list[dict]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _read_score_rows(path: Path | None) -> list[dict]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["episode_sparse_score"] = float(row["episode_sparse_score"])
        row["agent0_sparse"] = float(row["agent0_sparse"]) if row["agent0_sparse"] else None
        row["agent1_sparse"] = float(row["agent1_sparse"]) if row["agent1_sparse"] else None
    return rows


def _event_order(rows: list[dict]) -> list[str]:
    present = {row["event"] for row in rows}
    ordered = [event for event in CATEGORY_KEYS if event in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _partner_order(rows: list[dict]) -> list[str]:
    return sorted({row["partner"] for row in rows}, key=_natural_partner_key)


def _mean(values: list[float]) -> float | str:
    if not values:
        return ""
    return sum(values) / len(values)


def _write_matrix(rows: list[dict], partners: list[str], events: list[str], output_path: Path, agent_filter: str | None) -> None:
    values = defaultdict(list)
    for row in rows:
        if agent_filter is not None and row["adaptive_agent"] != agent_filter:
            continue
        values[(row["event"], row["partner"])].append(row["count"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["event", *partners])
        for event in events:
            writer.writerow([event, *[_mean(values[(event, partner)]) for partner in partners]])


def _read_matrix(path: Path) -> tuple[list[str], list[list]]:
    with path.open(newline="") as file:
        reader = csv.reader(file)
        columns = next(reader)
        rows = []
        for row in reader:
            parsed = [row[0]]
            for value in row[1:]:
                parsed.append(float(value) if value != "" else None)
            rows.append(parsed)
    return columns, rows


def _write_heatmap(matrix_path: Path, output_path: Path) -> Path | None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"warning: heatmap export skipped ({exc})")
        return None

    columns, rows = _read_matrix(matrix_path)
    partners = columns[1:]
    events = [row[0] for row in rows]
    values = np.asarray([[0.0 if value is None else float(value) for value in row[1:]] for row in rows])

    width = max(12.0, len(partners) * 0.45)
    height = max(8.0, len(events) * 0.26)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xlabel("Partner policy")
    ax.set_ylabel("Adaptive policy event")
    ax.set_xticks(range(len(partners)))
    ax.set_xticklabels(partners, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(events)))
    ax.set_yticklabels(events, fontsize=8)
    ax.set_title("HSP adaptive eval event counts")
    fig.colorbar(image, ax=ax, label="Count")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _write_matrices(rows: list[dict], output_dir: Path) -> dict[str, Path]:
    partners = _partner_order(rows)
    events = _event_order(rows)
    outputs = {
        "mean": output_dir / "hsp_adaptive_event_counts_matrix_mean.csv",
        "adaptive_agent0": output_dir / "hsp_adaptive_event_counts_matrix_agent0.csv",
        "adaptive_agent1": output_dir / "hsp_adaptive_event_counts_matrix_agent1.csv",
    }
    _write_matrix(rows, partners, events, outputs["mean"], agent_filter=None)
    _write_matrix(rows, partners, events, outputs["adaptive_agent0"], agent_filter="agent0")
    _write_matrix(rows, partners, events, outputs["adaptive_agent1"], agent_filter="agent1")
    return outputs


def _score_mean(score_rows: list[dict], agent_filter: str | None) -> float | None:
    values = [
        row["episode_sparse_score"]
        for row in score_rows
        if agent_filter is None or row["adaptive_agent"] == agent_filter
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _write_score_summary(score_rows: list[dict], output_path: Path) -> Path | None:
    if not score_rows:
        return None
    values = {
        "overall_average": _score_mean(score_rows, None),
        "agent0_average": _score_mean(score_rows, "agent0"),
        "agent1_average": _score_mean(score_rows, "agent1"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(values))
        writer.writeheader()
        writer.writerow(values)
    return output_path


def _wandb_key_component(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)
    return value.strip("_") or "unknown"


def _gif_panel_name(row: dict) -> str | None:
    agent0 = row.get("agent0_policy", "")
    agent1 = row.get("agent1_policy", "")
    adaptive_agent = row.get("adaptive_agent", "")
    if adaptive_agent == "agent0":
        partner_agent = "agent1"
    elif adaptive_agent == "agent1":
        partner_agent = "agent0"
    elif agent0 == "hsp_adaptive":
        adaptive_agent = "agent0"
        partner_agent = "agent1"
    elif agent1 == "hsp_adaptive":
        adaptive_agent = "agent1"
        partner_agent = "agent0"
    else:
        return None

    return "__".join(
        [
            f"adaptive_{_wandb_key_component(adaptive_agent)}",
            f"partner_{_wandb_key_component(partner_agent)}",
            f"agent0_{_wandb_key_component(agent0)}",
            f"agent1_{_wandb_key_component(agent1)}",
        ]
    )


def _make_gif_panel_payload(gif_rows: list[dict], wandb) -> dict:
    payload = {}
    for row in gif_rows:
        partner = row.get("partner", "")
        gif_path = Path(row.get("gif_path", ""))
        if not partner or not gif_path.is_file():
            continue
        panel_name = _gif_panel_name(row)
        if panel_name is None:
            continue
        payload[f"final_eval_gifs/{panel_name}"] = wandb.Video(str(gif_path), format="gif")
    return payload


def _upload_to_wandb(
    args,
    rows: list[dict],
    matrix_paths: dict[str, Path],
    score_rows: list[dict],
    score_summary_path: Path | None,
    heatmap_path: Path | None,
    gif_rows: list[dict],
) -> None:
    import wandb

    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name,
        "group": args.wandb_group or f"HSP_{args.layout}",
        "job_type": "evaluation_table",
        "mode": args.wandb_mode,
        "config": {
            "layout": args.layout,
            "input_csv": str(args.input),
            "num_rows": len(rows),
            "num_partners": len(_partner_order(rows)),
            "num_events": len(_event_order(rows)),
            "score_input": str(args.score_input) if args.score_input is not None else None,
            "num_score_rows": len(score_rows),
            "gif_manifest": str(args.gif_manifest) if args.gif_manifest is not None else None,
            "num_gif_rows": len(gif_rows),
            "skip_gif_upload": args.skip_gif_upload,
        },
        "settings": wandb.Settings(
            start_method=os.environ.get("WANDB_START_METHOD", "thread"),
            x_disable_stats=True,
            x_disable_machine_info=True,
        ),
    }
    if args.wandb_run_id:
        init_kwargs["id"] = args.wandb_run_id
        init_kwargs["resume"] = "allow"

    run = wandb.init(**init_kwargs)

    long_columns = list(rows[0])
    run.log(
        {
            "final_eval_tables/adaptive_events_long": wandb.Table(
                columns=long_columns,
                data=[[row[column] for column in long_columns] for row in rows],
            )
        }
    )

    tables = {}
    for name, path in matrix_paths.items():
        columns, data = _read_matrix(path)
        tables[f"final_eval_tables/{name}_event_by_partner"] = wandb.Table(columns=columns, data=data)
    run.log(tables)
    score_payload = {
        "final_eval_adaptive_summary/overall_average": _score_mean(score_rows, None),
        "final_eval_adaptive_summary/agent0_average": _score_mean(score_rows, "agent0"),
        "final_eval_adaptive_summary/agent1_average": _score_mean(score_rows, "agent1"),
    }
    score_payload = {key: value for key, value in score_payload.items() if value is not None}
    if score_payload:
        run.log(score_payload)
    if heatmap_path is not None:
        run.log({"final_eval_tables/mean_event_heatmap": wandb.Image(str(heatmap_path))})
    if not args.skip_gif_upload:
        gif_payload = _make_gif_panel_payload(gif_rows, wandb)
        if gif_payload:
            run.log(gif_payload)

    artifact = wandb.Artifact(
        name=f"hsp-final-eval-event-counts-{args.layout}",
        type="evaluation",
        metadata={"layout": args.layout},
    )
    artifact.add_file(str(args.input), name=args.input.name)
    for path in matrix_paths.values():
        artifact.add_file(str(path), name=path.name)
    if args.score_input is not None and args.score_input.exists():
        artifact.add_file(str(args.score_input), name=args.score_input.name)
    if score_summary_path is not None:
        artifact.add_file(str(score_summary_path), name=score_summary_path.name)
    if heatmap_path is not None:
        artifact.add_file(str(heatmap_path), name=heatmap_path.name)
    if args.gif_manifest is not None and args.gif_manifest.exists():
        artifact.add_file(str(args.gif_manifest), name=args.gif_manifest.name)
    run.log_artifact(artifact)
    run.finish()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=TRANSPLANT_ROOT / "final_eval" / "hsp_adaptive_event_counts_long.csv")
    parser.add_argument("--score-input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=TRANSPLANT_ROOT / "final_eval")
    parser.add_argument("--gif-manifest", type=Path, default=None)
    parser.add_argument("--layout", default="risky_multipath_subgoal")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get(
            "WANDB_ENTITY",
            "dhwngus41-daegu-gyeongbuk-institute-of-science-technology",
        ),
    )
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-run-id", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--skip-gif-upload", action="store_true")
    args = parser.parse_args(argv)
    if args.wandb_run_name is None:
        args.wandb_run_name = f"09_final_eval_HSP_{args.layout}"
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.wandb_project:
        args.wandb_project = os.environ.get("WANDB_PROJECT") or f"{RISKY_ENV_NAME}_{args.layout}"
    rows = _read_rows(args.input)
    if not rows:
        raise SystemExit(f"error: no rows found in {args.input}")
    score_rows = _read_score_rows(args.score_input)
    gif_rows = _read_gif_rows(args.gif_manifest)
    matrix_paths = _write_matrices(rows, args.output_dir)
    score_summary_path = _write_score_summary(
        score_rows,
        args.output_dir / "hsp_adaptive_sparse_score_summary.csv",
    )
    heatmap_path = _write_heatmap(
        matrix_paths["mean"],
        args.output_dir / "hsp_adaptive_event_counts_heatmap_mean.png",
    )
    print(f"wrote long event table: {args.input}")
    for name, path in matrix_paths.items():
        print(f"wrote {name} matrix: {path}")
    if args.score_input is not None:
        print(f"read {len(score_rows)} sparse score rows: {args.score_input}")
    if score_summary_path is not None:
        print(f"wrote sparse score summary: {score_summary_path}")
    if heatmap_path is not None:
        print(f"wrote mean heatmap: {heatmap_path}")
    if args.gif_manifest is not None:
        print(f"read {len(gif_rows)} gif manifest rows: {args.gif_manifest}")
    if args.use_wandb:
        _upload_to_wandb(
            args,
            rows,
            matrix_paths,
            score_rows,
            score_summary_path,
            heatmap_path,
            gif_rows,
        )


if __name__ == "__main__":
    main()
