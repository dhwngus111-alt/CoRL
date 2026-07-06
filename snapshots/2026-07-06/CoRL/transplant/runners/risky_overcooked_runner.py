"""Risky-native logging wrapper around HSP shared runner.

Structured WandB logging with stage-based sections:
  - stage1_hsp/   : Stage 1 HSP biased policy training
  - stage1_mep/   : Stage 1 MEP population training
  - stage2/       : Stage 2 adaptive policy training against population

All event keys follow risky repo's EVENT_TYPES naming convention.
"""

from __future__ import annotations

import csv
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import wandb

from hsp.runner.shared.overcooked_runner import OvercookedRunner as HSPSharedRunner, _t2n
from transplant.adapters.risky_overcooked_env import CATEGORY_KEYS
from transplant.hsp_selection_logging import log_hidden_utility_selection


STAGE2_TRAIN_METRICS = (
    "value_loss",
    "policy_loss",
    "dist_entropy",
    "actor_grad_norm",
    "critic_grad_norm",
    "ratio",
    "average_episode_rewards",
)
STAGE1_MEP_TRAIN_METRICS = (
    "value_loss",
    "policy_loss",
    "average_episode_rewards",
)
STAGE1_MEP_ENV_REWARD_METRICS = (
    "ep_sparse_r",
    "ep_shaped_r",
)
STAGE1_HSP_TRAIN_METRICS = (
    "average_episode_rewards",
)
STAGE1_HSP_ENV_REWARD_METRICS = {
    "ep_sparse_r",
    "ep_shaped_r",
    "ep_sparse_r_by_agent0",
    "ep_sparse_r_by_agent1",
    "ep_shaped_r_by_agent0",
    "ep_shaped_r_by_agent1",
}
FCP_S1_SPARSE_HISTORY_FILENAME = "fcp_s1_sparse_history.csv"
FCP_S1_SPARSE_HISTORY_FIELDS = (
    "step",
    "ep_sparse_r",
    "ep_sparse_r_by_agent0",
    "ep_sparse_r_by_agent1",
    "ep_shaped_r",
    "ep_shaped_r_by_agent0",
    "ep_shaped_r_by_agent1",
)
STAGE2_SAMPLING_FIELDS = (
    "step",
    "pair",
    "partner",
    "partner_group",
    "direction",
    "sampling_prob",
    "matchup_count",
)
STAGE2_REWARD_SHAPING_FIELDS = (
    "reward_shaping_factor",
    "reward_shaping_progress",
    "reward_shaping_anneal_step",
    "reward_shaping_horizon",
)
STAGE2_TRAIN_ENV_METRICS = {
    "ep_sparse_r",
    "adaptive_ep_sparse_r",
    "adaptive_ep_shaped_r",
}
STAGE2_TRAIN_FIELDS = ("step", *STAGE2_TRAIN_METRICS, *STAGE2_REWARD_SHAPING_FIELDS)


class RiskyOvercookedRunner(HSPSharedRunner):
    """Use HSP training loops while reporting Risky EVENT_TYPES directly."""

    def __init__(self, config):
        local_run_dir = config.get("local_run_dir")
        if local_run_dir is not None and getattr(config.get("all_args"), "use_wandb", False):
            run = getattr(wandb, "run", None)
            run_dir = getattr(run, "dir", None)
            if run_dir:
                Path(run_dir).mkdir(parents=True, exist_ok=True)

        super().__init__(config)
        self.final_render_envs = config.get("final_render_envs")
        if local_run_dir is not None:
            self._use_local_run_dir(Path(local_run_dir))

        self.risky_env_name = self.env_name
        # HSP loops guard sparse/shaped reward bookkeeping on this field.
        self.env_name = "Overcooked"

        # ── Stage / Section detection ─────────────────────────────────
        self._wandb_section = self._detect_section()

        # ── Matchup tracking (Stage 2) ────────────────────────────────
        self._matchup_counter: Counter = Counter()
        self._stage2_latest_sampling: dict[str, float] = {}
        self._stage2_log_raw_scalars = self._truthy_env("STAGE2_LOG_RAW_SCALARS")
        self._stage2_csv_paths = self._init_stage2_csv_paths()
        self._stage1_mep_defined_policies: set[str] = set()
        self._stage1_mep_final_gifs_logged = False
        self._stage1_hsp_metrics_defined = False
        self._stage1_hsp_selection_logged = False
        self._stage1_hsp_final_gifs_logged = False

    def _use_local_run_dir(self, local_run_dir: Path) -> None:
        local_run_dir.mkdir(parents=True, exist_ok=True)
        current_policy_config = Path(self.run_dir) / "policy_config.pkl"
        if current_policy_config.exists():
            shutil.copy2(current_policy_config, local_run_dir / "policy_config.pkl")
        self.run_dir = str(local_run_dir)
        self.save_dir = str(local_run_dir)

    # ══════════════════════════════════════════════════════════════════
    #  Stage Detection
    # ══════════════════════════════════════════════════════════════════

    def _detect_section(self) -> str:
        """Detect current training stage from args and return section prefix."""
        algo = getattr(self.all_args, "algorithm_name", "")
        stage = getattr(self.all_args, "stage", 1)

        if algo in ("mep", "adaptive"):
            if stage == 2:
                return "stage2"
            else:
                return "stage1_mep"
        else:
            # mappo / rmappo / mappg → HSP S1 training
            return "stage1_hsp"

    @property
    def wandb_section(self) -> str:
        return self._wandb_section

    @staticmethod
    def _truthy_env(name: str) -> bool:
        return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _as_scalar(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return None
            return float(np.mean(value))
        if isinstance(value, (int, float, np.number)):
            return float(value)
        return None

    @staticmethod
    def _partner_group(partner: str) -> str:
        if partner.startswith("mep"):
            return "mep"
        if partner.startswith("hsp"):
            return "hsp"
        return "other"

    @property
    def _adaptive_policy_name(self) -> str:
        return str(getattr(self.all_args, "adaptive_agent_name", "hsp_adaptive") or "hsp_adaptive")

    def _direct_adaptive_pair(self, pair: str):
        if pair.startswith(("agent0-", "agent1-", "either-")):
            return None
        adaptive_name = self._adaptive_policy_name
        left_prefix = f"{adaptive_name}-"
        right_suffix = f"-{adaptive_name}"
        if pair.startswith(left_prefix):
            partner = pair[len(left_prefix) :]
            if partner:
                return adaptive_name, partner, 0, partner
        if pair.endswith(right_suffix):
            partner = pair[: -len(right_suffix)]
            if partner:
                return partner, adaptive_name, 1, partner
        return None

    def _init_stage2_csv_paths(self) -> dict[str, Path]:
        root = Path(self.run_dir)
        return {
            "train": root / "stage2_train_summary.csv",
        }

    def _write_csv_rows(self, name: str, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
        if not rows:
            return
        path = self._stage2_csv_paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _stage2_sampling_rows(self, total_num_steps: int) -> list[dict]:
        rows = []
        for pair in sorted(set(self._stage2_latest_sampling) | set(self._matchup_counter)):
            parsed = self._direct_adaptive_pair(pair)
            if parsed is None:
                continue
            agent0, agent1, _, partner = parsed
            rows.append(
                {
                    "step": total_num_steps,
                    "pair": pair,
                    "partner": partner,
                    "partner_group": self._partner_group(partner),
                    "direction": f"{agent0}_vs_{agent1}",
                    "sampling_prob": self._stage2_latest_sampling.get(pair, ""),
                    "matchup_count": self._matchup_counter.get(pair, 0),
                }
            )
        return rows

    def _stage2_train_summary_row(self, train_infos: dict, total_num_steps: int) -> dict:
        row = {"step": total_num_steps}
        adaptive_name = self._adaptive_policy_name
        for metric in STAGE2_TRAIN_METRICS:
            value = train_infos.get(f"{adaptive_name}-{metric}", train_infos.get(metric))
            scalar = self._as_scalar(value)
            row[metric] = "" if scalar is None else scalar
        row.update(self._stage2_reward_shaping_metrics(total_num_steps))
        return row

    def _stage2_reward_shaping_anneal_step(self, total_num_steps: int) -> float:
        try:
            reward_steps = self.trainer.reward_shaping_steps()
        except Exception:
            return float(total_num_steps)
        scalar = self._as_scalar(reward_steps)
        if scalar is None:
            return float(total_num_steps)
        return scalar

    def _stage2_reward_shaping_metrics(self, total_num_steps: int) -> dict[str, float]:
        horizon = float(getattr(self.all_args, "reward_shaping_horizon", 0) or 0)
        initial = float(getattr(self.all_args, "initial_reward_shaping_factor", 1.0))
        anneal_step = self._stage2_reward_shaping_anneal_step(total_num_steps)
        if horizon <= 0:
            progress = 0.0
            factor = initial
        else:
            progress = min(max(anneal_step / horizon, 0.0), 1.0)
            factor = max(initial * (1.0 - progress), 0.0)
        return {
            "reward_shaping_factor": float(factor),
            "reward_shaping_progress": float(progress),
            "reward_shaping_anneal_step": float(anneal_step),
            "reward_shaping_horizon": float(horizon),
        }

    def _stage2_reward_shaping_payload(self, total_num_steps: int) -> dict[str, float]:
        metrics = self._stage2_reward_shaping_metrics(total_num_steps)
        return {
            "stage2/reward_shaping/factor": metrics["reward_shaping_factor"],
            "stage2/reward_shaping/progress": metrics["reward_shaping_progress"],
            "stage2/reward_shaping/anneal_step": metrics["reward_shaping_anneal_step"],
            "stage2/reward_shaping/horizon": metrics["reward_shaping_horizon"],
        }

    def _log_stage2_reward_shaping(self, total_num_steps: int) -> None:
        if self._wandb_section != "stage2":
            return
        payload = self._stage2_reward_shaping_payload(total_num_steps)
        if self.use_wandb:
            wandb.log(payload, step=int(total_num_steps))
        elif hasattr(self, "writter"):
            for key, value in payload.items():
                self.writter.add_scalars(key, {key: value}, int(total_num_steps))

    def _stage2_final_num_steps(self) -> int:
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        return int(episodes * self.episode_length * self.n_rollout_threads)

    def _stage2_table_payload(self, sampling_rows: list[dict]) -> dict:
        payload = {}
        if sampling_rows:
            payload["stage2_tables/latest_sampling_by_pair"] = wandb.Table(
                columns=list(STAGE2_SAMPLING_FIELDS),
                data=[[row[field] for field in STAGE2_SAMPLING_FIELDS] for row in sampling_rows],
            )
        return payload

    def _stage2_train_env_payload(self, env_infos: dict) -> dict[str, float]:
        payload = {}
        for key in sorted(env_infos):
            metric = key[len("train_") :] if key.startswith("train_") else key
            if metric not in STAGE2_TRAIN_ENV_METRICS:
                continue
            scalar = self._as_scalar(env_infos[key])
            if scalar is None:
                continue
            payload[f"stage2/train_env/{metric}"] = scalar
        return payload

    @staticmethod
    def _stage1_mep_policy_metric(key: str):
        for metric in STAGE1_MEP_TRAIN_METRICS:
            suffix = f"-{metric}"
            if key.endswith(suffix):
                return key[: -len(suffix)], metric
        return None, None

    @staticmethod
    def _stage1_mep_selfplay_reward_metric(key: str):
        for metric in STAGE1_MEP_ENV_REWARD_METRICS:
            suffix = f"-{metric}"
            if not key.endswith(suffix):
                continue
            pair_name = key[: -len(suffix)]
            left, sep, right = pair_name.partition("-")
            if sep and left == right and left.startswith("mep") and left[3:].isdigit():
                return left, metric
        return None, None

    def _stage1_mep_local_step(self, train_infos: dict, policy_name: str, total_num_steps: int) -> int:
        raw_step = train_infos.get(f"{policy_name}-total_num_steps")
        if raw_step is None:
            return int(total_num_steps)
        return int(raw_step) // max(1, int(self.num_agents))

    def _define_stage1_mep_wandb_metrics(self, policy_name: str) -> None:
        if policy_name in self._stage1_mep_defined_policies:
            return
        step_key = f"stage1_mep/train/{policy_name}-local_env_step"
        wandb.define_metric(step_key)
        for metric in STAGE1_MEP_TRAIN_METRICS:
            wandb.define_metric(f"stage1_mep/train/{policy_name}-{metric}", step_metric=step_key)
        self._stage1_mep_defined_policies.add(policy_name)

    @property
    def _stage1_hsp_prefix(self) -> str:
        seed_label = getattr(self.all_args, "hsp_wandb_seed_label", "")
        if seed_label:
            return seed_label
        return "stage1_hsp"

    def _define_stage1_hsp_wandb_metrics(self) -> None:
        if self._stage1_hsp_metrics_defined:
            return
        step_key = f"{self._stage1_hsp_prefix}/local_env_step"
        wandb.define_metric(step_key)
        wandb.define_metric(f"{self._stage1_hsp_prefix}/train/*", step_metric=step_key)
        for key in STAGE1_HSP_TRAIN_METRICS:
            wandb.define_metric(f"{self._stage1_hsp_prefix}/train/{key}", step_metric=step_key)
        wandb.define_metric(f"{self._stage1_hsp_prefix}/env/*", step_metric=step_key)
        for key in STAGE1_HSP_ENV_REWARD_METRICS:
            wandb.define_metric(f"{self._stage1_hsp_prefix}/env/{key}", step_metric=step_key)
        self._stage1_hsp_metrics_defined = True

    def _log_stage1_hsp_hidden_utility_selection(self) -> None:
        if self._stage1_hsp_selection_logged or self._wandb_section != "stage1_hsp":
            return
        log_hidden_utility_selection(
            all_args=self.all_args,
            run_dir=self.run_dir,
            prefix=self._stage1_hsp_prefix,
            use_wandb=self.use_wandb,
        )
        self._stage1_hsp_selection_logged = True

    def _is_fcp_s1_run(self) -> bool:
        return str(getattr(self.all_args, "experiment_name", "")).startswith("fcp-S1")

    def _write_fcp_s1_sparse_history(self, env_infos: dict, total_num_steps: int) -> None:
        if self._wandb_section != "stage1_hsp" or not self._is_fcp_s1_run():
            return

        row = {"step": int(total_num_steps)}
        for key in FCP_S1_SPARSE_HISTORY_FIELDS:
            if key == "step":
                continue
            scalar = self._as_scalar(env_infos.get(key))
            row[key] = "" if scalar is None else scalar

        if row["ep_sparse_r"] == "":
            return

        path = Path(self.run_dir) / FCP_S1_SPARSE_HISTORY_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FCP_S1_SPARSE_HISTORY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # ══════════════════════════════════════════════════════════════════
    #  Override: log_system → no-op (system metrics 불필요)
    # ══════════════════════════════════════════════════════════════════

    def log_system(self):
        """System 메트릭 (메모리/Slack 알림) 제거."""
        pass

    # ══════════════════════════════════════════════════════════════════
    #  Override: log_train / log_env → section prefix 부여
    # ══════════════════════════════════════════════════════════════════

    def log_train(self, train_infos, total_num_steps):
        """Train metrics에 section prefix를 붙여 wandb에 로깅."""
        section = self._wandb_section
        if section == "stage1_mep":
            prefixed = {}
            for key, value in train_infos.items():
                policy_name, _ = self._stage1_mep_policy_metric(key)
                if policy_name is None:
                    continue
                prefixed[f"{section}/train/{policy_name}-local_env_step"] = self._stage1_mep_local_step(
                    train_infos, policy_name, total_num_steps
                )
                prefixed[f"{section}/train/{key}"] = value
            if self.use_wandb and prefixed:
                for key in train_infos:
                    policy_name, _ = self._stage1_mep_policy_metric(key)
                    if policy_name is not None:
                        self._define_stage1_mep_wandb_metrics(policy_name)
                wandb.log(prefixed, step=total_num_steps)
            elif hasattr(self, "writter"):
                for key, value in prefixed.items():
                    self.writter.add_scalars(key, {key: value}, total_num_steps)
            return

        if section == "stage1_hsp":
            prefix = self._stage1_hsp_prefix
            prefixed = {
                f"{prefix}/local_env_step": int(total_num_steps),
                **{
                    f"{prefix}/train/{key}": value
                    for key, value in train_infos.items()
                    if key in STAGE1_HSP_TRAIN_METRICS
                },
            }
            if self.use_wandb and prefixed:
                self._define_stage1_hsp_wandb_metrics()
                if getattr(self.all_args, "hsp_wandb_seed_label", ""):
                    wandb.log(prefixed)
                else:
                    wandb.log(prefixed, step=total_num_steps)
            elif hasattr(self, "writter"):
                for key, value in prefixed.items():
                    self.writter.add_scalars(key, {key: value}, total_num_steps)
            return

        if section == "stage2" and not self._stage2_log_raw_scalars:
            row = self._stage2_train_summary_row(train_infos, total_num_steps)
            self._write_csv_rows("train", STAGE2_TRAIN_FIELDS, [row])
            adaptive_name = self._adaptive_policy_name
            scalars = {
                f"stage2/train/{adaptive_name}/{metric}": row[metric]
                for metric in STAGE2_TRAIN_METRICS
                if row[metric] != ""
            }
            scalars.update(self._stage2_reward_shaping_payload(total_num_steps))
            if self.use_wandb and scalars:
                wandb.log(scalars, step=total_num_steps)
            elif hasattr(self, "writter"):
                for key, value in scalars.items():
                    self.writter.add_scalars(key, {key: value}, total_num_steps)
            return

        prefixed = {}
        for k, v in train_infos.items():
            prefixed[f"{section}/train/{k}"] = v
        if section == "stage2":
            prefixed.update(self._stage2_reward_shaping_payload(total_num_steps))
        if self.use_wandb:
            wandb.log(prefixed, step=total_num_steps)
        elif hasattr(self, "writter"):
            for k, v in prefixed.items():
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """Env metrics에 section prefix를 붙여 wandb에 로깅."""
        section = self._wandb_section
        if section == "stage1_mep":
            prefixed = {}
            for key in sorted(env_infos):
                policy_name, metric = self._stage1_mep_selfplay_reward_metric(key)
                if policy_name is None:
                    continue
                scalar = self._as_scalar(env_infos[key])
                if scalar is None:
                    continue
                prefixed[f"{section}/env/{policy_name}/{metric}"] = scalar
            if self.use_wandb and prefixed:
                wandb.log(prefixed, step=total_num_steps)
            elif hasattr(self, "writter"):
                for k, v in prefixed.items():
                    self.writter.add_scalars(k, {k: v}, total_num_steps)
            return

        if section == "stage2" and not self._stage2_log_raw_scalars:
            scalar_payload = self._stage2_train_env_payload(env_infos)
            sampling_rows = self._stage2_sampling_rows(total_num_steps)
            if self.use_wandb:
                payload = {
                    **scalar_payload,
                    **self._stage2_table_payload(sampling_rows),
                }
                if payload:
                    wandb.log(payload, step=total_num_steps)
            elif hasattr(self, "writter"):
                for key, value in scalar_payload.items():
                    self.writter.add_scalars(key, {key: value}, total_num_steps)
            return

        prefixed = {}
        for k, v in env_infos.items():
            if section == "stage1_hsp" and k not in STAGE1_HSP_ENV_REWARD_METRICS:
                continue
            if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                if section == "stage1_hsp":
                    prefixed[f"{self._stage1_hsp_prefix}/env/{k}"] = np.mean(v)
                else:
                    prefixed[f"{section}/env/{k}"] = np.mean(v)
            elif isinstance(v, (int, float)):
                if section == "stage1_hsp":
                    prefixed[f"{self._stage1_hsp_prefix}/env/{k}"] = v
                else:
                    prefixed[f"{section}/env/{k}"] = v
        if section == "stage1_hsp":
            self._write_fcp_s1_sparse_history(env_infos, total_num_steps)
        if self.use_wandb and prefixed:
            if section == "stage1_hsp":
                prefixed[f"{self._stage1_hsp_prefix}/local_env_step"] = int(total_num_steps)
                self._define_stage1_hsp_wandb_metrics()
                if getattr(self.all_args, "hsp_wandb_seed_label", ""):
                    wandb.log(prefixed)
                else:
                    wandb.log(prefixed, step=total_num_steps)
            else:
                wandb.log(prefixed, step=total_num_steps)
        elif hasattr(self, "writter"):
            for k, v in prefixed.items():
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    # ══════════════════════════════════════════════════════════════════
    #  Override: _log_sampling_prob → section prefix 부여
    # ══════════════════════════════════════════════════════════════════

    def _log_sampling_prob(self, sampling_prob_dict, total_num_steps):
        """Sampling probability에 section prefix를 붙여 로깅."""
        section = self._wandb_section
        if section == "stage1_mep":
            return

        if section == "stage2":
            for key, value in sampling_prob_dict.items():
                if key.startswith("sampling_prob/"):
                    self._stage2_latest_sampling[key.split("/", 1)[1]] = float(value)
            if not self._stage2_log_raw_scalars:
                return

        prefixed = {f"{section}/pool/{k}": v for k, v in sampling_prob_dict.items()}
        if self.use_wandb:
            wandb.log(prefixed, step=total_num_steps)

    # ══════════════════════════════════════════════════════════════════
    #  Override: _on_trainer_reset → matchup count 추적
    # ══════════════════════════════════════════════════════════════════

    def _on_trainer_reset(self, map_ea2t, episode):
        """map_ea2t가 갱신될 때 policy pool matchup 빈도를 추적."""
        # env별로 (agent0_trainer, agent1_trainer) 쌍을 카운트
        n_envs = self.n_rollout_threads
        for e in range(n_envs):
            t0 = map_ea2t.get((e, 0), "unknown")
            t1 = map_ea2t.get((e, 1), "unknown")
            self._matchup_counter[f"{t0}-{t1}"] += 1

        if self._wandb_section == "stage1_mep":
            return

        if self._wandb_section == "stage2" and not self._stage2_log_raw_scalars:
            return

        # 주기적으로 matchup count를 wandb에 로깅
        if self.use_wandb and hasattr(self, "total_num_steps"):
            section = self._wandb_section
            matchup_log = {
                f"{section}/pool/matchup_count/{pair}": count
                for pair, count in self._matchup_counter.items()
            }
            wandb.log(matchup_log, step=self.total_num_steps)

    # ══════════════════════════════════════════════════════════════════
    #  Override: _on_multi_policy_env_infos → risky event count 추가
    # ══════════════════════════════════════════════════════════════════

    def _on_multi_policy_env_infos(self, episode_env_infos, infos):
        """naive_train_with_multi_policy의 episode_env_infos에 39개 risky event count를 추가."""
        if not hasattr(self, "trainer") or not hasattr(self.trainer, "map_ea2t"):
            return

        for e, info in enumerate(infos):
            categories = info.get("episode", {}).get("ep_category_r_by_agent")
            if categories is None:
                continue

            agent0_trainer = self.trainer.map_ea2t.get((e, 0), "unknown")
            agent1_trainer = self.trainer.map_ea2t.get((e, 1), "unknown")

            for log_name in [
                f"{agent0_trainer}-{agent1_trainer}",
                f"agent0-{agent0_trainer}",
                f"agent1-{agent1_trainer}",
                f"either-{agent0_trainer}",
                f"either-{agent1_trainer}",
            ]:
                for agent_idx in range(self.num_agents):
                    for event_idx, event_name in enumerate(CATEGORY_KEYS):
                        key = f"{log_name}-ep_{event_name}_by_agent{agent_idx}"
                        episode_env_infos[key].append(categories[agent_idx][event_idx])

    # ══════════════════════════════════════════════════════════════════
    #  Risky event category helpers (기존 코드 유지)
    # ══════════════════════════════════════════════════════════════════

    def _append_risky_category_logs(self, eval_env_infos, eval_info):
        """Eval 결과에 risky event count (39개 전부) 추가."""
        categories = eval_info["episode"].get("ep_category_r_by_agent")
        if categories is None:
            return
        for agent_idx in range(self.num_agents):
            for event_idx, event_name in enumerate(CATEGORY_KEYS):
                eval_env_infos[f"eval_ep_{event_name}_by_agent{agent_idx}"].append(
                    categories[agent_idx][event_idx]
                )

    def _append_risky_train_category_logs(self, env_infos, info):
        """Train 결과에 risky event count (39개 전부) 추가."""
        categories = info["episode"].get("ep_category_r_by_agent")
        if categories is None:
            return
        for agent_idx in range(self.num_agents):
            for event_idx, event_name in enumerate(CATEGORY_KEYS):
                env_infos[f"ep_{event_name}_by_agent{agent_idx}"].append(
                    categories[agent_idx][event_idx]
                )

    @staticmethod
    def _flatten_gif_paths(raw_paths) -> list[Path]:
        paths = []
        for raw_path in raw_paths or []:
            if isinstance(raw_path, (list, tuple)):
                paths.extend(Path(path) for path in raw_path if path)
            elif raw_path:
                paths.append(Path(raw_path))
        return [path for path in paths if path.exists()]

    def _final_stage1_mep_local_steps(self) -> int:
        episodes = int(self.all_args.num_env_steps) // self.episode_length // self.n_rollout_threads
        return int(episodes * self.episode_length * self.n_rollout_threads)

    def _sample_stage1_mep_final_pairs(self, gif_episodes: int) -> list[tuple[str, str]]:
        adaptive_name = str(getattr(self.all_args, "adaptive_agent_name", "mep_adaptive"))
        population = sorted(
            name
            for name in getattr(self.trainer, "population", {}).keys()
            if name != adaptive_name
        )
        ordered_pairs = [(agent0, agent1) for agent0 in population for agent1 in population]
        if gif_episodes <= 0 or not ordered_pairs:
            return []

        rng_seed = int(getattr(self.all_args, "seed", 1)) * 1000003 + 17
        rng = np.random.default_rng(rng_seed)
        replace = len(ordered_pairs) < gif_episodes
        indices = rng.choice(len(ordered_pairs), size=gif_episodes, replace=replace)
        return [ordered_pairs[int(index)] for index in np.atleast_1d(indices)]

    @torch.no_grad()
    def _run_stage1_mep_final_render_rollout(
        self, pairs: list[tuple[str, str]]
    ) -> list[Path]:
        if self.final_render_envs is None or not pairs:
            return []

        render_threads = min(int(getattr(self.final_render_envs, "num_envs", len(pairs)) or 1), len(pairs))
        pairs = pairs[:render_threads]
        map_ea2p = {
            (env_idx, agent_idx): pair[agent_idx]
            for env_idx, pair in enumerate(pairs)
            for agent_idx in range(self.num_agents)
        }
        self.policy.set_map_ea2p(map_ea2p, load_unused_to_cpu=True)
        if hasattr(self.final_render_envs, "reset_featurize_type"):
            featurize_type = [
                [self.policy.featurize_type[map_ea2p[(env_idx, agent_idx)]] for agent_idx in range(self.num_agents)]
                for env_idx in range(render_threads)
            ]
            self.final_render_envs.reset_featurize_type(featurize_type)

        policy_pool = self.policy.policy_pool
        [policy.reset(render_threads, self.num_agents) for policy in policy_pool.values()]
        for env_idx in range(render_threads):
            for agent_idx in range(self.num_agents):
                policy_name = map_ea2p[(env_idx, agent_idx)]
                if not policy_name.startswith("script:"):
                    policy_pool[policy_name].register_control_agent(env_idx, agent_idx)

        reset_choose = np.ones(render_threads) == 1
        eval_obs, _, _ = self.final_render_envs.reset(reset_choose)
        for _ in range(self.episode_length):
            eval_actions = np.full(
                (render_threads, self.num_agents, 1), fill_value=0
            ).tolist()
            for policy in policy_pool.values():
                if len(policy.control_agents) <= 0:
                    continue
                policy.prep_rollout()
                policy.to(self.device)
                obs = np.stack([eval_obs[env_idx][agent_idx] for env_idx, agent_idx in policy.control_agents])
                actions = policy.step(obs, policy.control_agents, deterministic=True)
                for action, (env_idx, agent_idx) in zip(actions, policy.control_agents):
                    eval_actions[env_idx][agent_idx] = action

            eval_obs, _, _, _, _, _ = self.final_render_envs.step(np.array(eval_actions))

        if not hasattr(self.final_render_envs, "get_last_render_gif_paths"):
            return []
        return self._flatten_gif_paths(self.final_render_envs.get_last_render_gif_paths())

    def _log_stage1_mep_final_gifs(self, total_num_steps: int) -> None:
        if self._stage1_mep_final_gifs_logged or self._wandb_section != "stage1_mep":
            return
        gif_episodes = max(
            0, int(getattr(self.all_args, "mep_final_gif_episodes", 0) or 0)
        )
        if gif_episodes <= 0 or self.final_render_envs is None:
            return

        pairs = self._sample_stage1_mep_final_pairs(gif_episodes)
        gif_paths = self._run_stage1_mep_final_render_rollout(pairs)[:gif_episodes]
        self._stage1_mep_final_gifs_logged = True
        print(
            f"generated {len(gif_paths)}/{gif_episodes} final MEP S1 GIFs "
            f"at step {int(total_num_steps)}"
        )
        if not self.use_wandb or not gif_paths:
            return

        payload = {"stage1_mep_final_gifs/local_env_step": int(total_num_steps)}
        for gif_index, (gif_path, pair) in enumerate(zip(gif_paths, pairs), start=1):
            agent0, agent1 = pair
            payload[f"stage1_mep_final_gifs/gif_{gif_index:02d}_{agent0}_vs_{agent1}"] = (
                wandb.Video(str(gif_path), format="gif")
            )
        wandb.log(payload)

    @torch.no_grad()
    def _run_stage1_hsp_final_render_rollout(self) -> list[Path]:
        if self.final_render_envs is None:
            return []

        render_threads = int(getattr(self.final_render_envs, "num_envs", 1) or 1)
        obs, _, _ = self.final_render_envs.reset()
        obs = np.stack(obs)
        rnn_states = np.zeros(
            (
                render_threads,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        masks = np.ones((render_threads, self.num_agents, 1), dtype=np.float32)

        for _ in range(self.episode_length):
            self.trainer.prep_rollout()
            action, rnn_states = self.trainer.policy.act(
                np.concatenate(obs),
                np.concatenate(rnn_states),
                np.concatenate(masks),
                deterministic=True,
            )
            actions = np.array(np.split(_t2n(action), render_threads))
            rnn_states = np.array(np.split(_t2n(rnn_states), render_threads))

            obs, _, _, dones, _, _ = self.final_render_envs.step(actions)
            obs = np.stack(obs)
            rnn_states[dones == True] = np.zeros(
                ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            masks = np.ones((render_threads, self.num_agents, 1), dtype=np.float32)
            masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if not hasattr(self.final_render_envs, "get_last_render_gif_paths"):
            return []
        return self._flatten_gif_paths(self.final_render_envs.get_last_render_gif_paths())

    def _log_stage1_hsp_final_gifs(self, total_num_steps: int) -> None:
        if self._stage1_hsp_final_gifs_logged or self._wandb_section != "stage1_hsp":
            return
        gif_episodes = max(
            0, int(getattr(self.all_args, "hsp_final_gif_episodes", 0) or 0)
        )
        if gif_episodes <= 0 or self.final_render_envs is None:
            return

        gif_paths = self._run_stage1_hsp_final_render_rollout()[:gif_episodes]
        if len(gif_paths) != gif_episodes:
            raise RuntimeError(
                f"expected {gif_episodes} final HSP S1 GIFs for "
                f"{self._stage1_hsp_prefix}, generated {len(gif_paths)}"
            )
        self._stage1_hsp_final_gifs_logged = True
        print(
            f"generated {len(gif_paths)}/{gif_episodes} final HSP S1 GIFs "
            f"for {self._stage1_hsp_prefix} at step {total_num_steps}"
        )
        if not self.use_wandb or not gif_paths:
            return

        prefix = self._stage1_hsp_prefix
        step = int(total_num_steps)
        payload = {f"{prefix}/final_gifs/local_env_step": step}
        for gif_index, gif_path in enumerate(gif_paths, start=1):
            payload[f"{prefix}/final_gifs/step_{step}_gif_{gif_index}"] = wandb.Video(
                str(gif_path), format="gif"
            )
        wandb.log(payload)

    # ══════════════════════════════════════════════════════════════════
    #  Override: run() → Stage 1 HSP 학습 루프
    # ══════════════════════════════════════════════════════════════════

    def run(self):
        """Stage 1 HSP training loop with risky event logging."""
        self._wandb_section = "stage1_hsp"
        self._log_stage1_hsp_hidden_utility_selection()
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                obs = np.stack(obs)
                total_num_steps += self.n_rollout_threads
                self.envs.anneal_reward_shaping_factor([total_num_steps] * self.n_rollout_threads)
                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                self.insert(data)

            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode < 50:
                if episode % 1 == 0:
                    self.save(episode)
            elif episode < 100:
                if episode % 2 == 0:
                    self.save(episode)
            elif episode % self.save_interval == 0 or episode == episodes - 1:
                self.save(episode)

            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Layout {} Algo {} Exp {} updates {}/{} episodes, "
                    "total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.layout_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))

                env_infos = defaultdict(list)
                for info in infos:
                    self._append_risky_train_category_logs(env_infos, info)
                    env_infos["ep_sparse_r_by_agent0"].append(info["episode"]["ep_sparse_r_by_agent"][0])
                    env_infos["ep_sparse_r_by_agent1"].append(info["episode"]["ep_sparse_r_by_agent"][1])
                    env_infos["ep_shaped_r_by_agent0"].append(info["episode"]["ep_shaped_r_by_agent"][0])
                    env_infos["ep_shaped_r_by_agent1"].append(info["episode"]["ep_shaped_r_by_agent"][1])
                    env_infos["ep_sparse_r"].append(info["episode"]["ep_sparse_r"])
                    env_infos["ep_shaped_r"].append(info["episode"]["ep_shaped_r"])

                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

        self._log_stage1_hsp_final_gifs(total_num_steps)

    # ══════════════════════════════════════════════════════════════════
    #  Override: train_mep() wrapper → section 설정
    # ══════════════════════════════════════════════════════════════════

    def train_mep(self):
        """train_mep 진입 시 stage에 따라 wandb section을 설정."""
        stage = getattr(self.all_args, "stage", 1)
        if stage == 1:
            self._wandb_section = "stage1_mep"
        else:
            self._wandb_section = "stage2"
            self._matchup_counter.clear()
            self._stage2_latest_sampling.clear()
            self._log_stage2_reward_shaping(0)
        super().train_mep()
        if stage == 1:
            self._log_stage1_mep_final_gifs(self._final_stage1_mep_local_steps())
        else:
            self._log_stage2_reward_shaping(self._stage2_final_num_steps())

    def finalize_stage2_logging(self) -> None:
        """Upload compact Stage 2 CSV files as one W&B artifact."""
        if self._wandb_section != "stage2" or not self.use_wandb:
            return
        existing = [path for path in self._stage2_csv_paths.values() if path.exists()]
        if not existing:
            return
        artifact = wandb.Artifact(
            name=f"hsp-stage2-training-tables-{self.all_args.layout_name}",
            type="training",
            metadata={
                "layout": self.all_args.layout_name,
                "experiment_name": self.experiment_name,
                "seed": self.all_args.seed,
            },
        )
        for path in existing:
            artifact.add_file(str(path), name=path.name)
        wandb.log_artifact(artifact)

    # ══════════════════════════════════════════════════════════════════
    #  Override: evaluate_one_episode_with_multi_policy → risky events
    # ══════════════════════════════════════════════════════════════════

    def evaluate_one_episode_with_multi_policy(self, policy_pool: Dict, map_ea2p: Dict):
        """Evaluate one episode and log Risky event categories by EVENT_TYPES."""

        [policy.reset(self.n_eval_rollout_threads, self.num_agents) for policy in policy_pool.values()]
        for env_idx in range(self.n_eval_rollout_threads):
            for agent_id in range(self.num_agents):
                if not map_ea2p[(env_idx, agent_id)].startswith("script:"):
                    policy_pool[map_ea2p[(env_idx, agent_id)]].register_control_agent(
                        env_idx, agent_id
                    )

        eval_env_infos = defaultdict(list)
        reset_choose = np.ones(self.n_eval_rollout_threads) == 1
        eval_obs, _, _ = self.eval_envs.reset(reset_choose)

        extract_info_keys = []
        infos = None
        eval_infos = None
        for _ in range(self.all_args.episode_length):
            eval_actions = np.full(
                (self.n_eval_rollout_threads, self.num_agents, 1), fill_value=0
            ).tolist()
            for policy_name, policy in policy_pool.items():
                if len(policy.control_agents) == 0:
                    continue
                policy.prep_rollout()
                policy.to(self.device)
                obs_lst = [eval_obs[e][a] for (e, a) in policy.control_agents]
                info_lst = None
                if infos is not None:
                    info_lst = {
                        key: [infos[e][key][a] for e, a in policy.control_agents]
                        for key in extract_info_keys
                    }
                agents = policy.control_agents
                actions = policy.step(
                    np.stack(obs_lst, axis=0),
                    agents,
                    info=info_lst,
                    deterministic=not self.all_args.eval_stochastic,
                )
                for action, (env_idx, agent_id) in zip(actions, agents):
                    eval_actions[env_idx][agent_id] = action

            eval_actions = np.array(eval_actions)
            eval_obs, _, _, _, eval_infos, _ = self.eval_envs.step(eval_actions)
            infos = eval_infos

        for eval_info in eval_infos:
            self._append_risky_category_logs(eval_env_infos, eval_info)
            for agent_idx in range(self.num_agents):
                eval_env_infos[f"eval_ep_sparse_r_by_agent{agent_idx}"].append(
                    eval_info["episode"]["ep_sparse_r_by_agent"][agent_idx]
                )
                eval_env_infos[f"eval_ep_shaped_r_by_agent{agent_idx}"].append(
                    eval_info["episode"]["ep_shaped_r_by_agent"][agent_idx]
                )
            eval_env_infos["eval_ep_sparse_r"].append(eval_info["episode"]["ep_sparse_r"])
            eval_env_infos["eval_ep_shaped_r"].append(eval_info["episode"]["ep_shaped_r"])

        return eval_env_infos
