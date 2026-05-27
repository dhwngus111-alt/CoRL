"""Separated Risky Overcooked runner with HSP S1 bundle logging."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch
import wandb

from hsp.runner.separated.overcooked_runner import (
    OvercookedRunner as HSPSeparatedRunner,
    _t2n,
)
from transplant.hsp_selection_logging import log_hidden_utility_selection


HSP_S1_TRAIN_REWARD_KEYS = {"average_episode_rewards"}
HSP_S1_ENV_REWARD_KEYS = {
    "ep_sparse_r",
    "ep_shaped_r",
    "ep_sparse_r_by_agent0",
    "ep_sparse_r_by_agent1",
    "ep_shaped_r_by_agent0",
    "ep_shaped_r_by_agent1",
}


class RiskySeparatedOvercookedRunner(HSPSeparatedRunner):
    """Keep separated HSP training intact while organizing W&B by seed."""

    def __init__(self, config):
        super().__init__(config)
        self.final_render_envs = config.get("final_render_envs")
        self.risky_env_name = self.env_name
        self.env_name = "Overcooked"
        self._stage1_hsp_seed_label = getattr(
            self.all_args, "hsp_wandb_seed_label", f"seed{int(self.all_args.seed):02d}"
        )
        self._stage1_hsp_metrics_defined = False
        self._stage1_hsp_selection_logged = False
        self._stage1_hsp_final_gifs_logged = False

        local_run_dir = config.get("local_run_dir")
        if local_run_dir is not None:
            self._use_local_run_dir(Path(local_run_dir))

    def _use_local_run_dir(self, local_run_dir: Path) -> None:
        local_run_dir.mkdir(parents=True, exist_ok=True)
        current_policy_config = Path(self.run_dir) / "policy_config.pkl"
        if current_policy_config.exists():
            shutil.copy2(current_policy_config, local_run_dir / "policy_config.pkl")
        self.run_dir = str(local_run_dir)
        self.save_dir = str(local_run_dir)

    @property
    def _stage1_hsp_prefix(self) -> str:
        return self._stage1_hsp_seed_label

    def _define_stage1_hsp_wandb_metrics(self) -> None:
        if self._stage1_hsp_metrics_defined:
            return
        step_key = f"{self._stage1_hsp_prefix}/local_env_step"
        wandb.define_metric(step_key)
        wandb.define_metric(f"{self._stage1_hsp_prefix}/train/*", step_metric=step_key)
        for agent_id in range(self.num_agents):
            wandb.define_metric(
                f"{self._stage1_hsp_prefix}/train/agent{agent_id}/average_episode_rewards",
                step_metric=step_key,
            )
        wandb.define_metric(
            f"{self._stage1_hsp_prefix}/train/average_episode_rewards",
            step_metric=step_key,
        )
        wandb.define_metric(f"{self._stage1_hsp_prefix}/env/*", step_metric=step_key)
        for key in HSP_S1_ENV_REWARD_KEYS:
            wandb.define_metric(f"{self._stage1_hsp_prefix}/env/{key}", step_metric=step_key)
        self._stage1_hsp_metrics_defined = True

    def _log_stage1_hsp_hidden_utility_selection(self) -> None:
        if self._stage1_hsp_selection_logged:
            return
        log_hidden_utility_selection(
            all_args=self.all_args,
            run_dir=self.run_dir,
            prefix=self._stage1_hsp_prefix,
            use_wandb=self.use_wandb,
        )
        self._stage1_hsp_selection_logged = True

    def run(self):
        self._log_stage1_hsp_hidden_utility_selection()
        result = super().run()
        self._log_stage1_hsp_final_gifs(self._final_train_steps())
        return result

    def _final_train_steps(self) -> int:
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        return int(episodes * self.episode_length * self.n_rollout_threads)

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
    def _flatten_gif_paths(raw_paths) -> list[Path]:
        paths = []
        for raw_path in raw_paths or []:
            if isinstance(raw_path, (list, tuple)):
                paths.extend(Path(path) for path in raw_path if path)
            elif raw_path:
                paths.append(Path(raw_path))
        return [path for path in paths if path.exists()]

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
            actions = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                action, rnn_state = self.trainer[agent_id].policy.act(
                    obs[:, agent_id],
                    rnn_states[:, agent_id],
                    masks[:, agent_id],
                    deterministic=True,
                )
                actions.append(_t2n(action))
                rnn_states[:, agent_id] = _t2n(rnn_state)

            actions = np.array(actions).transpose(1, 0, 2)
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
        if self._stage1_hsp_final_gifs_logged:
            return
        gif_episodes = max(
            0, int(getattr(self.all_args, "hsp_final_gif_episodes", 0) or 0)
        )
        if gif_episodes <= 0 or self.final_render_envs is None:
            return

        gif_paths = self._run_stage1_hsp_final_render_rollout()[:gif_episodes]
        self._stage1_hsp_final_gifs_logged = True
        print(
            f"generated {len(gif_paths)}/{gif_episodes} final HSP S1 GIFs "
            f"for {self._stage1_hsp_prefix} at step {total_num_steps}"
        )
        if not self.use_wandb or not gif_paths:
            return

        payload = {f"{self._stage1_hsp_prefix}/final_gifs/local_env_step": int(total_num_steps)}
        for gif_index, gif_path in enumerate(gif_paths, start=1):
            payload[
                f"{self._stage1_hsp_prefix}/final_gifs/"
                f"step_{int(total_num_steps)}_gif_{gif_index}"
            ] = wandb.Video(str(gif_path), format="gif")
        wandb.log(payload)

    def log_train(self, train_infos, total_num_steps):
        if not self.use_wandb:
            return super().log_train(train_infos, total_num_steps)

        self._define_stage1_hsp_wandb_metrics()
        prefix = self._stage1_hsp_prefix
        payload = {f"{prefix}/local_env_step": int(total_num_steps)}
        metric_values = {}

        for agent_id in range(self.num_agents):
            for key, value in train_infos[agent_id].items():
                if key not in HSP_S1_TRAIN_REWARD_KEYS:
                    continue
                scalar = self._as_scalar(value)
                if scalar is None:
                    continue
                payload[f"{prefix}/train/agent{agent_id}/{key}"] = scalar
                metric_values.setdefault(key, []).append(scalar)

        for key, values in metric_values.items():
            if values:
                payload[f"{prefix}/train/{key}"] = float(np.mean(values))

        wandb.log(payload)

    def log_env(self, env_infos, total_num_steps):
        if not self.use_wandb:
            return super().log_env(env_infos, total_num_steps)

        self._define_stage1_hsp_wandb_metrics()
        prefix = self._stage1_hsp_prefix
        payload = {f"{prefix}/local_env_step": int(total_num_steps)}
        for key, value in env_infos.items():
            if key not in HSP_S1_ENV_REWARD_KEYS:
                continue
            if len(value) > 0:
                payload[f"{prefix}/env/{key}"] = float(np.mean(value))
        wandb.log(payload)

    def log_system(self):
        pass
