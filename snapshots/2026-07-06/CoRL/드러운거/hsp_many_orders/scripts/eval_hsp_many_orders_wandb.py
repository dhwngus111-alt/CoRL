#!/home/isl_jhoh/miniconda3/envs/corl/bin/python
import argparse
import csv
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import wandb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_HSP_ROOT = Path("/home/isl_jhoh/CoRL/HSP")
DEFAULT_TEST_ROOT = Path("/home/isl_jhoh/CoRL/test/hsp_many_orders")
DEFAULT_RUN_ROOT = DEFAULT_TEST_ROOT / "final_eval"
DEFAULT_POLICY_POOL = DEFAULT_TEST_ROOT / "policy_pool"
DEFAULT_STUB_ROOT = Path("/home/isl_jhoh/CoRL/test/hsp_many_orders/stubs")


PARTNERS = OrderedDict(
    [
        ("tomato_placement", "script:place_tomato_in_pot"),
        ("tomato_place_delivery", "script:place_tomato_and_deliver_soup"),
        ("onion_placement", "script:place_onion_in_pot"),
        ("onion_place_delivery", "script:place_onion_and_deliver_soup"),
        ("delivery", "script:deliver_soup"),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the many_orders HSP adaptive policy and upload metrics/GIFs to wandb."
    )
    parser.add_argument("--hsp-root", type=Path, default=DEFAULT_HSP_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--population-yaml", type=Path, default=None)
    parser.add_argument("--policy-pool", type=Path, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="Overcooked")
    parser.add_argument("--wandb-run-name", type=str, default="hsp_many_orders_final_eval")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--agent0-policy-name", type=str, default="hsp_adaptive")
    parser.add_argument("--partners", nargs="*", default=list(PARTNERS.keys()))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=400)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-threads", type=int, default=100)
    parser.add_argument("--gifs-per-partner", type=int, default=3)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--eval-stochastic", action="store_true", default=True)
    parser.add_argument("--deterministic-eval", action="store_false", dest="eval_stochastic")
    return parser.parse_args()


def load_hsp_modules(hsp_root: Path):
    sys.path.insert(0, str(DEFAULT_STUB_ROOT))
    sys.path.insert(0, str(hsp_root))
    from hsp.config import get_config
    from hsp.envs.env_wrappers import ChooseSubprocVecEnv
    from hsp.envs.overcooked_new.Overcooked_Env import Overcooked as OvercookedNew
    from hsp.runner.shared.overcooked_runner import OvercookedRunner

    return get_config, ChooseSubprocVecEnv, OvercookedNew, OvercookedRunner


def build_hsp_args(get_config, argv):
    parser = get_config()
    parser.add_argument("--layout_name", type=str, default="many_orders")
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--initial_reward_shaping_factor", type=float, default=1.0)
    parser.add_argument("--reward_shaping_factor", type=float, default=1.0)
    parser.add_argument("--reward_shaping_horizon", type=int, default=100000000)
    parser.add_argument("--use_phi", default=False, action="store_true")
    parser.add_argument("--use_hsp", default=False, action="store_true")
    parser.add_argument("--random_index", default=False, action="store_true")
    parser.add_argument("--use_agent_policy_id", default=True, action="store_true")
    parser.add_argument("--overcooked_version", default="new", choices=["new", "old"])
    parser.add_argument("--use_detailed_rew_shaping", default=False, action="store_true")
    parser.add_argument("--random_start_prob", default=0.0, type=float)
    parser.add_argument("--store_traj", default=False, action="store_true")
    parser.add_argument("--population_yaml_path", type=str, required=True)
    parser.add_argument("--agent0_policy_name", type=str, required=True)
    parser.add_argument("--agent1_policy_name", type=str, required=True)
    return parser.parse_known_args(argv)[0]


def make_eval_env(all_args, run_dir, ChooseSubprocVecEnv, OvercookedNew):
    def get_env_fn(rank):
        def init_env():
            env = OvercookedNew(all_args, run_dir, featurize_type=("bc", "bc"), rank=rank)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    return ChooseSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def evaluate_pair(
    args,
    get_config,
    ChooseSubprocVecEnv,
    OvercookedNew,
    OvercookedRunner,
    partner_name,
    partner_policy,
    run_dir,
    eval_episodes,
    eval_threads,
    render,
):
    argv = [
        "--env_name",
        "Overcooked",
        "--algorithm_name",
        "population",
        "--experiment_name",
        f"eval-hsp-many-orders-{partner_name}",
        "--layout_name",
        "many_orders",
        "--num_agents",
        "2",
        "--seed",
        str(args.seed),
        "--episode_length",
        str(args.episode_length),
        "--n_eval_rollout_threads",
        str(eval_threads),
        "--eval_episodes",
        str(eval_episodes),
        "--population_yaml_path",
        str(args.population_yaml),
        "--agent0_policy_name",
        args.agent0_policy_name,
        "--agent1_policy_name",
        partner_policy,
        "--overcooked_version",
        "new",
        "--use_agent_policy_id",
    ]
    if args.eval_stochastic:
        argv.append("--eval_stochastic")
    if render:
        argv.append("--use_render")

    all_args = build_hsp_args(get_config, argv)
    all_args.cuda = bool(all_args.cuda and torch.cuda.is_available())

    device = torch.device("cuda:0" if all_args.cuda else "cpu")
    torch.set_num_threads(all_args.n_training_threads)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_eval_env(all_args, run_dir, ChooseSubprocVecEnv, OvercookedNew)
    eval_envs = make_eval_env(all_args, run_dir, ChooseSubprocVecEnv, OvercookedNew)
    try:
        runner = OvercookedRunner(
            {
                "all_args": all_args,
                "envs": envs,
                "eval_envs": eval_envs,
                "num_agents": all_args.num_agents,
                "device": device,
                "run_dir": run_dir,
            }
        )
        featurize_type = runner.policy.load_population(str(args.population_yaml), evaluation=True)
        map_ea2p = {}
        for e in range(all_args.n_eval_rollout_threads):
            map_ea2p[(e, 0)] = args.agent0_policy_name
            map_ea2p[(e, 1)] = partner_policy
        runner.policy.set_map_ea2p(map_ea2p)

        agent0_featurize_type = featurize_type.get(args.agent0_policy_name, "ppo")
        agent1_featurize_type = featurize_type.get(partner_policy, "ppo")
        eval_envs.reset_featurize_type(
            [(agent0_featurize_type, agent1_featurize_type) for _ in range(all_args.n_eval_rollout_threads)]
        )
        return runner.evaluate_with_multi_policy(num_eval_episodes=eval_episodes)
    finally:
        envs.close()
        eval_envs.close()


def flatten_eval_metrics(eval_infos, agent0_policy_name, partner_policy):
    metrics = {}
    prefix = f"{agent0_policy_name}-{partner_policy}-"
    for key, value in eval_infos.items():
        if key.startswith(prefix):
            short_key = key[len(prefix) :]
            metrics[short_key] = float(np.mean(value))
    return metrics


def collect_gifs(run_dir: Path, layout: str):
    gif_root = run_dir / "gifs" / layout
    if not gif_root.exists():
        return []
    return sorted(gif_root.glob("traj_num_*/*.gif"))


def wandb_config(args):
    config = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    return config


def write_metrics_csv(rows, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["partner", "partner_policy", "eval_ep_sparse_r", "eval_ep_shaped_r"])
        writer.writerows(rows)


def write_bar_figure(rows, value_index: int, title: str, ylabel: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partners = [row[0] for row in rows]
    values = [row[value_index] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(partners, values, color="#2f6f73")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("partner")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_local_summary(rows, run_root: Path):
    write_metrics_csv(rows, run_root / "metrics" / "partner_performance.csv")
    write_bar_figure(
        rows,
        2,
        "Many Orders HSP Sparse Reward by Partner",
        "eval_ep_sparse_r",
        run_root / "figures" / "sparse_reward_by_partner.png",
    )
    write_bar_figure(
        rows,
        3,
        "Many Orders HSP Shaped Reward by Partner",
        "eval_ep_shaped_r",
        run_root / "figures" / "shaped_reward_by_partner.png",
    )


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["WANDB_MODE"] = args.wandb_mode
    args.hsp_root = args.hsp_root.resolve()
    args.run_root = args.run_root.resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)
    if args.policy_pool is None:
        args.policy_pool = Path(os.environ.get("POLICY_POOL", DEFAULT_POLICY_POOL))
    if args.population_yaml is None:
        args.population_yaml = args.policy_pool / "many_orders" / "hsp" / "s2" / "eval.yml"
    args.population_yaml = args.population_yaml.resolve()
    args.policy_pool = args.policy_pool.resolve()
    os.environ["POLICY_POOL"] = str(args.policy_pool)

    unknown_partners = [name for name in args.partners if name not in PARTNERS]
    if unknown_partners:
        raise ValueError(f"Unknown partners: {unknown_partners}. Known partners: {list(PARTNERS)}")
    if not args.population_yaml.exists():
        raise FileNotFoundError(args.population_yaml)

    get_config, ChooseSubprocVecEnv, OvercookedNew, OvercookedRunner = load_hsp_modules(args.hsp_root)

    wandb_entity = args.wandb_entity or os.environ.get("WANDB_ENTITY") or None
    if wandb_entity in ("", "WANDB_NAME", "wandb_name", "user"):
        wandb_entity = None

    run = wandb.init(
        entity=wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        group="many_orders",
        job_type="final-eval",
        config=wandb_config(args),
        tags=["hsp", "many_orders", "final_eval"],
    )

    rows = []
    try:
        for partner_idx, partner_name in enumerate(args.partners):
            partner_policy = PARTNERS[partner_name]
            partner_dir = args.run_root / partner_name
            scalar_dir = partner_dir / "scalar_eval"
            render_dir = partner_dir / "render_eval"
            shutil.rmtree(render_dir, ignore_errors=True)
            scalar_dir.mkdir(parents=True, exist_ok=True)
            render_dir.mkdir(parents=True, exist_ok=True)

            eval_infos = evaluate_pair(
                args,
                get_config,
                ChooseSubprocVecEnv,
                OvercookedNew,
                OvercookedRunner,
                partner_name,
                partner_policy,
                scalar_dir,
                args.eval_episodes,
                args.eval_threads,
                render=False,
            )
            metrics = flatten_eval_metrics(eval_infos, args.agent0_policy_name, partner_policy)
            sparse_reward = metrics.get("eval_ep_sparse_r", float("nan"))
            shaped_reward = metrics.get("eval_ep_shaped_r", float("nan"))
            rows.append([partner_name, partner_policy, sparse_reward, shaped_reward])

            log_payload = {
                f"performance/{partner_name}/{key}": value for key, value in metrics.items()
            }
            log_payload.update(
                {
                    "overall/partner_index": partner_idx,
                    "overall/eval_ep_sparse_r": sparse_reward,
                    "overall/eval_ep_shaped_r": shaped_reward,
                }
            )
            wandb.log(log_payload, step=partner_idx)

            evaluate_pair(
                args,
                get_config,
                ChooseSubprocVecEnv,
                OvercookedNew,
                OvercookedRunner,
                partner_name,
                partner_policy,
                render_dir,
                args.gifs_per_partner,
                1,
                render=True,
            )
            gifs = collect_gifs(render_dir, "many_orders")[: args.gifs_per_partner]
            for gif_idx, gif_path in enumerate(gifs, start=1):
                wandb.log(
                    {
                        f"gifs/{partner_name}/episode_{gif_idx}": wandb.Video(
                            str(gif_path), format="gif", caption=f"{args.agent0_policy_name} vs {partner_policy}"
                        )
                    },
                    step=partner_idx,
                )

        write_local_summary(rows, args.run_root)

        table = wandb.Table(
            columns=["partner", "partner_policy", "eval_ep_sparse_r", "eval_ep_shaped_r"],
            data=rows,
        )
        wandb.log(
            {
                "overall/performance_table": table,
                "overall/sparse_reward_by_partner": wandb.plot.bar(
                    table, "partner", "eval_ep_sparse_r", title="Many Orders HSP Sparse Reward by Partner"
                ),
                "overall/shaped_reward_by_partner": wandb.plot.bar(
                    table, "partner", "eval_ep_shaped_r", title="Many Orders HSP Shaped Reward by Partner"
                ),
            }
        )
    finally:
        if rows:
            write_local_summary(rows, args.run_root)
        run.finish()


if __name__ == "__main__":
    main()
