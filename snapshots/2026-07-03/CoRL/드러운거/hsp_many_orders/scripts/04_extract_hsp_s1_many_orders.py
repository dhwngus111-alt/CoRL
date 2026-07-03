#!/home/isl_jhoh/miniconda3/envs/corl/bin/python
import os
import re
import shutil
from pathlib import Path

import yaml


TEST_ROOT = Path(os.environ.get("TEST_ROOT", "/home/isl_jhoh/CoRL/test/hsp_many_orders"))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", TEST_ROOT / "results"))
POLICY_POOL = Path(os.environ.get("POLICY_POOL", TEST_ROOT / "policy_pool"))
RUN_BASE = RESULTS_ROOT / "Overcooked" / "many_orders" / "mappo" / "hsp-S1"
DEST = POLICY_POOL / "many_orders" / "hsp" / "s1"
SEED_START = int(os.environ.get("HSP_S1_SEED_START", "1"))
SEED_END = int(os.environ.get("HSP_S1_SEED_END", "36"))


def find_run_file_dirs():
    candidates = []
    candidates.extend(RUN_BASE.glob("wandb/run-*/files"))
    candidates.extend(RUN_BASE.glob("run*/models"))
    candidates.extend(RUN_BASE.glob("run*"))
    return sorted({p for p in candidates if p.exists()}, key=lambda p: p.stat().st_mtime, reverse=True)


def read_seed(files_dir: Path):
    config_path = files_dir / "config.yaml"
    if config_path.exists():
        try:
            data = yaml.safe_load(config_path.read_text()) or {}
            seed = data.get("seed")
            if isinstance(seed, dict):
                seed = seed.get("value")
            if seed is not None:
                return int(seed)
        except Exception:
            pass

    for path in [files_dir, *files_dir.parents]:
        match = re.search(r"seed[_-]?(\d+)", path.name)
        if match:
            return int(match.group(1))
    return None


def checkpoint_version(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def latest_checkpoint(files_dir: Path, agent_id: int):
    ckpts = sorted(files_dir.glob(f"actor_agent{agent_id}_periodic_*.pt"), key=checkpoint_version)
    if not ckpts:
        raise FileNotFoundError(f"No actor_agent{agent_id} checkpoint in {files_dir}")
    return ckpts[-1]


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    run_dirs = find_run_file_dirs()
    if not run_dirs:
        raise FileNotFoundError(f"No HSP S1 run directories found under {RUN_BASE}")

    run_by_seed = {}
    for run_dir in run_dirs:
        if not list(run_dir.glob("actor_agent0_periodic_*.pt")):
            continue
        seed = read_seed(run_dir)
        if seed is None:
            print(f"Skipping run without seed metadata: {run_dir}")
            continue
        run_by_seed.setdefault(seed, run_dir)

    missing = []
    for seed in range(SEED_START, SEED_END + 1):
        run_dir = run_by_seed.get(seed)
        if run_dir is None:
            missing.append(seed)
            continue
        src_w0 = latest_checkpoint(run_dir, 0)
        src_w1 = latest_checkpoint(run_dir, 1)
        dst_w0 = DEST / f"hsp{seed}_w0_actor.pt"
        dst_w1 = DEST / f"hsp{seed}_w1_actor.pt"
        shutil.copy2(src_w0, dst_w0)
        shutil.copy2(src_w1, dst_w1)
        print(f"seed {seed}: {src_w0.name}, {src_w1.name} -> {DEST}")

    if missing:
        raise RuntimeError(f"Missing HSP S1 runs for seeds: {missing}")
    print("HSP S1 extraction complete.")


if __name__ == "__main__":
    main()
