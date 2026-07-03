#!/home/isl_jhoh/miniconda3/envs/corl/bin/python
import os
import shutil
from pathlib import Path

import yaml


TEST_ROOT = Path(os.environ.get("TEST_ROOT", "/home/isl_jhoh/CoRL/test/hsp_many_orders"))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", TEST_ROOT / "results"))
POLICY_POOL = Path(os.environ.get("POLICY_POOL", TEST_ROOT / "policy_pool"))
RUN_BASE = RESULTS_ROOT / "Overcooked" / "many_orders" / "adaptive" / "hsp-S2"
S2_DIR = POLICY_POOL / "many_orders" / "hsp" / "s2"
TRAIN_YML = S2_DIR / "train.yml"
EVAL_YML = S2_DIR / "eval.yml"
ADAPTIVE_NAME = "hsp_adaptive"


def find_run_file_dirs():
    candidates = []
    candidates.extend(RUN_BASE.glob("wandb/run-*/files"))
    candidates.extend(RUN_BASE.glob("run*/models"))
    candidates.extend(RUN_BASE.glob("run*"))
    return sorted({p for p in candidates if p.exists()}, key=lambda p: p.stat().st_mtime, reverse=True)


def checkpoint_version(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def find_latest_adaptive_actor():
    for run_dir in find_run_file_dirs():
        ckpts = sorted((run_dir / ADAPTIVE_NAME).glob("actor_periodic_*.pt"), key=checkpoint_version)
        if ckpts:
            return ckpts[-1], run_dir
    raise FileNotFoundError(f"No {ADAPTIVE_NAME}/actor_periodic_*.pt found under {RUN_BASE}")


def write_eval_yml():
    data = yaml.safe_load(TRAIN_YML.read_text())
    if ADAPTIVE_NAME not in data:
        raise KeyError(f"{ADAPTIVE_NAME} missing from {TRAIN_YML}")
    data[ADAPTIVE_NAME]["train"] = False
    data[ADAPTIVE_NAME]["model_path"] = {"actor": "many_orders/hsp/s2/hsp_adaptive.pt"}
    EVAL_YML.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"Wrote {EVAL_YML}")


def main():
    S2_DIR.mkdir(parents=True, exist_ok=True)
    src, run_dir = find_latest_adaptive_actor()
    dst = S2_DIR / "hsp_adaptive.pt"
    shutil.copy2(src, dst)
    print(f"Using HSP S2 run directory: {run_dir}")
    print(f"{src} -> {dst}")
    write_eval_yml()
    print("HSP S2 extraction complete.")


if __name__ == "__main__":
    main()
