#!/home/isl_jhoh/miniconda3/envs/corl/bin/python
import os
import shutil
from pathlib import Path


TEST_ROOT = Path(os.environ.get("TEST_ROOT", "/home/isl_jhoh/CoRL/test/hsp_many_orders"))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", TEST_ROOT / "results"))
POLICY_POOL = Path(os.environ.get("POLICY_POOL", TEST_ROOT / "policy_pool"))
RUN_BASE = RESULTS_ROOT / "Overcooked" / "many_orders" / "mep" / "mep-S1"
DEST = POLICY_POOL / "many_orders" / "mep" / "s1"
NUM_POLICIES_FOR_HSP = 6


def find_run_file_dirs():
    candidates = []
    candidates.extend(RUN_BASE.glob("wandb/run-*/files"))
    candidates.extend(RUN_BASE.glob("run*/models"))
    candidates.extend(RUN_BASE.glob("run*"))
    return sorted({p for p in candidates if p.exists()}, key=lambda p: p.stat().st_mtime, reverse=True)


def checkpoint_version(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def pick_checkpoints(actor_files):
    actor_files = sorted(actor_files, key=checkpoint_version)
    if len(actor_files) < 3:
        raise RuntimeError(f"Need at least 3 checkpoints, found {len(actor_files)}")
    init_ckpt = actor_files[0]
    mid_ckpt = actor_files[len(actor_files) // 2]
    final_ckpt = actor_files[-1]
    return {"init": init_ckpt, "mid": mid_ckpt, "final": final_ckpt}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    run_dirs = find_run_file_dirs()
    if not run_dirs:
        raise FileNotFoundError(f"No MEP S1 run directories found under {RUN_BASE}")

    selected_run = None
    for run_dir in run_dirs:
        if (run_dir / "mep1").exists() and list((run_dir / "mep1").glob("actor_periodic_*.pt")):
            selected_run = run_dir
            break
    if selected_run is None:
        raise FileNotFoundError(f"No MEP actor checkpoints found under {RUN_BASE}")

    print(f"Using MEP S1 run directory: {selected_run}")
    copied = []
    for policy_id in range(1, NUM_POLICIES_FOR_HSP + 1):
        policy_name = f"mep{policy_id}"
        actor_files = list((selected_run / policy_name).glob("actor_periodic_*.pt"))
        checkpoints = pick_checkpoints(actor_files)
        for tag, src in checkpoints.items():
            dst = DEST / f"{policy_name}_{tag}_actor.pt"
            shutil.copy2(src, dst)
            copied.append(dst)
            print(f"{src} -> {dst}")

    if len(copied) != NUM_POLICIES_FOR_HSP * 3:
        raise RuntimeError(f"Expected {NUM_POLICIES_FOR_HSP * 3} copied files, got {len(copied)}")
    print("MEP S1 extraction complete.")


if __name__ == "__main__":
    main()
