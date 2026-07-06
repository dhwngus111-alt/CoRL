# Snapshot 2026-07-06

This snapshot captures the current `/scratch/isllab0213/isl_jhoh/CoRL`
workspace for the CoMeDi population-training debugging handoff, excluding model
checkpoints and generated training artifacts.

Included:
- `CoMeDi/`
- `CoMeDi_transplant/`
- `FCP_transplant/`
- `HSP/`
- `MEP/`
- `MEP_transplant/`
- `risked_overcooked/`
- `transplant/`
- `드러운거/` lightweight scripts, stubs, and notes
- `environment.yml`
- root README, prompt summary, requirements, and artifact exclusion rules
- `Comedi.pdf`, `FCP.pdf`, `HSP.pdf`, and `mep.pdf`

Highlights:
- upstream CoMeDi source snapshot, excluding nested Git metadata
- CoMeDi transplant training and evaluation entrypoints for Risky Overcooked
- FCP transplant scripts added after the 2026-07-03 snapshot
- HSP, MEP, and Risky Overcooked source snapshots used by the transplant work
- local notes needed to reconstruct the experiment lineage without copying
  heavyweight checkpoints or run outputs

Excluded:
- trained checkpoints and policy pools
- generated results under `results/`
- final evaluation outputs
- W&B local run directories and `*.wandb` artifacts
- logs and local debug outputs
- nested `.git` metadata
- Python caches
- local archive and video exports
- `*.pt`, `*.pth`, `*.ckpt`, `*.pkl`, `*.pickle`, `*.npy`, and `*.npz`

Snapshot size on disk: about `141M`.

Regular files included: `5629`.

No file larger than `50M` was present in this snapshot.
