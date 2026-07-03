# Snapshot 2026-07-03

This snapshot captures the lightweight `/home/isl_jhoh/CoRL` workspace prepared
for migration to another server, excluding the GAMMA line of work and generated
training artifacts.

Included:
- `CoMeDi/`
- `CoMeDi_transplant/`
- `transplant/`
- `HSP/`
- `MEP/`
- `MEP_transplant/`
- `risked_overcooked/`
- `드러운거/` lightweight scripts, stubs, and notes
- `environment.yml`
- root README, prompt summary, and artifact exclusion rules
- `Comedi.pdf`, `HSP.pdf`, and `mep.pdf`
- transplant README files and Korean development journals/notes

Highlights:
- upstream CoMeDi source snapshot, excluding its nested Git metadata
- CoMeDi transplant training/evaluation entrypoints for Risky Overcooked
- HSP-to-RiskyOvercooked adapter, runners, scripts, smoke checks, and event eval helpers
- MEP source and RiskyOvercooked transplant scripts
- local HSP source snapshot with transplant-related changes
- Risky Overcooked source snapshot with custom subgoal layouts
- notes needed to reconstruct the experiment lineage without copying heavy checkpoints

Excluded:
- `GAMMA/`, `GAMMA_transplant/`, and `gamma.pdf`
- trained checkpoints and policy pools
- generated results under `results/`
- final evaluation outputs
- W&B local run directories
- logs and large local test outputs
- local archive exports such as `*.zip` and `*.tar.gz`
- nested `.git` metadata
- Python caches
- `*.pt`, `*.pth`, `*.ckpt`, `*.pkl`, `*.pickle`, `*.npy`, and `*.npz`

Snapshot size before Git object packing: about `128M`.

No file larger than `50M` was present in this snapshot.
