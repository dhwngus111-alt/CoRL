# Snapshot 2026-07-03

This snapshot captures the lightweight HSP/Risky Overcooked/CoMeDi transplant
workspace prepared for migration to another server.

Included:
- `CoMeDi_transplant/`
- `transplant/`
- `HSP/`
- `risked_overcooked/`
- `environment.yml`
- root README and artifact exclusion rules
- transplant README files and Korean development journals/notes

Highlights:
- CoMeDi transplant training/evaluation entrypoints for Risky Overcooked
- HSP-to-RiskyOvercooked adapter, runners, scripts, smoke checks, and event eval helpers
- local HSP source snapshot with transplant-related changes
- Risky Overcooked source snapshot with custom subgoal layouts
- notes needed to reconstruct the experiment lineage without copying heavy checkpoints

Excluded:
- trained checkpoints and policy pools
- generated results under `results/`
- final evaluation outputs
- W&B local run directories
- logs and large local test outputs
- nested `.git` metadata
- Python caches
- `*.pt`, `*.pth`, `*.ckpt`, `*.pkl`, and `*.pickle`

Snapshot size before Git object packing: about `11M`.

No file larger than `10M` was present in this snapshot.
