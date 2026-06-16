# Snapshot 2026-06-16

This snapshot captures the local `/home/isl_jhoh/CoRL` workspace as a code-focused pipeline backup.

Included:
- `HSP/` pipeline and environment code
- `risky/` Risky Overcooked code and local code/config edits
- `risked_overcooked/` newly cloned environment-only package from `dhwngus111-alt/risked_overcooked`
- `transplant/` pipeline code, adapters, runners, scripts, tests, notes, and policy config directories
- top-level workspace notes and lightweight configuration files

Highlights:
- Added the new `risked_overcooked` package snapshot.
- Preserved current Risky Overcooked environment/subgoal code context.
- Preserved transplant/HSP pipeline code while excluding generated experiment outputs.

Excluded:
- W&B local run directories
- generated training/evaluation results
- local scratch/log directories
- policy/model weight artifacts such as `.pt`, `.pth`, and `.ckpt`
- Python caches and bytecode
- nested `.git` metadata
- local debug logs such as `tea_debug.log`
- large scratch/output folders not needed for pipeline code recovery

Snapshot size before Git object packing: about `108M`.

No file larger than `50M` was present in this snapshot.
