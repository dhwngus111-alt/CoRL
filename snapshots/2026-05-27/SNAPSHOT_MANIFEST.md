# CoRL Snapshot Manifest

- Snapshot date: 2026-05-27
- Source path: `/home/isl_jhoh/CoRL`
- Target repository: `dhwngus111-alt/CoRL`
- Snapshot path: `snapshots/2026-05-27/CoRL`
- Original source size: about 48G
- GitHub snapshot size: about 181M
- Snapshot file count: 957
- Files larger than 90M in snapshot: none

## Excluded From Snapshot

The following paths were excluded because they are git metadata, generated caches,
experiment tracking data, or large generated training outputs:

- `.git/`
- `HSP/.git/`
- `risky/.git/`
- `**/__pycache__/`
- `*.pyc`
- `wandb/`
- `test/`
- `transplant/results/`
- `HSP/hsp/scripts/results/`

## Notes

The original workspace contains nested git repositories under `HSP/` and `risky/`.
Their working-tree contents were copied into this snapshot, while their internal
`.git` directories were omitted so GitHub stores the actual files instead of
submodule pointers.

The `test/` workspace was omitted because this snapshot is intended to preserve
the transplant-related work and supporting source trees.
