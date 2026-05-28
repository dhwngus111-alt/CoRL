# Snapshot 2026-05-28

This snapshot captures the local `/home/isl_jhoh/CoRL` workspace after updating the Risky multipath subgoal experiment configuration and launching paired HSP S1 runs.

Included:
- `HSP/`
- `risky/`
- `transplant/`
- workspace notes and journals
- policy/config artifacts needed for the transplant/HSP pipeline that were already present in the local workspace

Highlights:
- Risky subgoal puddle disable duration updated from 20 to 60 steps across subgoal layouts
- transplant HSP experiment defaults updated to `episode_length=200` and `subgoal_disable_steps=60`
- Risky subgoal timer tick behavior adjusted so newly activated puddle timers are not decremented in the same transition
- `risky_multipath_subgoal_tomato` layout included
- HSP hidden utility notes and launch commands recorded for:
  - GPU 0 Risky-aware hidden utility run
  - GPU 3 HSP-style comparison run with Risky-specific weights fixed to zero
- `transplant/tests/test_risky_pickup_mapping.py` passing with the new subgoal timer assertion

Excluded:
- `test/hsp_many_orders/`
- large training outputs under `transplant/results/RiskyOvercooked/`
- HSP script run outputs under `HSP/hsp/scripts/results/`
- Risky study/eval generated outputs
- transplant eval outputs under `transplant/biased_eval/` and `transplant/final_eval/`
- local run logs under `transplant/logs/`
- local scratch captures under `transplant/scratch/`
- smoke render outputs under `transplant/results/`
- Risky legacy study model artifacts under `risky/src/study_1/models/`
- local debug logs such as `tea_debug.log`
- W&B local run directories
- nested `.git` metadata
- Python caches
- `.DS_Store`

Snapshot size before Git object packing: about `163M`.

No file larger than `50M` was present in this snapshot.
