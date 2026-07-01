# Risky Overcooked MEP baseline pipeline

This directory contains thin wrappers for applying the MEP baseline to the
Risked Overcooked transplant setup. It reuses the implementation under
`transplant/`, reads only source MEP policies from the existing transplant
artifacts, and writes MEP-specific outputs here.

The pipeline does not retrain MEP Stage 1. It prepares a local
`MEP_transplant/policy_pool` from already trained MEP Stage-1 policies created
during the HSP pipeline, then trains a MEP adaptive policy with prioritized
sampling. The source HSP/transplant run may contain 12 MEP policies, but the
MEP-paper baseline pool uses only the first five source policies:
`mep1..mep5`. For each of those five policies, the existing init/mid/final
checkpoints are copied, giving 15 total Stage-2 partner policies.

Supported layouts:

- `risky_dualpath_subgoal`
- `risky_mixed_coordination_subgoal`
- `risky_multipath_subgoal`

Default run sequence for one layout:

```bash
export LAYOUT=risky_multipath_subgoal

python MEP_transplant/scripts/01_prepare_mep_s1_pool.py --layout "$LAYOUT"
bash MEP_transplant/scripts/02_train_mep_s2_risky.sh
python MEP_transplant/scripts/03_extract_mep_s2_risky.py --layout "$LAYOUT"
bash MEP_transplant/scripts/04_eval_mep_risky.sh
```

To run all three configured layouts:

```bash
bash MEP_transplant/scripts/run_all_mep_layouts.sh
```

Generated files are written under `MEP_transplant/policy_pool`,
`MEP_transplant/results`, and `MEP_transplant/final_eval`.

Default input root:

- `MEP_SOURCE_POLICY_POOL=/home/isl_jhoh/CoRL/transplant/policy_pool`
