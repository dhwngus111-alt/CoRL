# Risky Multipath HSP Transplant

This directory connects the original HSP baseline to the Risky Overcooked
`risky_multipath` layout without editing either original repository:

- `/home/isl_jhoh/CoRL/HSP` is imported as the algorithm implementation.
- `/home/isl_jhoh/CoRL/risky` is imported as the Risky environment.
- all new wrappers, scripts, configs, checkpoints, and eval logs live under
  `/home/isl_jhoh/CoRL/transplant`.

## Flow

```bash
cd /home/isl_jhoh/CoRL

# Import/env smoke check and initial policy_config YAML files.
bash transplant/scripts/00_check_env.sh

# MEP S1 population training, then checkpoint extraction.
bash transplant/scripts/01_train_mep_s1_risky.sh
python transplant/scripts/02_extract_mep_s1_risky.py

# HSP S1 biased-policy training, then checkpoint extraction.
SEED_MAX=36 bash transplant/scripts/03_train_hsp_s1_risky.sh
python transplant/scripts/04_extract_hsp_s1_risky.py

# Evaluate biased policies and greedily select diverse HSP policies.
SEED_MAX=36 bash transplant/scripts/05_eval_events_risky.sh
python transplant/scripts/06_greedy_select_risky.py --k 18

# HSP S2 adaptive training, extraction, and final eval.
bash transplant/scripts/07_train_hsp_s2_risky.sh
python transplant/scripts/08_extract_hsp_s2_risky.py
bash transplant/scripts/09_eval_hsp_risky.sh
```

For a small pipeline check, keep the normal runtime path and override only the
training budget, for example `NUM_ENV_STEPS=120000`. Greedy-selection event
evaluation uses the HSP `eval_events.sh` style `EVAL_EPISODES=100` and
`N_EVAL_ROLLOUT_THREADS=100` defaults; other eval paths follow HSP parser
defaults, `EVAL_EPISODES=32` and `N_EVAL_ROLLOUT_THREADS=1`.

## Adapter Boundary

`transplant/adapters/risky_overcooked_env.py` is the main transplant point. It:

- maps HSP action indices `0..5` to Risky `Action.INDEX_TO_ACTION`;
- returns PPO lossless observations and centralized critic `share_obs`;
- reports all six actions as available in v1;
- maps Risky raw events into the HSP hidden-utility category schema;
- keeps HSP hidden utility weights at `len(HIDDEN_UTILITY_KEYS) + 1` dimensions.
