# Multipath MEP vs Uniform Random Evaluation

This experiment evaluates the 12 final MEP policies from
`risky_multipath_subgoal` against a uniform-random partner.

## Run

```bash
cd /home/isl_jhoh/CoRL

CUDA_VISIBLE_DEVICES=3 \
bash transplant/tests/mep_test/run_mep_vs_random.sh
```

W&B settings are kept in `transplant/tests/mep_test/config.sh`. The default
project is `mep_random_test`, so W&B creates/uses that separate project.

CUDA is attempted first. If CUDA initialization or tensor allocation fails,
this evaluation prints the failure and explicitly falls back to CPU. This does
not change the fail-fast behavior of the training entrypoints.

For a local/offline W&B run:

```bash
WANDB_MODE=offline \
bash transplant/tests/mep_test/run_mep_vs_random.sh
```

## Evaluation settings

- layout: `risky_multipath_subgoal`
- policies: final checkpoints for `mep1` through `mep12`
- episodes per policy: 5
- MEP roles: agent0 for episodes 1/3/5, agent1 for episodes 2/4
- MEP action: deterministic
- partner action: uniform random over all six actions
- episode length: 200
- slip probability: 0.4
- subgoal disable steps: 60
- time cost: 0

## Outputs

Each invocation creates a new timestamped directory:

```text
transplant/tests/mep_test/outputs/run_YYYYMMDD_HHMMSS/
├── config/mep_final_eval.yml
├── csv/episodes.csv
├── csv/policy_summary.csv
├── csv/gif_manifest.csv
├── gifs/mep1/...mep12/
├── wandb/
└── eval.log
```

The W&B run uses project `mep_random_test`. Sections
`mep1/` through `mep12/` contain the five GIFs, an episode table, mean sparse
reward, mean delivery count, and MEP/random delivery credit.

## Tests

```bash
/home/isl_jhoh/miniconda3/envs/corl/bin/python \
  -m unittest transplant.tests.mep_test.test_mep_vs_random
```
