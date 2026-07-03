# Risked Overcooked HSP pipeline

This directory connects the HSP implementation in `HSP/` to the Risked
Overcooked environment in `risked_overcooked/`. Training code, adapters,
configuration builders, extraction steps, evaluations, and tests live under
`transplant/`. Generated checkpoints and experiment outputs are intentionally
not part of the reproducible source snapshot.

## Environment setup

Run these commands from the snapshot's `CoRL` directory:

```bash
conda env create -f environment.yml
conda activate corl
python -m pip install -e ./HSP -e ./risked_overcooked

# Required only for online W&B logging.
wandb login
```

The recorded environment uses Python 3.10.20 and PyTorch 2.6.0 with CUDA
12.4 wheels. A compatible NVIDIA driver is required for training. The scripts
refuse to silently fall back to CPU when CUDA was requested.

`transplant/common.sh` uses the active `python` by default on a new machine.
Set `PYTHON=/path/to/python` when a different interpreter is required. W&B can
be disabled with `USE_WANDB=0`; no API keys or credentials are stored here.
Optional high-memory Slack alerts use `SLACK_WEBHOOK_URL` and remain disabled
when that variable is unset.

## Supported layouts

- `risky_dualpath_subgoal`
- `risky_mixed_coordination_subgoal`
- `risky_multipath_subgoal`

The common defaults are episode length 200, puddle disable duration 60,
slip probability 0.4, training time cost -0.3, 12 MEP Stage-1 policies, and
36 HSP Stage-1 seeds. Evaluation forces `time_cost=0`.

## Reproduce the pipeline

First verify imports, the layout, and generated policy configuration:

```bash
USE_WANDB=0 bash transplant/scripts/00_check_env.sh
```

Run all three layouts and all stages:

```bash
bash transplant/scripts/run_all_risked_layouts.sh
```

For one layout, set `LAYOUT` and execute the stages in order:

```bash
export LAYOUT=risky_dualpath_subgoal

bash transplant/scripts/00_check_env.sh
bash transplant/scripts/01_train_mep_s1_risky.sh
python transplant/scripts/02_extract_mep_s1_risky.py --layout "$LAYOUT"
bash transplant/scripts/03_train_hsp_s1_risky.sh
python transplant/scripts/04_extract_hsp_s1_risky.py --layout "$LAYOUT"
bash transplant/scripts/05_eval_events_risky.sh
python transplant/scripts/06_greedy_select_risky.py --layout "$LAYOUT" --k 18
bash transplant/scripts/07_train_hsp_s2_risky.sh
python transplant/scripts/08_extract_hsp_s2_risky.py --layout "$LAYOUT"
bash transplant/scripts/09_eval_hsp_risky.sh
```

For a short pipeline check, reduce only the runtime budget, for example:

```bash
USE_WANDB=0 NUM_ENV_STEPS=120000 N_ROLLOUT_THREADS=4 \
  bash transplant/scripts/01_train_mep_s1_risky.sh
```

Stage 03 trains all HSP seeds sequentially in one W&B run. Each seed gets its
own `seedXX` section and, by default, three deterministic final-policy GIFs in
`seedXX/final_gifs`.

## Generated outputs

The following paths are created by the pipeline and are excluded from source
snapshots:

- `transplant/results/`: training runs and checkpoints;
- `transplant/policy_pool/**/*.pt`: extracted and trained policy checkpoints;
- `transplant/biased_eval/`: Stage-1 event evaluation outputs;
- `transplant/final_eval/`: final tables, manifests, text logs, and GIFs;
- `wandb/` and nested `wandb/`: local W&B run data.

The snapshot retains reusable population YAML/config files and test fixtures,
but never includes trained `.pt`, `.pth`, or `.ckpt` files.

## Validation

```bash
PYTHONPATH="$PWD:$PWD/HSP:$PWD/risked_overcooked/src" \
  python -m unittest \
    transplant.tests.test_risky_pickup_mapping \
    transplant.tests.mep_test.test_mep_vs_random
```

The adapter boundary is `transplant/adapters/risky_overcooked_env.py`. It maps
HSP action indices to Risked actions, builds actor/critic observations, maps
Risked events into the HSP hidden-utility schema, and controls evaluation GIF
capture.
