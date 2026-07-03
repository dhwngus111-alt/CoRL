# HSP Risky Overcooked Transplant

This repository contains the code and notes used to connect HSP-style
population/adaptive training to the Risky Overcooked environment, plus the
CoMeDi transplant entrypoints built on top of that integration.

It intentionally excludes trained checkpoints, policy pools, generated results,
W&B artifacts, and other large experiment outputs. Those artifacts should be
stored separately.

## Contents

- `transplant/`: HSP-to-RiskyOvercooked adapter code, runners, training/eval
  entrypoints, scripts, README, and development notes.
- `CoMeDi_transplant/`: CoMeDi population/adaptive training and evaluation
  wrappers for the Risky Overcooked transplant stack.
- `HSP/`: local HSP source snapshot with the code changes needed by this
  transplant work.
- `risked_overcooked/`: Risky Overcooked environment source snapshot and custom
  layouts used by the transplant experiments.
- `environment.yml`: Python environment used on the source machine.

## Upstream CoMeDi

The original CoMeDi implementation is available from Stanford ILIAD:

```bash
git clone -b neurips_comedi --single-branch https://github.com/Stanford-ILIAD/Diverse-Conventions CoMeDi
```

The local source tree was checked against commit
`8d3251406f95e8d170ae1d8a7785a95da5960b20` on the `neurips_comedi` branch.

## Setup Sketch

```bash
conda env create -f environment.yml
conda activate corl
export CORL_ROOT=/path/to/this/repo
export HSP_ROOT=${CORL_ROOT}/HSP
export RISKED_ROOT=${CORL_ROOT}/risked_overcooked
export PYTHONPATH=${CORL_ROOT}:${HSP_ROOT}:${RISKED_ROOT}/src:${PYTHONPATH}
```

For CoMeDi transplant scripts:

```bash
cd CoMeDi_transplant
bash scripts/01_train_comedi_population.sh
bash scripts/02_train_comedi_adaptive.sh
bash scripts/03_eval_comedi_risky.sh
```

Most scripts expose their key knobs through environment variables in
`CoMeDi_transplant/scripts/common.sh` and `transplant/common.sh`.

## Artifact Policy

Do not commit generated model artifacts or run outputs. In particular, keep
these out of Git:

- `results/`
- `final_eval/`
- `logs/`
- `wandb/`
- `policy_pool/`
- `*.pt`, `*.pth`, `*.ckpt`, `*.pkl`

