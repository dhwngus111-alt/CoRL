# HSP many_orders paper-style pipeline

This directory runs the HSP pipeline on `many_orders` using the original HSP code under
`/home/isl_jhoh/CoRL/HSP`. The original HSP source is only imported or linked; new local
outputs are written under `/home/isl_jhoh/CoRL/test/hsp_many_orders`.

## Run

Set basic environment variables:

```bash
export USER_NAME=isl_jhoh
export GPU=1
```

`WANDB_ENTITY` is optional. If not set, scripts try to resolve it from your wandb login (`~/.netrc`)
and fall back to `USER_NAME`.
For this account, the upload entity is `dhwngus41-daegu-gyeongbuk-institute-of-science-technology`;
if `WANDB_ENTITY=dhwngus41` is set, `common.sh` maps it to that workspace slug.
The W&B project is `Overcooked`.
`WANDB__SERVICE_WAIT` and `WANDB_INIT_TIMEOUT` default to `300` seconds so slow W&B startup
does not abort runs.

Then run the full pipeline:

```bash
/home/isl_jhoh/CoRL/test/hsp_many_orders/run_full_pipeline.sh
```

For separate GPUs:

```bash
export GPU_MEP_S1=1
export GPU_HSP_S1=1
export GPU_EVAL_EVENTS=1
export GPU_HSP_S2=1
```

## Steps

1. `00_check_env.sh`: checks paths and Python dependencies.
2. `01_train_mep_s1_many_orders.sh`: trains MEP S1, needed for the HSP S2 pool.
3. `02_extract_mep_s1_many_orders.py`: copies `mep1~mep6` init/mid/final checkpoints.
4. `03_train_hsp_s1_many_orders.sh`: trains 36 HSP biased policies with Table 14 random-search values.
5. `04_extract_hsp_s1_many_orders.py`: copies HSP S1 `w0/w1` actors into the policy pool.
6. `05_eval_events_many_orders.sh`: evaluates event counts for HSP policy filtering.
7. `06_greedy_select_many_orders.sh`: selects 18 diverse HSP policies and writes S2 `train.yml`.
8. `07_train_hsp_s2_many_orders.sh`: trains `hsp_adaptive` against the 36-policy pool.
9. `08_extract_hsp_s2_many_orders.py`: copies final `hsp_adaptive.pt` and writes S2 `eval.yml`.
10. `09_eval_hsp_many_orders_wandb.sh`: logs final partner metrics and 3 GIFs per partner to wandb.

## Local Outputs

New training checkpoints and W&B run files are stored here:

- `results/Overcooked/many_orders/mep/mep-S1/`
- `results/Overcooked/many_orders/mappo/hsp-S1/`
- `results/Overcooked/many_orders/adaptive/hsp-S2/`

Extracted policy files are stored in `policy_pool/many_orders/...`.

Event-count evaluation text files are stored in `biased_eval/many_orders/`.

Final evaluation local artifacts are stored in `final_eval/`:

- `final_eval/metrics/partner_performance.csv`
- `final_eval/figures/sparse_reward_by_partner.png`
- `final_eval/figures/shaped_reward_by_partner.png`

## Notes

The paper-style HSP S2 pool is:

- 18 MEP checkpoints: `mep1~mep6` x `init/mid/final`
- 18 HSP biased policies selected from 36 HSP S1 candidates

The `stubs/` directory is intentionally added to `PYTHONPATH` by `common.sh`. It only patches
old dependency assumptions in the original HSP code: duplicate Gym registration, `np.Inf`, and
notebook-only `ipywidgets` imports.

Final wandb evaluation logs:

- `performance/{partner}/...`
- `gifs/{partner}/episode_1..3`
- `overall/performance_table`
- `overall/sparse_reward_by_partner`
- `overall/shaped_reward_by_partner`
