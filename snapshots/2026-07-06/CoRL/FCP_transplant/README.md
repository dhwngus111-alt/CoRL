# FCP (Fictitious Co-Play) on Risked Overcooked

This directory adds the **FCP baseline** (Strouse et al., NeurIPS 2021) to the
Risked Overcooked environment, following the *HSP repository's* FCP
implementation rather than the FCP paper directly.

It is a thin orchestration overlay: it **reuses `transplant/` verbatim** (the
proven port of the HSP MAPPO + population machinery to Risked Overcooked) and
**does not modify `HSP/` or `transplant/`**. Only the FCP-specific pipeline
lives here. Results and checkpoints are written under `FCP_transplant/` so they
stay separate from the HSP transplant run.

## How FCP maps onto the HSP machinery

In the HSP repo, FCP is not a standalone method — it reuses the self-play (SP)
Stage-1 trainer and the adaptive Stage-2 trainer (`train_fcp` is a stub;
`train_overcooked_adaptive.py` runs the best-response). We mirror that here:

| Stage | HSP repo original | This overlay |
|---|---|---|
| **S1: partner pool** | `train_overcooked_sp.py` → `runner.run()` (**shared** self-play, 12 seeds; no `--share_policy`) | `train_fcp_s1_bundle.py` → `transplant.train_risky_hsp.train_hsp_s1` with `use_hsp=False`, `random_index=False`, `share_policy=True` (same shared `runner.run()` self-play loop) |
| **checkpoint filter** | `extract_sp_S1_models.py` (init / mid=½·final reward / final) | `extract_fcp_s1.py` (one `actor_periodic_*.pt` per seed: `init` = v0, `mid` = closest to ½ final `ep_sparse_r`, `final` = last) |
| **S2 pool** | `fcp/s2/train.yml` (36 frozen + `fcp_adaptive`) | written by `extract_fcp_s1.py` |
| **S2: best-response** | `train_overcooked_adaptive.py --stage 2 --exp fcp` | `transplant.train_risky_adaptive --algorithm_name adaptive --stage 2 --adaptive_agent_name fcp_adaptive` (uniform sampling) |

> **Shared vs separated self-play.** HSP's `--share_policy` flag is `store_false`
> (default `True`), so HSP FCP's SP script — which does *not* pass it — trains a
> **shared** policy (one network plays both slots). We keep that: the S1 bundle
> hard-sets `share_policy=True` and the 01 script does not pass `--share_policy`.
> The shared runner saves `actor_periodic_*.pt`; the separated runner (used by
> HSP-S1 biased play) would instead save `actor_agent{0,1}_periodic_*.pt`.

12 self-play seeds × 3 checkpoints (init/mid/final) = **36 frozen partners**.
The adaptive agent is trained as their **uniformly-sampled best response**
(no prioritized-sampling flags — exactly HSP's FCP config).

## Hyperparameters (follow HSP's FCP baseline)

**Algorithm** — HSP config defaults, unchanged: `lr = critic_lr = 5e-4`,
`gamma = 0.99`, `gae_lambda = 0.95`, `clip_param = 0.2`, `entropy_coef = 0.01`,
`ppo_epoch = 15`, `num_mini_batch = 1`, `max_grad_norm = 10`, `hidden_size = 64`,
`layer_N = 1`, `use_valuenorm = True`, CNN `32,3,1,1 64,3,1,1 32,3,1,1`.
Partners are MLP MAPPO (`--algorithm_name mappo --use_recurrent_policy`, which —
because HSP's flag is `store_false` — disables recurrence, matching
`train_overcooked_sp.sh`). The Stage-2 adaptive agent is recurrent (`rnn`), as in
HSP's `train_overcooked_adaptive.py`.

**Stage 1** — `seed 1..12`, `n_rollout_threads 100`, `num_env_steps 1e7`,
`save_interval 25`, `reward_shaping_horizon 1e8`.

**Stage 2** — `n_rollout_threads 300`, `num_env_steps 1e8`, `save_interval 20`,
`population_size 36`, `train_env_batch 1`, `--use_agent_policy_id`, uniform
sampling.

**Environment** — the risked-specific knobs (`p_slip = 0.4`, `time_cost = -0.3`,
`subgoal_disable_steps = 60`, `episode_length = 200`, `distance_shaping_rew`,
`subgoal_press_rew`, `handoff_shaping`) are **inherited from the shared
transplant study config**, not from HSP's original Overcooked, so FCP is
compared against MEP/HSP on an identical environment. Evaluation forces
`time_cost = 0`. These live in `common.sh` and are overridable via environment
variables.

## Environment / interpreter

`common.sh` defaults `PYTHON` to this server's project conda env,
`/home/isllab0213/miniconda3/envs/risky_overcooked/bin/python` (torch
2.4.1+cu124), so no export is needed here. On another machine, override it:

```bash
export PYTHON=/path/to/your/env/bin/python
```

`common.sh` derives all other paths from its own location (CoRL root, `HSP/`,
`risked_overcooked/src`, `transplant/compat`) and points
`TRANSPLANT_OUTPUT_ROOT` / `POLICY_POOL` at `FCP_transplant/`, so no absolute
paths need editing.

## Run the pipeline

```bash
cd /scratch/isllab0213/isl_jhoh/CoRL
export LAYOUT=risky_dualpath_subgoal          # or risky_mixed_coordination_subgoal, risky_multipath_subgoal
# The bash sub-scripts source common.sh internally, but the two inline
# extraction steps below run in *this* shell, so export PYTHON here too.
# (Same value common.sh would resolve; override for another machine.)
export PYTHON=/home/isllab0213/miniconda3/envs/risky_overcooked/bin/python

# 0) verify imports, build policy_config.pkl, env smoke check
bash FCP_transplant/scripts/00_check_env.sh

# 1) Stage 1: 12 self-play seeds -> checkpoints
bash FCP_transplant/scripts/01_train_fcp_s1_risky.sh

# 2) extract init/mid/final -> 36 frozen partners + fcp/s2/train.yml
#    Add --require-reward-history to FAIL rather than fall back if the exact
#    half-final-reward 'mid' criterion is unavailable (see note below).
"$PYTHON" FCP_transplant/scripts/02_extract_fcp_s1_risky.py --layout "$LAYOUT" --require-seeds 12

# 3) Stage 2: best-response adaptive agent (uniform sampling)
bash FCP_transplant/scripts/03_train_fcp_s2_risky.sh

# 4) extract trained fcp_adaptive -> fcp/s2/eval.yml
"$PYTHON" FCP_transplant/scripts/04_extract_fcp_s2_risky.py --layout "$LAYOUT"

# 5) evaluate fcp_adaptive vs each pool partner (deliveries + event tables)
bash FCP_transplant/scripts/05_eval_fcp_risky.sh
```

Or the whole thing for one layout (`run_all_fcp.sh` sources `common.sh` once so
every stage — including the inline extraction steps — uses the same
`common.sh`-resolved interpreter):

```bash
LAYOUT=risky_dualpath_subgoal bash FCP_transplant/scripts/run_all_fcp.sh
```

For strict HSP-FCP fidelity (exact half-final-reward `mid` selection), make
extraction fail if the local reward curve is missing:

```bash
REQUIRE_REWARD_HISTORY=1 \
  LAYOUT=risky_dualpath_subgoal bash FCP_transplant/scripts/run_all_fcp.sh
```

### 'mid' checkpoint selection and W&B

HSP's FCP picks the mid checkpoint as the one closest to **half the final
episode sparse reward**. FCP Stage 1 now writes `fcp_s1_sparse_history.csv`
under each seed local run directory regardless of W&B mode, and the extractor
uses that CSV as the canonical reward curve. TensorBoard events are only a
legacy fallback.

- `mid` no longer requires `USE_WANDB=0`; W&B can stay on for monitoring.
- W&B history is not used for extraction, because it depends on network/account
  state.
- If neither local CSV nor TensorBoard history exists, `mid` falls back to the
  temporal-middle checkpoint and prints a `WARNING`. Pass
  `--require-reward-history` to make this a hard error instead.

### Short wiring check (reduced budget)

```bash
USE_WANDB=0 NUM_ENV_STEPS=120000 N_ROLLOUT_THREADS=4 SEED_MAX=2 \
  bash FCP_transplant/scripts/01_train_fcp_s1_risky.sh
```

## Files

```
FCP_transplant/
  bootstrap.py              # points output roots at FCP_transplant, wires transplant/hsp/risked paths
  common.sh                 # HSP-FCP hyperparameters + shared risky env config; derives all paths
  train_fcp_s1_bundle.py    # Stage 1: 12-seed plain self-play (reuses transplant train_hsp_s1, use_hsp=False)
  extract_fcp_s1.py         # init/mid/final per seed -> fcp/s1 + fcp/s2/train.yml
  extract_fcp_s2.py         # trained fcp_adaptive -> fcp/s2/eval.yml
  scripts/
    00_check_env.sh
    01_train_fcp_s1_risky.sh
    02_extract_fcp_s1_risky.py
    03_train_fcp_s2_risky.sh
    04_extract_fcp_s2_risky.py
    05_eval_fcp_risky.sh
    run_all_fcp.sh
```

## Generated outputs (not source)

- `FCP_transplant/results/` — training runs and checkpoints
- `FCP_transplant/policy_pool/<layout>/fcp/{s1,s2}/*.pt` — extracted / trained policies
- `FCP_transplant/final_eval/` — evaluation score and event tables
