# FBLeWM

Independent Forward / Backward Causal Latent Imaginer extension of official LeWM
(PushT / OGBCube / TwoRoom via the same Rich pipeline).

- **Keeps** official Encoder, action-conditioned Predictor, CEM, and JEPA+SIGReg training objective.
- **Adds** two detached single-step imaginers trained in the **same** run:
  - Forward `F(z) → z_next` (no action / goal / history)
  - Backward `B(z) → z_previous` (no current state / action)
- **Eval**: one checkpoint × base modes `{official, forward, backward}` × offsets `{25,50,75,100}` = **12 units**, plus optional F/B fusion modes (below).

Original LeWM at `/home/yuanben/WorldModel/LeWM` is not modified by this project. Data is reused via symlink / `LOCAL_DATASET_DIR`.

## Layout

| Path | Role |
|------|------|
| `fblewm.py` | Model API: encode / predict / imagine_* / rollout / get_cost |
| `planning.py` | Mode names + fusion helpers |
| `module.py` | Official modules + `CausalLatentImaginer` |
| `train.py` | Joint official + F + B training |
| `policy.py` | `compute_imagine_steps` + `FBWorldModelPolicy` |
| `eval.py` | Single mode / offset Hydra entry |
| `checkpoint_utils.py` | FBLeWM loader (rejects official LeWM ckpts) |
| `scripts/run_fblewm_pipeline.py` | Rich + Popen + tqdm staged runner |
| `scripts/eval_fblewm_matrix.py` | Matrix eval (base + fusion modes) |
| `config/train/`, `config/eval/` | Hydra configs |
| `tests/` | Contract tests (manual) |
| `paper/` | Overleaf/LaTeX draft (own git; see `paper/README.md`) |

Artifacts (created at runtime):

- `FBLeWM/.stable-wm/checkpoints/`
- `FBLeWM/logs/runs/<timestamp>_pusht/`
- `FBLeWM/outputs/eval/<run_id>/`
- `FBLeWM/.cache/huggingface/`

## Environment

```bash
conda activate lewm
cd /home/yuanben/WorldModel/FBLeWM
source scripts/env.sh
```

Optional deps (only if missing in `lewm`):

```bash
pip install -r scripts/requirements-fblewm.txt
```

`scripts/env.sh` locks `FBLEWM_ROOT`, `STABLEWM_HOME`, `HF_HOME`, reuses `/home/yuanben/WorldModel/LeWM/data`, and unsets `HF_ENDPOINT`.

## Cube / TwoRoom data (shared under LeWM/data; no server HF download)

Upload archives next to PushT under the **shared** data root
(`/home/yuanben/WorldModel/LeWM/data/`, see `LeWM/data/README_DATASETS.md`):

| Task | File | Server path |
|------|------|-------------|
| TwoRoom | `tworoom.tar.zst` | `LeWM/data/tworoom.tar.zst` |
| Cube | `cube_single_expert.tar.zst` | `LeWM/data/cube_single_expert.tar.zst` |

```bash
# After upload — joint train (B→z) then 16-cell eval
python scripts/run_fblewm_pipeline.py --task tworoom --only-stage train --skip-deps --epochs 10
python scripts/run_fblewm_pipeline.py --task tworoom --only-stage eval --skip-deps \
  --policy fblewm_tworoom/weights_epoch_10.pt \
  --eval-modes official,forward,backward,fusion_avg05

python scripts/run_fblewm_pipeline.py --task cube --only-stage train --skip-deps --epochs 10
python scripts/run_fblewm_pipeline.py --task cube --only-stage eval --skip-deps \
  --policy fblewm_cube/weights_epoch_10.pt \
  --eval-modes official,forward,backward,fusion_avg05
```

PushT CLI defaults are unchanged (`--task pusht`).

## Imagination schedule

```text
k = max((goal_offset - elapsed - plan_len) / action_block, 0)
plan_len = 25, action_block = 5
```

| offset | replan depths `k` |
|--------|-------------------|
| 25 | `[0]` |
| 50 | `[5, 0]` |
| 75 | `[10, 5, 0]` |
| 100 | `[15, 10, 5, 0]` |

- **Forward cost**: `MSE(F^k(P(z, a)), z_goal)`
- **Backward cost**: `MSE(P(z, a), B^k(z_goal))` with `B^k` computed once per env outside CEM
- At `k=0`, F/B are bypassed and match official CEM cost

## Fusion planning modes (eval-only)

Same `k` schedule. CEM still optimizes actions only.

| mode | cost |
|------|------|
| `fusion_avg05` | `0.5 C_F + 0.5 C_B` |
| `fusion_avg07` | `0.7 C_F + 0.3 C_B` |
| `fusion_ofb` | `(C_official + C_F + C_B) / 3` |
| `fusion_max` | `max(C_F, C_B)` |
| `fusion_min` | `min(C_F, C_B)` |
| `switch_remain` | if `(goal_offset - elapsed) > 50` use F else B |
| `switch_offset` | if `offset >= 100` use F else `fusion_avg05` |
| `meet` | `MSE(F^{k//2}(P), B^{k-k//2}(z_goal))` |

Shorthand `--modes fusion` expands to all seven.

## Manual commands (Agent did not run these)

### 1) Contract tests

```bash
conda activate lewm
cd /home/yuanben/WorldModel/FBLeWM
source scripts/env.sh

python -m pytest tests/test_model_contracts.py tests/test_gradient_isolation.py \
  tests/test_imagination_schedule.py tests/test_policy_state.py \
  tests/test_eval_fairness.py tests/test_progress_parser.py \
  tests/test_fusion_modes.py -q
```

Expected: all tests pass; no GPU required for these unit tests.

### 2) 10-epoch joint retrain (`fblewm_bp`, does not overwrite v1)

`--backward-target` selects the Backward objective (Official/Forward are unchanged):

- `pred` — unary `B(z_{t+1}) → p_t` (default PushT)
- `encoder` — unary `B(z_{t+1}) → z_t`
- `now` — conditional `g ← B(z_now, g)` in z-space (default TwoRoom v2)

Checkpoints go to **`fblewm_bp/`**; legacy **`fblewm/`** is protected and will not be overwritten.

```bash
conda activate lewm
cd /home/yuanben/WorldModel/FBLeWM
source scripts/env.sh

python scripts/run_fblewm_pipeline.py --task pusht --only-stage train \
  --skip-deps --epochs 10 \
  --train-run-name fblewm_bp \
  --backward-target pred
```

Expected products:

- Run logs under `logs/runs/<timestamp>_pusht/`
- Checkpoint `.stable-wm/checkpoints/fblewm_bp/weights_epoch_10.pt` (+ `config.json`)
- Legacy `.stable-wm/checkpoints/fblewm/` unchanged
- Train log shows `official` / `forward` / `backward` train & val losses

Eval the new run (after training):

```bash
python scripts/run_fblewm_pipeline.py --task pusht --only-stage eval --skip-deps \
  --policy fblewm_bp/weights_epoch_10.pt \
  --eval-modes official,forward,backward \
  --starts-manifest outputs/eval/20260811_025219_pusht/starts_manifest.json
```

### 3) Full 12-unit evaluation

```bash
python scripts/run_fblewm_pipeline.py --task pusht --only-stage eval \
  --skip-deps --policy fblewm/weights_epoch_10.pt
```

Expected products under `outputs/eval/<run_id>/`:

- `starts_manifest.json` (shared across all 12 units)
- `results.json`, `results.jsonl`, `summary.txt`
- `videos/<mode>/offset_<N>/`

Pipeline progress advances only on `eval i/12 DONE` (not on `START`). Non-zero exit on failure.

### 4) Fusion matrix eval via Rich pipeline (7 × 4 = 28 units)

**Important:** always `source scripts/env.sh` inside FBLeWM so `STABLEWM_HOME` is
`FBLeWM/.stable-wm` (not a leftover LeWM path).

```bash
conda activate lewm
cd /home/yuanben/WorldModel/FBLeWM
source scripts/env.sh
echo "$STABLEWM_HOME"   # must be .../FBLeWM/.stable-wm

python scripts/run_fblewm_pipeline.py --task pusht --only-stage eval --skip-deps \
  --policy fblewm/weights_epoch_10.pt \
  --eval-modes fusion \
  --eval-offsets 25,50,75,100 \
  --starts-manifest outputs/eval/20260811_025219_pusht/starts_manifest.json
```

Subset via the same Rich runner:

```bash
python scripts/run_fblewm_pipeline.py --task pusht --only-stage eval --skip-deps \
  --policy fblewm/weights_epoch_10.pt \
  --eval-modes fusion_avg05,fusion_avg07,fusion_max,fusion_min,meet \
  --starts-manifest outputs/eval/20260811_025219_pusht/starts_manifest.json
```

### Optional smoke stages

```bash
python scripts/run_fblewm_pipeline.py --task pusht --only-stage model_smoke --skip-deps
python scripts/run_fblewm_pipeline.py --task pusht --only-stage gpu_gate --skip-deps
```

## Implementation status

**Implemented in this tree**

- Scaffold under `/home/yuanben/WorldModel/FBLeWM` (LeWM left as-is for this project)
- Official backbone + independent F/B imaginers
- Same-run training with detach before Imaginer losses
- Policy elapsed / flush / terminated + dynamic `k`
- Cost fusion / stage switch / meet-in-the-middle planning modes
- Matrix eval + Rich/Popen pipeline + tests + this README

**Not executed by the Agent**

- `pytest`
- dependency install
- GPU smoke / training / evaluation / fusion matrix
- dataset download or conversion

Do not treat this README as evidence that tests or training already passed.
