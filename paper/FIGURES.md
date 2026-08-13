# FBLeWM figures and numbers (writing lookup)

Last updated: 2026-08-13. All success rates are **%**, `n=50`, `seed=42`, same `starts_manifest` within each task. Checkpoint is always **epoch 10**. Backward in these figures is unary **B→z** (`loss.backward.target=encoder`), not `pred` and not `now`.

**Claim for the paper:** F/B give the short-horizon CEM a longer-range scoring view. Do **not** claim \(F^k(p)\approx z_{t+k}\) or full latent-norm alignment.

**Not in these figures yet:** Cube; TwoRoom v2 (`now`); PushT `fblewm_bp` (B→p). Numbers for `fblewm_bp` are in the appendix table below.

---

## 1. Protocol (shared)

| Item | Value |
|------|--------|
| CEM default | `horizon=5`, `receding=5`, `action_block=5` → `plan_len=25` env steps |
| Imagination depth | \(k=\max((o - t - 25)/5,\,0)\) for goal offset \(o\), elapsed \(t\) |
| Cost, Official | \(\|P_{25}-z_g\|^2\) |
| Cost, Forward | \(\|F^k(P_{25})-z_g\|^2\) |
| Cost, Backward | \(\|P_{25}-B^k(z_g)\|^2\) (unary) |
| Cost, Fusion (`avg05`) | \(0.5\,C_F+0.5\,C_B\) |
| \(k=0\) | F/B bypassed; must match Official |

Offset 25 under default CEM has \(k=0\) at the first replan, so Official / Forward / Backward coincide (PushT 82/82/82; TwoRoom 90/90/90 except Backward 92 by one episode).

---

## 2. Figure files

| File | Use in writing | Task |
|------|----------------|------|
| [`fig_avg05_vs_baselines.png`](../eval/20260811_061335_pusht/fig_avg05_vs_baselines.png) | Main PushT success vs offset | PushT |
| [`fig_avg05_vs_baselines.png`](../eval/20260813_022142_tworoom/fig_avg05_vs_baselines.png) | Main TwoRoom success vs offset | TwoRoom |
| [`fig_pusht_backward_zz.png`](fig_pusht_backward_zz.png) | Train: Official / F / B val loss + \(\\|z\\|\) (aligned) | PushT |
| [`fig_tworoom_backward_zz.png`](fig_tworoom_backward_zz.png) | Train: same layout; SIGReg warmup + F/B output-norm gap | TwoRoom |

`zz` in the train filenames = unary \(B(z)\to z\). Do not overwrite these when later plotting `zp` or `now`.

---

## 3. PushT success (main table)

**Checkpoint:** `fblewm/weights_epoch_10.pt`  
**Starts:** `9d47d5e78eb0b98693bdb739e6e107fca7b52a7fa5d38189e418c242d7d2a703`  
**Default-CEM baselines** (`outputs/eval/20260811_025219_pusht/`):

| mode | 25 | 50 | 75 | 100 |
|------|---:|---:|---:|----:|
| Official | 82 | 44 | 22 | 12 |
| Forward | 82 | 62 | 42 | 28 |
| Backward | 82 | 70 | 34 | 10 |
| Fusion avg05 | 82 | 78 | 50 | 18 |

Counts: Official 41/22/11/6; Forward 41/31/21/14; Backward 41/35/17/5; avg05 41/39/25/9. Fusion 50–100 from `20260811_061335_pusht/`; Official/F/B from `20260811_025219_pusht/`. Same starts.

**Δ vs Official (percentage points):**

| mode | 25 | 50 | 75 | 100 |
|------|---:|---:|---:|----:|
| Forward | 0 | +18 | +20 | +16 |
| Backward | 0 | +26 | +12 | −2 |
| Fusion avg05 | 0 | +34 | +28 | +6 |

**What to say:** On PushT, Backward helps most at offset 50; Forward is more stable at 75/100; Fusion avg05 is the strongest mid-range mix. Offset 100 is still hard (Fusion 18 vs Official 12).

### PushT figure vs the table (caption must mention this)

The plotted `fig_avg05_vs_baselines.png` uses **CEM horizon=2** (`plan_len=10`) **only at offset 25**, so F/B are active at the short goal. Other offsets are default CEM=5.

| mode | 25 (CEM=2) | 50 (CEM=5) | 75 (CEM=5) | 100 (CEM=5) |
|------|-----------:|-----------:|-----------:|------------:|
| Official | 72 | 44 | 22 | 12 |
| Forward | 86 | 62 | 42 | 28 |
| Backward | 76 | 70 | 34 | 10 |
| Fusion avg05 | 84 | 78 | 50 | 18 |

Offset-25 CEM=2 sources: `20260811_080750_pusht/` (O/F/B) and `20260811_081323_pusht/` (avg05). Same starts hash as the default-CEM table.

If the paper’s protocol is CEM=5 everywhere, **quote the 82/82/82 row at offset 25**, not 72/86/76. Use the CEM=2 panel only if you explicitly discuss enabling F/B at offset 25.

---

## 4. TwoRoom success (main table)

**Checkpoint:** `fblewm_tworoom/weights_epoch_10.pt`  
**Starts:** `55d795562181c8f5eb12def43fb995cedd0d45c909fd1f39e38518155e16ad80`  
**Eval dir:** `outputs/eval/20260813_022142_tworoom/`  
**CEM=5 at all offsets** (matches the figure).

| mode | 25 | 50 | 75 | 100 |
|------|---:|---:|---:|----:|
| Official | 90 | 50 | 32 | 18 |
| Forward | 90 | 90 | 84 | 66 |
| Backward | 92 | 54 | 40 | 24 |
| Fusion avg05 | 90 | 76 | 58 | 32 |

Counts: Official 45/25/16/9; Forward 45/45/42/33; Backward 46/27/20/12; avg05 45/38/29/16.

**Δ vs Official:**

| mode | 25 | 50 | 75 | 100 |
|------|---:|---:|---:|----:|
| Forward | 0 | +40 | +52 | +48 |
| Backward | +2 | +4 | +8 | +6 |
| Fusion avg05 | 0 | +26 | +26 | +14 |

Paired vs Official, Forward@50: 21 episodes saved, 1 lost (same 50 starts). Official@25 and Forward@25 fail on the **same five** episodes.

**What to say:** TwoRoom bottleneck is horizon mismatch (Official already 90% at 25, then 50/32/18). Forward restores long-offset success by ranking 25-step plans after \(F^k\). Backward is only a small lift. Fusion sits between Forward and Official; it does **not** beat Forward here.

---

## 5. Train figures (loss + \(\|z\|\))

Validation epoch means. Official \(=\mathrm{pred}+0.09\times\mathrm{SIGReg}\). Forward/Backward \(=\mathrm{step}+\mathrm{roll}\). Target \(\|z\|\approx\sqrt{192}\approx 13.9\) after projector BatchNorm.

### 5.1 PushT — `fig_pusht_backward_zz.png`

Metrics: `~/.cache/stable-pretraining/runs/20260810/031648/2d8e32169c86/metrics.csv`

| epoch | Official | pred | SIGReg | F loss | F out / tgt | B loss | B out / tgt |
|------:|---------:|-----:|-------:|-------:|------------:|-------:|------------:|
| 0 | 0.207 | 0.023 | 2.04 | 0.420 | 12.29 / 13.70 | 0.438 | 12.33 / 13.78 |
| 9 | 0.123 | 0.003 | 1.33 | 0.222 | 13.01 / 13.84 | 0.247 | 13.03 / 13.87 |

SIGReg is calm from epoch 0. F/B output norms stay on the sphere (end gap ~0.8). Plot y-axis for \(\|z\|\) is 0–16.

### 5.2 TwoRoom — `fig_tworoom_backward_zz.png`

Metrics: `~/.cache/stable-pretraining/runs/20260812/065606/dd74b35c7e9d/metrics.csv`  
Drawn from **epoch 0** (warmup included). \(\|z\|\) axis 0–24 so tgt≈21 is not clipped.

| epoch | Official | pred | SIGReg | F loss | F out / tgt | B loss | B out / tgt |
|------:|---------:|-----:|-------:|-------:|------------:|-------:|------------:|
| 0 | 7.03 | 1.49 | 61.5 | 4.05 | 6.85 / 21.12 | 3.51 | 8.93 / 21.15 |
| 1 | 5.73 | 1.12 | 51.2 | 3.56 | 6.50 / 19.69 | 3.08 | 7.70 / 19.69 |
| 2 | 0.94 | 0.19 | 8.38 | 1.68 | 6.21 / 13.75 | 1.65 | 6.66 / 13.82 |
| 9 | 0.16 | 0.007 | 1.68 | 1.65 | 5.64 / 13.69 | 1.68 | 5.65 / 13.75 |

Official drop after epoch 1 is mostly SIGReg (encoder onto the BN sphere), not pred recovering from failure. F/B **output** stays ~5.6 after the target has settled at ~13.7.

**What to say:** This is a training diagnostic, not a contradiction of the TwoRoom success table. Eval cost only needs candidate ranking; collapsed scale can still preserve direction. PushT shows F/B can also help when norms **are** aligned. Do not frame TwoRoom success as “caused by compression.”

---

## 6. Other fusion modes (PushT only, no dedicated figure)

Same ckpt / starts as §3. From `20260811_061335_pusht/` except `fusion_ofb` (`20260812_060947_pusht/`).

| mode | 25 | 50 | 75 | 100 |
|------|---:|---:|---:|----:|
| fusion_avg05 | 82 | **78** | **50** | 18 |
| fusion_avg07 | 82 | 74 | 46 | 20 |
| fusion_max | 82 | 64 | 46 | 16 |
| fusion_min | 82 | 68 | 46 | **32** |
| switch_remain | 80 | 70 | 42 | 20 |
| switch_offset | 82 | 78 | 50 | 28 |
| meet | 82 | 64 | 38 | 26 |
| fusion_ofb | 82 | 68 | 30 | 20 |

avg05 is the plotted Fusion series. fusion_min is best at offset 100 (32) but weaker than avg05 at 50.

---

## 7. Ablation: PushT B→p (`fblewm_bp`) — numbers only

Same starts as §3. Eval: `20260812_024719_pusht/`.

| mode | 25 | 50 | 75 | 100 |
|------|---:|---:|---:|----:|
| Official | 80 | 32 | 22 | 10 |
| Forward | 80 | 68 | 34 | 30 |
| Backward | 76 | 56 | 20 | 8 |

Backward is worse than B→z (§3). Forward still helps. Do not pool this Official row with §3 (different joint-train run).

---

## 8. Caption / sentence templates

**PushT success (CEM=5):**  
*Success rate (%) vs goal offset on PushT (`n=50`, shared starts). Official LeWM uses 25-step CEM. Forward / Backward / Fusion (avg05) keep the same CEM and add action-free imagination in the cost. Gains appear at offsets 50–100, where Official is myopic relative to the goal.*

**TwoRoom success:**  
*Same protocol on TwoRoom. Official already reaches 90% at offset 25 and drops once the goal lies beyond one CEM horizon. Forward recovers 90/84/66 at 50/75/100. This supports long-range ranking, not calibrated latent alignment.*

**Train norms:**  
*Validation losses and latent norms over 10 epochs. PushT: F/B outputs stay near the BatchNorm sphere (\(\|z\|\approx\sqrt{d}\)). TwoRoom: SIGReg spikes at epochs 0–1; after the encoder settles, F/B outputs remain at \(\approx 5.6\) vs target \(\approx 13.7\). Planning success in §4 still holds.*

---

## 9. Still running / not plotted

| Item | Status |
|------|--------|
| TwoRoom v2, `now`: \(g\leftarrow B(z_0,g)\), ckpt `fblewm_tworoom_v2` | Training (`20260813_044619_tworoom`) |
| Cube | Config ready; `cube_single_expert.tar.zst` SHA matched; not extracted / trained |
| Residual readout / imaginer delay | Not used in any figure above |
