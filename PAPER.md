# Paper writing map

Path-level index for the ICLR 2027 draft. This file lives at the repo root so Overleaf’s `paper/` compile tree stays unchanged. Do not paste numbers here; read them from the TeX tables or from local eval JSON.

Local clone:

```bash
git clone https://github.com/ybenzou/Forward-Backward-LeWM.git
```

Server checkout: `/home/yuanben/WorldModel/FBLeWM`. Same `main` as GitHub and Overleaf.

## Compile

Write only in `paper/iclr2027_conference.tex`. `paper/main.tex` just `\input`s that shell. Official `paper/iclr2027_conference.sty` / `.bst` / `math_commands.tex` stay untouched. Notation extras are in `paper/macros.tex`.

Figures used by the draft: `paper/figures/*.pdf`. Table fragments: `paper/tables/`. Compile notes: `paper/README.md`. Figure briefs (not the compiled PDFs): `paper/FIGURE_REQUIREMENTS.md`. Older numeric lookup (may lag the TeX): `paper/FIGURES.md`.

Overleaf main document: `paper/main.tex` or `paper/iclr2027_conference.tex`.

## Method in the TeX

| Label | What it is |
|-------|------------|
| `sec:method` | Method section |
| `sec:lewm` | LeWM four-frame interface and `eq:lewm` |
| `sec:imaginer` | Action-free Forward `F`, step / roll losses (`eq:forward-step`–`eq:forward-total`), Figure 2 |
| `sec:alignment` | Dynamic depth `k(e,o)` and HAS cost (`eq:depth`, `eq:has-cost`) |
| `sec:planning` | Receding-horizon CEM (`alg:has-cem`) |
| `sec:exp` / `sec:main` | Protocol and main table (`tab:main`, `fig:multiseed`) |
| `sec:analysis` | Process strips and held-out score contrast |
| `sec:ablation` | `k=0` identity and `H=50` control |
| `app:implementation` | Architecture, detach, extra training head |
| `app:protocol` | CEM / starts / scoring diagnostic |
| `app:ablation` | Action-conditioned Forward check |

## Code

| Path | Role |
|------|------|
| `train.py` | Joint LeWM + Forward (+ Backward) training; Hydra `seed` is applied with `pl.seed_everything` |
| `module.py` | Official modules + `CausalLatentImaginer` |
| `fblewm.py` | Encode / predict / `imagine_forward` / `get_cost` |
| `policy.py` | `compute_imagine_steps` (dynamic `k`) and `FBWorldModelPolicy` |
| `planning.py` | Mode names and fusion helpers |
| `eval.py` | Single mode / offset Hydra entry |
| `checkpoint_utils.py` | FBLeWM-only loader |
| `scripts/run_fblewm_pipeline.py` | Staged train / eval; `--seed` is passed through to Hydra |
| `scripts/eval_fblewm_matrix.py` | Offset × mode matrix; writes `results.json` / `summary.txt` / `starts_manifest.json` |
| `scripts/run_iclr_multiseed.py` | Ten eval groups (seeds 42–51) for the ICLR-bar figure |
| `scripts/run_iclr_longcem.py` | `H=50` (`h=10`) control, same starts |
| `scripts/run_forward_train_seeds.py` | Extra train seeds 3072–3074; isolated run names; does not overwrite paper checkpoints |
| `scripts/run_score_contrast_then_k.py` | Score-contrast diagnostic, then fixed-`k` sweep |
| `scripts/run_has_interpretability.py` | Process figure + latents + seed-42 `k` table |
| `scripts/run_k_ablation_multiseed.py` | Fixed `k∈{5,10,15}` on the ten groups |
| `tests/test_forward_train_seeds.py` | Contracts for the train-seed launcher |

Always `source scripts/env.sh` first so `STABLEWM_HOME` is this tree’s `.stable-wm`.

## Data (on disk, not in git)

`outputs/` and `.stable-wm/` are gitignored. They exist on this server only.

| Path | Role |
|------|------|
| `paper/tables/tab_k_depth.tex` | Seed-42 fixed-`k` fragment (`scripts/summarize_k_ablation.py`) |
| `paper/tables/tab_k_depth_multiseed.tex` | Ten-group `k` fragment (`scripts/plot_k_depth_multiseed.py`) |
| `outputs/eval/<run_id>/` | Ordinary matrix eval |
| `outputs/diag/iclr_bar/multiseed/{task}/seed_*/` | Main 10-group starts + `summary.txt` |
| `outputs/diag/iclr_bar/longcem/h10/{task}/seed_*/` | Long-CEM control |
| `outputs/diag/k_ablation/` | Seed-42 fixed-`k` units |
| `outputs/diag/k_ablation/multiseed/` | Ten-group fixed-`k` |
| `outputs/diag/has_score_contrast/v1/` | Held-out window scores (JSON) |
| `outputs/diag/process_compare/v2/` | Paired LeWM / HAS rollouts |
| `outputs/diag/has_latents/v1/` | Latent diagnostics |
| `outputs/diag/train_seeds/{task}/s<seed>/seed_*/` | Extra-train-seed evals |
| `.stable-wm/checkpoints/` | Epoch-10 weights used by the paper (`fblewm/`, `fblewm_tworoom/`, `fblewm_reacher_v1/`) |
| `/home/yuanben/WorldModel/LeWM/data/` | Shared datasets (`tworoom.tar.zst`, etc.) |

Paper checkpoints must not be overwritten; `run_forward_train_seeds.py` writes `fblewm_{task}_s<seed>/` instead.

## Regenerate figures

| Paper file | Script | Notes |
|------------|--------|-------|
| `paper/figures/fig_has_training.pdf` | `paper/interpretability_preview/plot_temporal_leap.py` | Figure 2 sketch; also writes a preview PNG next to the script |
| `paper/figures/fig_has_evaluation.pdf` | Drawn from `paper/FIGURE_REQUIREMENTS.md` (Figure 1) | Keep the compiled PDF in `paper/figures/` |
| `paper/figures/fig_iclr_multiseed.pdf` | `scripts/plot_iclr_multiseed.py` | Reads `outputs/diag/iclr_bar/` |
| `paper/figures/fig_has_process.pdf` | `scripts/plot_has_process.py` | Also copies into `outputs/figures/` |
| `paper/figures/fig_has_score_contrast.pdf` | `scripts/analyze_has_score_contrast.py` | Script default is `paper/interpretability_preview/`; copy the PDF into `paper/figures/` for the draft |

Preview PNGs under `paper/interpretability_preview/` are local scratch. Do not commit them.

## Do not commit

- Checkpoints: `*.pt`, `*.ckpt`, `.stable-wm/`
- Runtime trees: `outputs/`, `logs/`, `wandb/`
- Preview rasters: `paper/interpretability_preview/*.png`
- Calendar junk: `iclr2027-deadlines.ics`
