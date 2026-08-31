# Figure requirements for the HAS paper

This document specifies the two conceptual figures referenced by
`iclr2027_conference.tex`. It is separate from `FIGURES.md`, which indexes
experimental artifacts and numerical results.

## Figure 1: Temporal correspondence in terminal scoring

### Purpose

Show the exact mismatch addressed by HAS: a short action-conditioned
prediction is compared with a goal observation sampled from a later time.
The figure should make the two time indices visible before introducing any
network detail.

### Layout

- Use two horizontal rows with a shared left-to-right time axis.
- Mark the current replanning time \(e\), the short-plan endpoint \(e+H\),
  and the goal time \(o\).
- Top row, “LeWM terminal scoring”:
  - Current observation/latent at \(e\).
  - Five action blocks ending at \(\hat z_{e+H}\).
  - Goal image and \(z_g\) at \(o\).
  - A dashed distance bracket directly connecting
    \(\hat z_{e+H}\) and \(z_g\), labelled “different time indices”.
- Bottom row, “Horizon-Aligned Scoring”:
  - Reuse the same current state and action-conditioned endpoint.
  - Add \(k=(o-e-H)/b\) Forward steps between the endpoint and goal time.
  - Draw the terminal distance only between \(F^k(\hat z_{e+H})\) and
    \(z_g\).
- Use one color for action-conditioned prediction, a second for
  action-free Forward transitions, and a neutral color for encoded
  observations.

### Caption message

LeWM compares a short-rollout endpoint with a later goal. HAS applies the
Forward imaginer for the remaining number of action blocks before computing
the terminal distance.

### Claims the figure must not imply

- Do not show \(F^k(\hat z)\) as an exact reconstruction of the future
  observation.
- Do not imply that the action-free path identifies executable future
  controls.
- Do not claim that temporal mismatch is the only source of long-horizon
  planning error.
- Do not describe unchanged LeWM components as a contribution.

### Source material

- Symbols and timing: `eq:predict-endpoint`, `eq:lewm-cost`, and `eq:depth`.
- Default values: \(b=5\), \(H=25\), and offsets \(25,50,75,100\).
- No task screenshot is required; small PushT or TwoRoom thumbnails may be
  added only as visual context.

## Figure 2: Forward training and HAS planning

### Purpose

Connect the detached local supervision used to train \(F\) with its recursive
use inside the planning cost.

### Layout

Use two panels.

#### Panel (a): Training

- Show four encoder latents \(z_0,z_1,z_2,z_3\).
- Show LeWM predictions \(p_0,p_1,p_2\), aligned as
  \(p_i\approx z_{i+1}\).
- Highlight only the Forward pairs used by the legacy latent variant:
  - \(F(\bar p_0)\rightarrow\bar z_2\)
  - \(F(\bar p_1)\rightarrow\bar z_3\)
  - \(F(F(\bar p_0))\rightarrow\bar z_3\)
- Place stop-gradient symbols on both predictor inputs and encoder targets.
- Distinguish the one-step and recursive losses by line style.
- State “action-free” next to \(F\); do not draw action inputs into \(F\).

#### Panel (b): Planning

- Start with \(N\) CEM candidate action-block sequences.
- Route each candidate through five action-conditioned predictor transitions
  to obtain \(\hat z^{(n)}_{e+25}\).
- Route each endpoint through a shared \(F\) recursively \(k(e,o)\) times.
- Compute the squared latent distance to the single encoded goal \(z_g\).
- Feed candidate costs back to the CEM elite update.
- Add a small depth schedule example for offset \(100\):
  \(15\rightarrow10\rightarrow5\rightarrow0\) across replans.

### Caption message

The Forward imaginer is trained from detached local LeWM alignments, including
one recursive composition. At evaluation it extends each candidate endpoint
to the goal horizon for terminal ranking.

### Claims the figure must not imply

- Do not draw gradients from the Forward loss into the encoder or predictor.
- Do not include the backward auxiliary head, fusion modes, or later Forward
  variants.
- Do not imply that \(F\) receives candidate actions.
- Do not present the two-step training loss as evidence of accurate
  fifteen-step open-loop prediction.
- Do not show the Forward trajectory as a subgoal sequence executed by the
  controller.

### Source material

- Architecture: two residual MLP blocks, latent width \(192\), hidden width
  \(768\), GELU, output LayerNorm and linear projection.
- Losses: `eq:forward-step`, `eq:forward-roll`, and `eq:forward-total`.
- Planning depth and cost: `eq:depth` and `eq:has-cost`.
- CEM: 300 candidates, 30 update iterations, 30 elites, variance scale 1.0.

## Rendering and delivery

- Design for single-column legibility first; Figure 2 may span two columns if
  labels become crowded.
- Use vector output (`.pdf` preferred) and embed fonts.
- Match the paper typography; equations should use LaTeX-rendered symbols.
- Avoid gradients, decorative 3D effects, and dense prose inside the figure.
- Final deliverables:
  - `paper/figures/fig_has_temporal_alignment.pdf`
  - `paper/figures/fig_has_training_planning.pdf`
- Replace the boxed placeholders in the manuscript with `\includegraphics`
  while preserving the existing labels `fig:overview` and `fig:method`.
