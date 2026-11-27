# Architecture Audit

This document records what the paper specifies, what this repository
implements, and which details remain hypotheses. The reported parameter count
is **46M** (rounded in the paper), not 40M. Matching that count alone does not
establish an equivalent reproduction.

## Paper-supported architecture facts

The paper specifies a seven-block U-Net (three encoder blocks, one middle
block, three decoder blocks), five densely connected TFC convolution layers
per block, 3×3 TFC kernels, TFC growth rate 24, 32-dimensional condition
embeddings, and GPoCM modulation. Its preferred route in §4.2 is:

| Stage | Condition supplied to its controller |
|---|---|
| Encoder | Timbre embedding (e_t) |
| Middle | Mixed embedding (e_m = 0.5(e_t + e_p)) |
| Decoder | Position embedding (e_p) |

These facts do not specify every block width, the controller's hidden
dimensions, the TFC block's returned tensor, or the exact parameter-counting
convention. Those details cannot be recovered uniquely from the paper's
rounded 46M figure.

## Repository variants

| Config | Purpose | Channels | TFC growth | Route | Trainable parameters |
|---|---|---:|---:|---|---:|
| `paper_route_w24_v1` | Narrow reference anchor | 24 at all four scales | 24 | §4.2 split-stage | 2,241,292 |
| `growth24_projected_a_v1` | Parameter-matched, preserves 124-wide interfaces | 124 at all four scales | 24 | §4.2 split-stage | 45,750,952 |
| `growth24_projected_b_v1` | Parameter-matched structural hypothesis B | 96 / 144 / 144 / 240 | 24 | §4.2 split-stage | 46,017,316 |
| `growth24_projected_c_v1` | Backbone-heavy parameter-matched hypothesis | 96 / 192 / 336 / 456 | 24 | §4.2 split-stage | 46,039,900 |
| Existing `mid_dual_paper` ablation runs | Historical controlled ablations | Uniform width 124 in the released setup | 124 | Decoder-only mixed condition | 30,441,288 (30.44M) |
| Shipped author-release config | Historical split-stage baseline | Uniform width 124 | 124 | §4.2 split-stage | 46,463,992 |

The first candidate uses the repository's dense TFC convention: each layer
receives the original block input concatenated with all preceding layer
outputs, and the block returns the last layer's 24 channels. It is a useful
growth-rate reference, not proof of the authors' exact implementation.

Candidates A/B/C keep five dense layers that each add 24 channels, concatenate
the input and all five new feature maps, then apply a 1×1 projection to the
requested block width. The projection is an explicit reconstruction choice;
the paper does not confirm it. A preserves the current 124-channel outputs,
while B and C use nonuniform multi-scale widths. All three are parameter-
matched hypotheses, not author-verified models.

The old 30.44M ablation result remains a valid within-study result. It does
not establish the paper's architecture or permit direct comparison of
absolute SDR. The shipped split-stage config has 46,463,992 parameters with
uniform width/growth 124. Its rounded size is close to the paper's 46M
report, but its growth rate conflicts with the paper's stated value of 24.

## Conditioning and the shortcut hypothesis

In `split_stage_paper`, the decoder controller receives an embedding derived
from position alone. Its GPoCM operation still combines the generated gate
with audio activations, `sigmoid(W(e_p) h + b(e_p)) ⊙ h`. Position-only
control parameters therefore do not by themselves make the decoder a
position-to-source lookup table: the decoder also receives encoded audio
features and U-Net skip features.

A position shortcut may still be learned from correlated training layouts,
but the route definition alone does not demonstrate that it occurred or that
the audio path was bypassed. Treat the shortcut explanation as a hypothesis
unless supported by an input-dependence intervention or equivalent
measurement. `mid_dual_paper` is a separate route used for the repository's
historical ablations; it is not the paper's preferred §4.2 route.

## Parameter-count convention

The candidate counts are computed from instantiated PyTorch models. They
include trainable convolution, linear, embedding, and BatchNorm affine
parameters, and exclude BatchNorm running statistics. Generated GPoCM
matrices are activations produced at runtime and are not counted again as
parameters. Run:

```powershell
$env:PYTHONPATH = "src"
python scripts/audit_architecture_params.py configs/paper_route_w24_v1.yaml
python scripts/audit_architecture_params.py configs/growth24_projected_b_v1.yaml
```

The exact model definitions and assumptions are documented in
[`GROWTH24_46M_CANDIDATES.md`](GROWTH24_46M_CANDIDATES.md). Both configs are
marked candidate/test-only and are blocked from the paper-verified execution
path until stronger architecture evidence is available.

## Data split note

The paper's prose describes 65 training works and six validation works. The
official supplementary archive inspected for this reproduction contains 65
unique works, each represented in both layouts; the repository manifests use
59 works for training and six disjoint works for validation, covering those
65 unique works. This discrepancy is recorded for follow-up. The manifests
and audio files were left unchanged.

## Correction to the earlier architecture note

An earlier version of this document described the decoder's position-only
controller as a demonstrated lookup-table failure and described
`mid_dual_paper` as the paper's primary route. Those were stronger claims
than the available evidence supports. The current audit above supersedes
those interpretations; prior SDR measurements remain observations of their
documented model and training protocol.
