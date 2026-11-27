# Growth-24 and the Paper's 46M Parameter Report

## Summary

There is no mathematical conflict between a TFC growth rate of 24 and a
rounded total size of 46M. Growth rate controls how many channels each dense
TFC layer adds. Total size also depends on the block output widths, the
channels acted on by GPoCM, and the controller that generates each GPoCM
matrix.

The paper reports 46M, but does not give enough per-layer dimensions to
reconstruct one unique 46M model. This repository therefore provides three
parameter-matched candidate configs and one narrow reference config:

| Config | What it checks | Trainable parameter count |
|---|---|---:|
| `paper_route_w24_v1` | Uniform 24-channel reference with growth 24 and the paper's §4.2 condition route | 2,241,292 |
| `growth24_projected_a_v1` | Preserve 124-channel block interfaces while setting growth to 24 | 45,750,952 |
| `growth24_projected_b_v1` | Multi-scale widths 96/144/144/240, growth 24 | 46,017,316 |
| `growth24_projected_c_v1` | Wider backbone with widths 96/192/336/456, growth 24 | 46,039,900 |

All three parameter-matched configs are test-only and remain outside the
paper-verified execution path. None is labelled as the recovered author model.

## Evidence and unknowns

The paper specifies seven U-Net blocks, five TFC layers per block, 3×3 TFC
kernels, growth rate 24, a 32-dimensional condition embedding, GPoCM, the
encoder–timbre / middle–mixed / decoder–position route, and a reported model
size of 46M.

The paper does not specify the full stage-width schedule, whether each TFC
returns only the last dense layer's features or projects the full dense
concatenation, the exact GPoCM parameter generator dimensions, or whether
reported size counts buffers. The 46M figure is rounded and cannot identify
those missing design choices.

The upstream LaSAFT implementation is evidence for one plausible TFC
convention: dense layers concatenate earlier features as input and return
the last layer's feature map. It is not evidence that every unpublished
detail in the Interspeech model uses the same implementation.

## Candidate W24: narrow reference anchor

Config: [`configs/paper_route_w24_v1.yaml`](../configs/paper_route_w24_v1.yaml)

- Four scale widths are all 24; all seven blocks output 24 channels.
- Five dense TFC layers each add 24 channels; each block returns the last
  layer's 24-channel feature map.
- Decoder TFC input width is 48 because the upsampled feature and skip
  feature are concatenated.
- The route is encoder (e_t), middle (e_m), decoder (e_p).
- Current instantiated trainable count: **2,241,292**.

This isolates the published growth rate and preferred conditioning route
under a compact, auditable interpretation. It is a reference anchor, not an
attempt to force the count to 46M.

## Candidate A: preserve the 124-channel interfaces

Config: [`configs/growth24_projected_a_v1.yaml`](../configs/growth24_projected_a_v1.yaml)

- All seven blocks output 124 channels; decoder TFC inputs remain 248 after
  skip concatenation.
- Five TFC layers each add 24 channels; concatenate the input and five
  features, then project to 124 channels.
- Controller hidden dimensions are 64→128→384.
- Main trunk plus embeddings: 3,795,092 parameters.
- Controllers: 41,955,860 parameters (91.70%).
- Total: **45,750,952**. With the same trunk and H=256, the total is
  **31,812,648**.

This is the smallest interface change among the three 46M candidates. Its
controller dominates the parameter budget, so the count is reached mostly
by widening the gate generator.

## Candidate B: projected dense features

Config: [`configs/growth24_projected_b_v1.yaml`](../configs/growth24_projected_b_v1.yaml)

The stage widths are `C0,C1,C2,C3 = 96,144,144,240`. The seven block
output widths are 96, 144, 144, 240, 144, 144, 96. Encoder and middle TFC
input widths are 96, 144, 144, 240. Decoder input widths are 288, 288, 192:
each decoder concatenates a `2 × Ci`-wide upsample/skip tensor before its
TFC.

Each block uses five dense convolution layers. For input width `c`, the
layer inputs are `c, c+24, c+48, c+72, c+96`; each layer adds 24 channels.
The input and five layer outputs are concatenated, then a 1×1 convolution
projects `c+120` channels to that block's requested output width `C`.
GPoCM and TDF operate at width `C`.

This final 1×1 projection is an explicit hypothesis. It decouples growth
rate from block width, but it is not confirmed by the paper.

### Candidate B counting assumptions

- Three encoder blocks, one middle block, and three decoder blocks.
- The stage schedule above; three 4×4 stride-2 downsampling layers and three
  matching transposed convolutions.
- Five 3×3 dense convolutions per TFC; each adds 24 channels.
- Dense concatenation followed by 1×1 convolution and BatchNorm projection.
- TFC and projection convolutions have no bias; BatchNorm affine parameters
  are counted.
- TDF uses the repository's shared-frequency Linear layers, bottleneck
  factor 16, minimum hidden width 16, and bias-free Linear layers.
- Full `C × C` GPoCM matrices and `C`-wide biases are generated for
  all seven blocks.
- Three independent stage controllers use 32→64→128→256 Linear widths,
  LayerNorm on hidden layers, Linear biases, and no output normalization.
- Timbre and position embeddings contribute 224 parameters.
- Count trainable parameters; exclude BatchNorm running-statistic buffers.
  Runtime-generated GPoCM matrices are activations, not additional
  parameters.

For one projected TFC block with input width `c`, the five dense
convolutions and their BatchNorm affine parameters contribute
`Σ[l=0..4] (9×24×(c+24l) + 2×24) = 1080c + 52,080`. The final projection
contributes `(c+120)×C + 2C`. These terms are counted separately for all
seven block inputs, including the doubled decoder inputs after skip
concatenation.

For the controller, the shared hidden stack contributes 44,352 parameters
per controller. Its output layer contributes
`257 × Σ(Ci² + Ci)`, where 257 is hidden width 256 plus the Linear bias
term, and `Ci` ranges over the stages controlled by that MLP. Across the
three independent controllers, this gives 41,248,944 parameters. The
instantiated-model audit below is the authoritative total.

### Instantiated parameter audit

| Top-level component | Parameters |
|---|---:|
| Stem | 960 |
| Encoder TFC/TDF blocks | 842,064 |
| Downsampling layers | 1,106,976 |
| Middle TFC/TDF block | 403,216 |
| Upsampling layers | 1,106,688 |
| Decoder TFC/TDF blocks | 1,307,472 |
| Output head | 772 |
| Embeddings | 224 |
| Encoder controller | 13,169,856 |
| Middle controller | 14,909,232 |
| Decoder controller | 13,169,856 |
| **Total** | **46,017,316** |

About 89.64% of this candidate's parameters are in its three condition
controllers. That is a consequence of generating full dense GPoCM matrices
at the selected stage widths; matching 46M does not imply a balanced or
author-faithful parameter allocation.

## Candidate C: allocate more parameters to the backbone

Config: [`configs/growth24_projected_c_v1.yaml`](../configs/growth24_projected_c_v1.yaml)

- Stage widths are 96, 192, 336, 456; block outputs are 96, 192, 336, 456,
  336, 192, 96.
- Dense TFC layers still add 24 channels each and project to the requested
  block width.
- Controller hidden dimensions are 64→128→64.
- Main trunk plus embeddings: 11,688,820 parameters.
- Controllers: 34,351,080 parameters (74.61%).
- Total: **46,039,900**.

This shifts more parameters into the audio backbone, but changes both stage
widths and controller hidden width relative to B. Its training memory and
runtime have not been profiled.

## Shared parameter formulas

For each projected TFC block with input width `c`, output width `C`, and
growth 24:

- Dense convolutions plus BatchNorm affine parameters:
  `Σ[l=0..4](9×24×(c+24l)+2×24) = 1080c+52,080`.
- Dense concatenation projection:
  `(c+120)×C+2C`.
- TDF:
  `2×f×max(f//16,16)+4C`, where `f` is that block's frequency width.

For three controllers with hidden dimensions 64→128→H and full GPoCM
matrices:

`P_controller = 3(10,816 + 131H) + (H+1)×Σ_i(C_i²+C_i)`.

The counts above include down/up sampling, stem, head, embeddings, and all
seven TFC/TDF blocks. BatchNorm running-statistic buffers and runtime-created
GPoCM matrices are excluded.

## Why the existing 124-channel model is not a growth-24 solution

The repository's older uniform implementation ties each TFC layer's output
width to its block width. With the split-stage route and the remaining
current conventions, `C = g = 124` gives 46,463,992 parameters (46.46M), while
`C = g = 24` gives 2,241,292. Thus the old 46M-scale setting does not satisfy
the paper's stated growth rate. Candidates A/B/C explicitly separate the
internal TFC growth from its projected block output width.

The former decoder-only `mid_dual_paper` ablation setup is a separate
architecture and has 30,441,288 parameters (30.44M) in the released
configuration. The shipped split-stage author-release config has
46,463,992 parameters but uses growth rate 124. Its rounded count is close
to 46M, but it does not satisfy the paper's stated growth rate. The
ablation results remain measurements of their setup; they
do not resolve the authors' missing width and controller details.

## How to reproduce the counts

```powershell
$env:PYTHONPATH = "src"
python scripts/audit_architecture_params.py configs/paper_route_w24_v1.yaml
python scripts/audit_architecture_params.py configs/growth24_projected_a_v1.yaml
python scripts/audit_architecture_params.py configs/growth24_projected_b_v1.yaml
python scripts/audit_architecture_params.py configs/growth24_projected_c_v1.yaml
```

These parameter counts establish that the candidate definitions have the
stated sizes. They do not identify the author's implementation and do not
predict source-separation SDR.
