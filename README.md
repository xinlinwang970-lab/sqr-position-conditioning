# From Synthetic to Real: A Case Study in String Quartet Source Separation

**How can a string quartet separator use positional cues while generalizing beyond the seating arrangement used for training?**

This repository studies timbre- and position-conditioned source separation through an independent implementation and exploratory experiments. We train on synthesized SQR sources and evaluate on spatialized recordings of individual instrument performances, examining both separation quality and sensitivity to seating changes.

The project began as an independent reimplementation of Xu et al.'s [*Position also matters!*](https://www.isca-archive.org/interspeech_2025/xu25b_interspeech.html). It now documents conditioning routes, position-label perturbation, controller regularization, and the limits of the available evidence. It is not an official implementation or a verified numerical reproduction of the original system.

## Motivation

A string quartet contains two violins, a viola, and a cello. Their overlapping spectra make separation difficult, and the two violin parts share the same timbre category. Position supplies an additional target-selection cue.

A fixed seating arrangement also correlates instrument identity with position. Our working hypothesis is that a model may depend on these associations and lose separation quality when the arrangement changes.

We ask three questions:

1. **Conditioning architecture:** How might the placement of timbre and position conditioning affect transfer to unseen instrument-position combinations?
2. **Layout generalization:** How do position-label perturbation and controller regularization affect separation quality and robustness to seating changes?
3. **Synthetic-to-recorded transfer:** How do models trained on synthesized sources perform on mixtures built from recorded performances?

**Central argument:** evaluate cross-layout score changes together with absolute separation quality. Small layout differences alone do not establish effective separation or correct spatial reasoning.

## Problem setting and data

Given a stereo mixture $x$, a target timbre $t$, and a position label $p$, the model estimates one requested stem:

$$
\hat{s}=f_\theta(x,t,p).
$$

The two violin parts share timbre ID 0 and are distinguished by their position conditions.

| Target part | Timbre ID | Type2 position ID | Type1 position ID |
|---|---:|---:|---:|
| Violin 1 | 0 | 3 | 3 |
| Violin 2 | 0 | 2 | 0 |
| Viola | 1 | 1 | 1 |
| Cello | 2 | 0 | 2 |

Main training uses **Type2 only**. Type1 exchanges Violin 2 and Cello, introducing the unseen timbre-position combinations `(Violin, 0)` and `(Cello, 2)`. Violin 1 and Viola retain their target seats while the surrounding sources move. This tests a specific recombination of instrument and seat conditions, not arbitrary room geometries.

| Split | Source domain | Layouts | Selected records |
|---|---|---|---:|
| Training | Synthetic SQR sources | Type2 | 59 |
| Validation | Synthetic SQR sources | Type1 and Type2 | 6 source IDs × 2 = 12 |
| Test | URMP recorded sources | Type1 and Type2 | 6 source recordings × 2 = 12 |

The main selection uses the six validation IDs listed in the [SQR release](https://zenodo.org/records/14793971). Audio is stereo at 44.1 kHz; the STFT uses a 2048-sample window and 1024-sample hop. The local 59/6 selection differs from the paper's stated 65-training-plus-six-validation count; see [data provenance](docs/DATA_PROVENANCE.md) and the frozen [manifests](manifests).

**Recorded-source testing is not live-quartet testing.** The URMP-derived examples are spatialized mixtures of isolated instrument recordings, rather than simultaneous quartet recordings. Their musical material also differs from the synthetic validation set, so a validation-to-test gap cannot isolate the source-domain effect.

## Model and training interventions

### Conditioning routes

The separator uses a TFC–TDF U-Net backbone with gated point-wise convolutional modulation (GPoCM). Timbre and position embeddings are combined as $e_m=(e_t+e_p)/2$.

| Route | Encoder modulation | Middle modulation | Decoder modulation |
|---|---|---|---|
| `split_stage_paper` | Timbre $e_t$ | Combined $e_m$ | Position $e_p$ |
| `mid_dual_paper` | None | None | Combined $e_m$ |

Decoder-only conditioning leaves the encoder and middle query-independent while the decoder selects the target. These earlier stages still process timbral and spatial information in the mixture. Delaying explicit conditioning might reduce dependence on fixed instrument-seat associations, but this remains a hypothesis.

The routes also differ in controller count, model capacity, and information supplied to the decoder. Their historical comparisons do not isolate conditioning placement. The archived configurations use **TFC growth rate 124**, whereas the [paper specifies 24](https://www.isca-archive.org/interspeech_2025/xu25b_interspeech.pdf). Architecture labels containing `paper` and similar parameter counts do not establish architectural equivalence.

### Interventions within decoder-only conditioning

| Intervention | Training-time change |
|---|---|
| **Position-label permutation: P10/P15/P50** | Apply one random permutation of four position IDs to a batch with probability 0.10/0.15/0.50. Keep mixture audio, target audio, and timbre labels unchanged. |
| **Embedding noise: E2** | Add Gaussian noise to the position embedding. |
| **Embedding drop: E3** | Randomly zero the position embedding. |
| **Controller consistency: E4** | Retain the correct condition for supervised separation and add a penalty between decoder-controller parameters generated for original and permuted positions. |

Label permutation makes the position condition less reliable; it does not physically rerender a new layout. A permutation can leave a particular ID unchanged, so its application probability differs from the probability that an individual label changes.

E4 uses $\mathcal L=\mathcal L_{\mathrm{sep}}+\lambda\mathcal L_{\mathrm{reg}}$ with $\lambda=1$. The regularizer averages elementwise weight and bias MSEs across decoder modulation stages. Independent dropout masks in the compared controller passes also introduce stochastic differences into this penalty.

See the [experiment design](docs/E1-4_DESIGN.md) for configurations and execution details.

## Claims and evidence

These findings describe saved single-seed checkpoints. They do not establish causal mechanisms or an optimal configuration.

**1. The historical baseline shows a layout-associated performance gap.** Its Type2 median SDR is **1.636 dB**, compared with **-0.330 dB** on Type1: a pooled-median difference of **1.966 dB**. This motivates studying layout sensitivity; it does not identify its cause. [Baseline scores](results/test_results/author_release_12ep_warmup_cosine_test_results.json).

**2. P15 has the highest observed mean SDR among the three same-stage swap runs.** These runs use the same 59-record selection and decoder-only route, at **425 completed epochs / 25,075 script progress updates**.

| Run | Swap probability | Mean SDR | Type1 median | Type2 median | Δ pooled medians | Median paired abs(Δ) |
|---|---:|---:|---:|---:|---:|---:|
| P10 | 0.10 | 2.208 | 1.748 | 2.119 | 0.371 | 0.790 |
| P15 | 0.15 | 2.515 | 2.265 | 2.301 | 0.036 | 0.751 |
| P50 | 0.50 | 2.261 | 1.996 | 1.913 | -0.083 | 0.681 |

All scores are in dB. P50 has the smallest median absolute paired difference among these three runs, while P15 has higher separation quality. P15's nearly equal pooled medians do not imply near-zero changes for individual source-part pairs. A matched zero-permutation control is needed to isolate a permutation benefit. Raw scores: [P10](results/checkpoint_evals/P10_test_ep424.json), [P15](results/checkpoint_evals/P15_test_ep424.json), [P50](results/checkpoint_evals/P50_test_ep424.json).

**3. E4 shows smaller paired layout differences, with lower overall mean SDR than P15.** At the recorded 425-epoch snapshots, E4 has a smaller median absolute paired layout difference (**0.607 vs. 0.751 dB**) and lower overall mean SDR (**2.317 vs. 2.515 dB**). Its viola mean SDR across both layouts is higher (**1.105 vs. 0.955 dB**), a supplementary part-level observation. [E4 scores](results/checkpoint_evals/E4_test_ep424.json).

Attribution to the regularizer requires a matched $\lambda=0$ control with dropout and activation-recomputation/BatchNorm behavior aligned. A smaller layout difference alone does not demonstrate a more accurate representation of source position.

### How the scores are summarized

Metric: **fixed-reference, gain-dependent waveform SDR**, with no fitted reference gain or filter:

$$
\operatorname{SDR}=10\log_{10}\frac{\lVert s\rVert_2^2}{\lVert\hat{s}-s\rVert_2^2+\epsilon}.
$$

The historical evaluators use $\epsilon=10^{-8}$. Their implementation and aggregation have not been established as equivalent to the original paper's evaluator; cross-paper score differences cannot be attributed to architecture alone.

For one checkpoint, six recordings × two layouts × four parts yield **48 scores**, forming **24 pairs** matched by `(source_id, part_id)`. We use the repository's signed convention:

$$
d_{i,k}=\operatorname{SDR}_{i,k,\mathrm{Type2}}-\operatorname{SDR}_{i,k,\mathrm{Type1}}.
$$

- **Mean SDR:** mean of all 48 scores.
- **Δ pooled medians:** Type2 median minus Type1 median; each median pools 24 scores.
- **Median paired abs(Δ):** $\operatorname{median}_{i,k}|d_{i,k}|$, calculated after matching sources and parts.

The 24 pairs share six underlying recordings. They are correlated observations, not 24 independent recordings or 48 independent test sources. Read smaller paired differences together with separation quality: consistently poor separation can also have small differences.

Full definitions, per-part results, and input hashes are in the [paired analysis](docs/EXPERIMENT_REPORT.md#paired-layout-analysis) and [machine-readable summary](results/layout_pair_summary.json).

## Evidence boundaries

- **Historical comparisons are not matched architecture ablations.** The baseline uses a 60/5 split, dense sampling, a different conditioning route, and a different optimization budget. Its five validation IDs differ from the main six; cross-protocol training exposure prevents simply merging them into one unseen validation set.
- **Matched intervention controls are missing.** There is no same-protocol `p_swap=0` or E4 $\lambda=0$ control. E1cosine-B uses custom data with unresolved provenance and is excluded from the same-data swap comparison.
- **Execution differs across families.** P10/P15/P50 use activation-recomputation wrappers that update BatchNorm again during backward; E4 does not. E4's independent controller dropout masks further confound a position-only interpretation.
- **Statistical coverage is limited.** Single-seed runs, six URMP source recordings, and repeated test-set inspection support exploratory observations. Future selection should use validation data, repeated training, source-level uncertainty analysis, and independent test material.

See [Known Limitations](docs/KNOWN_LIMITATIONS.md) and [Reproducibility](REPRODUCIBILITY.md) for the full engineering audit and the distinction between saved-score verification and fresh model evaluation.

## Next experiments

The following are proposed controls, not completed results:

1. Compare both conditioning routes with permutation probabilities 0 and 0.15 under identical data, sampling, optimization, evaluation, and checkpoint-selection rules. Add capacity-matched controls to separate route effects from parameter-count differences.
2. Compare E4 with a matched zero-regularization run, controlling controller dropout and activation-recomputation/BatchNorm behavior.
3. Investigate growth-rate-24 architecture reconstruction separately from the historical intervention results. Matching the paper's parameter count is a constraint, not proof of faithful reproduction.
4. Evaluate multiple seeds and new held-out recordings, including per-part paired differences, interference leakage, output energy, gain sensitivity, and mixture consistency.

## Quick start

```bash
git clone https://github.com/xinlinwang970-lab/sqr-position-conditioning.git
cd sqr-position-conditioning
python -m pip install -e ".[torch,test]"
```

Verify the archived results and run the test suite:

```bash
python scripts/verify_results.py
python scripts/summarize_layout_pairs.py --check
python -m pytest -q
```

These checks validate saved-score arithmetic, paired statistics, preserved evidence hashes, and tested implementation behavior. They do not reproduce historical GPU training or full-song inference. Audio datasets and model weights are not bundled.

See [Quickstart](docs/QUICKSTART.md) for dataset preparation, training, and checkpoint evaluation.

## Repository guide

| Resource | Contents |
|---|---|
| [Data provenance](docs/DATA_PROVENANCE.md) and [manifests](manifests) | Source selections, audio conventions, and labels |
| [Experiment design](docs/E1-4_DESIGN.md) and [configs](configs) | Interventions and recorded training settings |
| [Experiment report](docs/EXPERIMENT_REPORT.md) and [results](results) | Saved scores, paired comparisons, and part-level analyses |
| [Source code](src/sqr_repro) and [scripts](scripts) | Models, data handling, training, evaluation, and verification |
| [Reproducibility](REPRODUCIBILITY.md) and [release audit](docs/AUDIT_REPORT.md) | Verification scope, historical corrections, and limitations |

## References and license

Xu, Y., Wang, Y., Wu, X., Li, X., and Yu, F. (2025). [*Position also matters! Separating Same Instruments in String Quartet using Timbral and Positional Cues*](https://www.isca-archive.org/interspeech_2025/xu25b_interspeech.html). Interspeech 2025, 3105–3109. DOI: [10.21437/Interspeech.2025-130](https://doi.org/10.21437/Interspeech.2025-130).

Dataset: [String Quartet Set with Reverb (SQR)](https://zenodo.org/records/14793971). Cite the original method and data separately from this repository; see [CITATION.cff](CITATION.cff).

Code is released under the [MIT License](LICENSE). Third-party assets retain their own terms; see [Third-Party Notices](THIRD_PARTY_NOTICES.md) and [Contributing](CONTRIBUTING.md).

Historical `author_release` identifiers describe experiments using released data, not author-provided model code. This repository documents an independent implementation and exploratory case study; broad generalization to independent live quartet performances remains unestablished.
