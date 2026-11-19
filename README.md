# SQR Position Conditioning

**How can a source-separation model use seating cues while remaining robust to a change in seating layout?**

We study timbre- and position-conditioned string quartet separation through an independent implementation and exploratory experiments.

## Motivation and problem

Two violins share the same timbre label, so position helps identify the target. A fixed training layout also correlates seat with part identity. Our working hypothesis is that this correlation can encourage dependence on the training layout. We investigate the trade-off between reliable target selection and robustness to layout changes.

Given a stereo mixture `x`, a target timbre `t`, and its seat `p`, the model predicts one target stem:

```text
y_hat = f_theta(x, t, p)
```

Targets are Violin 1, Violin 2, Viola, and Cello. Training uses **type2**; evaluation uses type2 and **type1**, which exchanges the seats of Violin 2 and Cello. Violin 1 and Viola retain their target seats while the surrounding source layout changes.

**Central argument:** evaluate layout robustness alongside separation quality. A causal position-shortcut mechanism remains a hypothesis to test.

## Interventions

| Intervention | Training-time change |
|---|---|
| **Position swap** | Apply one random permutation to the batch's position labels with probability `p_swap`; keep the audio and targets unchanged. |
| **Embedding noise (E2)** | Add Gaussian noise to the position embedding. |
| **Embedding drop (E3)** | Randomly zero the position embedding. |
| **Controller consistency (E4)** | Penalize differences between decoder modulation parameters generated for original and permuted positions. |

Implementation details and protocol differences: [experiment design](docs/E1-4_DESIGN.md).

## Claims and evidence

**1. The historical baseline shows a layout-associated performance gap.** Its type2 median SDR is **1.636 dB**, compared with **-0.330 dB** on type1: a difference of **1.966 dB**. This documents a layout-associated performance gap on the recorded test set; its cause requires further controls.

**2. P15 has the highest observed mean SDR among the three same-stage swap runs.** These runs use the same 59-item selection, at **425 completed epochs / 25,075 progress updates**.

| Run | Swap probability | Mean SDR | type1 median | type2 median | Δ pooled medians | Median paired abs(Δ) |
|---|---:|---:|---:|---:|---:|---:|
| P10 | 0.10 | 2.208 | 1.748 | 2.119 | 0.371 | 0.790 |
| P15 | 0.15 | 2.515 | 2.265 | 2.301 | 0.036 | 0.751 |
| P50 | 0.50 | 2.261 | 1.996 | 1.913 | -0.083 | 0.681 |

P15 has nearly equal pooled median SDRs across layouts. Pairwise layout differences remain; these single-seed observations do not establish an optimal swap probability.

**3. E4 shows smaller paired layout differences, with lower overall mean SDR than P15.** At the recorded 425-epoch snapshots, E4 has a smaller median absolute paired layout difference (**0.607 vs. 0.751 dB**) and lower overall mean SDR (**2.317 vs. 2.515 dB**). As a part-level observation, its viola mean SDR across both layouts is higher (**1.105 vs. 0.955 dB**).

These are descriptive saved-score comparisons. Attribution to the regularizer requires a matched λ=0 control, with dropout and activation-recomputation/BatchNorm behavior controlled. See [known limitations](docs/KNOWN_LIMITATIONS.md).

Metric: **gain-dependent waveform SDR, dB**. Mean pools 48 correlated queries; each layout median pools 24. **Δ pooled medians** subtracts the two layout medians. **Median paired abs(Δ)** first matches the same source and part, then takes the median absolute score change. Read stability together with separation quality. [Paired analysis](docs/EXPERIMENT_REPORT.md#paired-layout-analysis) documents all statistics and part-level results.

## Data and evidence boundaries

Main selection: **59 synthetic SQR type2 training records**, **6 validation IDs x 2 layouts**, and **6 URMP source recordings x 2 layouts** for testing. Four queries per test item give 48 scores. See [data provenance](docs/DATA_PROVENANCE.md).

- One seed and repeated test-set inspection limit conclusions. Future selection should use validation data and independent test material.
- The dense baseline differs in split, route, sampling, and budget; comparisons across intervention families have further confounds.
- A matched `p_swap=0` control is absent; the historical dense baseline cannot serve as that control.
- E1cosine-B points to unresolved custom data and is excluded from the same-data swap comparison.
- Audio and weights are excluded. Original-paper equivalence and broad generalization to independent live performances remain unestablished.

## Quick start and resources

```bash
python -m pip install -e ".[torch,test]"
python -m pytest -q
python scripts/verify_results.py
python scripts/summarize_layout_pairs.py --check
```

These commands verify the original results and the [derived paired summary](results/layout_pair_summary.json) without inference. See [Quickstart](docs/QUICKSTART.md) for generation, data, training, and evaluation commands, and [Reproducibility](REPRODUCIBILITY.md) for verification scope.

Browse [source](src/sqr_repro), [scripts](scripts), [configs](configs), [manifests](manifests), and [results](results). The [release audit](docs/AUDIT_REPORT.md) records historical corrections.

Reference: Xu et al., [Position also matters!](https://www.isca-archive.org/interspeech_2025/xu25b_interspeech.html), Interspeech 2025. Data: [SQR](https://zenodo.org/records/14793971). Historical `author_release` labels refer to local use of released data, not official models.

Code: [MIT](LICENSE). Third-party assets retain their own terms; see [notices](THIRD_PARTY_NOTICES.md). Contribution and citation information: [CONTRIBUTING.md](CONTRIBUTING.md) and [CITATION.cff](CITATION.cff).
