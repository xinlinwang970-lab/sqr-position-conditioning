# SQR Position Conditioning

**How can a source-separation model use seating cues while remaining robust to a change in seating layout?**

We study timbre- and position-conditioned string quartet separation through an independent implementation and exploratory experiments.

## Motivation and problem

Two violins share the same timbre label, so position helps identify the target. A fixed training layout also correlates seat with part identity. Our working hypothesis is that this correlation can encourage dependence on the training layout. We investigate the trade-off between reliable target selection and robustness to layout changes.

Given a stereo mixture `x`, a target timbre `t`, and its seat `p`, the model predicts one target stem:

```text
y_hat = f_theta(x, t, p)
```

Targets are Violin 1, Violin 2, Viola, and Cello. Training uses **type2**; evaluation uses type2 and **type1**, which exchanges the seats of Violin 2 and Cello.

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

**1. The historical baseline is sensitive to layout changes.** Its type2 median SDR is **1.636 dB**, compared with **-0.330 dB** on type1: a difference of **1.966 dB**. This establishes layout sensitivity on the recorded test set; its cause requires further controls.

**2. Swap runs show different quality/robustness trade-offs.** These runs use the same 59-item selection, at **425 completed epochs / 25,075 progress updates**.

| Run | Swap probability | Mean SDR | type1 median | type2 median | Difference |
|---|---:|---:|---:|---:|---:|
| P10 | 0.10 | 2.208 | 1.748 | 2.119 | 0.371 |
| P15 | 0.15 | 2.515 | 2.265 | 2.301 | 0.036 |
| P50 | 0.50 | 2.261 | 1.996 | 1.913 | -0.083 |

P15 has the highest observed mean in this comparison and similar scores across layouts. This supports further investigation of swap probability; the single-seed observations do not establish an optimum.

**3. Execution details constrain mechanism claims.** E4's independent dropout masks produce a positive penalty even without changing positions. Manual activation recomputation also updates BatchNorm during backward. These confounds are documented in the [diagnostic checks](results/provenance/behavior_checks.json) and [known limitations](docs/KNOWN_LIMITATIONS.md).

Metric: **gain-dependent waveform SDR, dB**. Mean pools 48 correlated queries; each layout median pools 24. Difference is `type2 median - type1 median`, not a paired statistic. Consider both quality and the magnitude of the difference. The [full report](docs/EXPERIMENT_REPORT.md) includes all runs, per-part scores, and later checkpoints.

## Data and evidence boundaries

Main selection: **59 synthetic SQR type2 training records**, **6 validation IDs x 2 layouts**, and **6 URMP source recordings x 2 layouts** for testing. Four queries per test item give 48 scores. See [data provenance](docs/DATA_PROVENANCE.md).

- One seed and repeated test-set inspection limit conclusions. Future selection should use validation data and independent test material.
- The dense baseline differs in split, route, sampling, and budget; comparisons across intervention families have further confounds.
- E1cosine-B points to unresolved custom data and is excluded from the same-data swap comparison.
- Audio and weights are excluded. Original-paper equivalence and broad generalization to independent live performances remain unestablished.

## Quick start and resources

```bash
python -m pip install -e ".[torch,test]"
python -m pytest -q
python scripts/verify_results.py
```

The last command recomputes saved summaries for 11 result files and 528 rows; it does not rerun inference. See [Quickstart](docs/QUICKSTART.md) for data, training, and evaluation commands, and [Reproducibility](REPRODUCIBILITY.md) for verification scope.

Browse [source](src/sqr_repro), [scripts](scripts), [configs](configs), [manifests](manifests), and [results](results). The [release audit](docs/AUDIT_REPORT.md) records historical corrections.

Reference: Xu et al., [Position also matters!](https://www.isca-archive.org/interspeech_2025/xu25b_interspeech.html), Interspeech 2025. Data: [SQR](https://zenodo.org/records/14793971). Historical `author_release` labels refer to local use of released data, not official models.

Code: [MIT](LICENSE). Third-party assets retain their own terms; see [notices](THIRD_PARTY_NOTICES.md). Contribution and citation information: [CONTRIBUTING.md](CONTRIBUTING.md) and [CITATION.cff](CITATION.cff).
