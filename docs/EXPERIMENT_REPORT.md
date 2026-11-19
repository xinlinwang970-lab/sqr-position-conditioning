# Historical experiment results — corrected release report

Scores below are recomputed from preserved per-query JSON, not newly generated from audio/checkpoints. The metric is gain-dependent waveform SDR in dB. Single seed 2026, repeated URMP test inspection, different data/routes/schedules and retained numerical confounds limit interpretation.

## Reported checkpoints

| Run | Completed epochs | Progress counter | Mean | Median | type1 median | type2 median | Δ pooled medians |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense baseline | 12 | 52,236 | 1.326802 | 1.160451 | -0.329886 | 1.636063 | 1.965948 |
| E1cosine-B (custom, provenance unresolved) | 275 | 16,225 | 1.618700 | 1.436325 | 1.391770 | 1.477811 | 0.086041 |
| E2 | 275 | 16,225 | 1.025635 | 0.814170 | 0.033785 | 1.481356 | 1.447570 |
| E3 | 275 | 16,225 | 1.408641 | 1.178148 | 0.677117 | 1.631006 | 0.953889 |
| P10 | 425 | 25,075 | 2.207668 | 2.008556 | 1.748030 | 2.118945 | 0.370914 |
| P15 | 425 | 25,075 | 2.514978 | 2.300746 | 2.265129 | 2.300746 | 0.035617 |
| P50 | 425 | 25,075 | 2.261170 | 1.923849 | 1.995984 | 1.913190 | -0.082794 |
| E4 | 425 | 25,075 | 2.316744 | 2.192866 | 2.091327 | 2.192866 | 0.101539 |

The subtraction compares two pooled medians. It is not the median paired per-recording difference, and a smaller difference alone does not imply higher separation quality.

A filename `ep274` denotes epoch index 274 after completing 275 epochs; `ep424` denotes 425 completed epochs. The baseline epoch convention is dense enumeration and cannot be compared to random-crop epochs as an equal budget. The saved script counter can advance on a skipped non-finite step.

## Paired layout analysis

This is post-hoc descriptive analysis of existing saved scores, with no new training or model inference. The [derived JSON](../results/layout_pair_summary.json) records input SHA-256 hashes, every paired score/difference, and part-level summaries.

### Pairing and definitions

Pair by `(source_id, part_id)`, with exactly one type1 and one type2 score. For this frozen selection, `composition_id` identifies the source recording; the legacy baseline adapter removes only the first layout prefix from `item_id`. Using the complete layout-prefixed item ID as the pairing key would separate the two layouts instead of matching them.

Each method must contain the same six source IDs, all four canonical parts, 24 pairs and 48 finite scores. Duplicate keys, missing layouts, inconsistent source/layout identities and different source cohorts fail explicitly.

For source `i` and part `k`, let `s1[i,k]` and `s2[i,k]` be type1/type2 SDRs and `d[i,k] = s2[i,k] - s1[i,k]`. All statistics use unrounded scores, in dB:

| Quantity | Definition |
|---|---|
| Overall mean SDR, Q | Mean of all 48 scores |
| Δ pooled medians, G | `median(s2) - median(s1)` |
| Median paired Δ, D | `median(d)` |
| Median paired abs(Δ) | `median(abs(d))` |
| Mean paired abs(Δ) | `mean(abs(d))` |
| Mean paired Δ | `mean(d)` |

Take paired differences before absolute values and aggregation. `median(abs(d))`, `abs(median(d))`, and `abs(median(s2)-median(s1))` measure different things. Smaller absolute paired differences describe less score change, but must be read together with separation quality; consistently poor separation can also have small differences.

### Saved-score summaries

The baseline is its historical 12-epoch endpoint. P10/P15/P50/E4 are the recorded 425-completed-epoch snapshots (epoch index 424, progress counter 25,075). Cross-family protocol differences remain.

| Run | Mean SDR, Q | Δ pooled medians, G | Median paired Δ, D | Median paired abs(Δ) | Mean paired abs(Δ) | Mean paired Δ |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1.326802 | 1.965948 | 0.955149 | 1.346252 | 2.428426 | 2.225927 |
| P10 | 2.207668 | 0.370914 | 0.495831 | 0.789737 | 1.084389 | 0.668102 |
| P15 | 2.514978 | 0.035617 | 0.630755 | 0.751140 | 0.826208 | 0.540444 |
| P50 | 2.261170 | -0.082794 | 0.605020 | 0.680916 | 0.826974 | 0.670765 |
| E4 | 2.316744 | 0.101539 | 0.419219 | 0.607307 | 0.733515 | 0.444917 |

P15's pooled-median difference is 0.035617 dB, while its median absolute paired difference is 0.751140 dB. Nearly equal pooled medians therefore do not imply near-zero changes for individual source-part pairs. P50 has a smaller median absolute paired difference (0.680916 dB), while the mean absolute differences for P15/P50 are very close (0.826208/0.826974 dB). These are different descriptive criteria, not evidence of an optimal probability.

P50's negative pooled-median difference (-0.082794 dB) also does not imply that a typical matched target performs better in type1. Its median paired delta is +0.605020 dB, mean paired delta is +0.670765 dB, and 19 of 24 pairs have higher type2 scores.

### E4 versus P15

At these recorded snapshots, E4 has a smaller median absolute paired layout difference (0.607307 vs. 0.751140 dB) and lower overall mean SDR (2.316744 vs. 2.514978 dB). This is an observed quality/layout-change trade-off. The part-level means below pool both layouts; each part contains six source pairs / 12 correlated scores.

| Part | P15 mean SDR | E4 mean SDR |
|---|---:|---:|
| violin1 | 2.618356 | 2.687954 |
| violin2 | 0.936205 | 0.796811 |
| viola | 0.955359 | 1.104731 |
| cello | 5.549990 | 4.677479 |

E4's viola mean is higher (1.104731 vs. 0.955359 dB). This is a supplementary part-level observation. The all-part absolute-difference metric must not be used as a viola-specific stability statistic; the JSON contains separate six-pair summaries for each part.

### Interpretation boundaries

The 24 pairs are clustered within six source recordings, and methods reuse those recordings. No significance tests, confidence intervals or population-level claims are made. Future inference should account for source-level clustering. For Violin 1 and Viola, the target seat remains fixed while the surrounding source configuration changes. For Violin 2 and Cello, target timbre-position combinations also change. The synthetic-to-recorded-source domain shift is a separate factor.

There is no matched `p_swap=0` control. The historical dense baseline differs in data selection, route, sampling and budget, so it cannot isolate the effect of swap. E1cosine-B is excluded from this same-data analysis because its custom-data provenance is unresolved.

Attribution of E4's behavior to the regularizer requires a matched `lambda=0` control and control of dropout effects. Independent controller dropout masks can produce a positive penalty even for an identity position permutation. P10/P15/P50 use manual activation-recomputation wrappers that update BatchNorm during backward, whereas E4 does not. A mechanism comparison must align these execution settings; these saved-score statistics alone do not identify a causal position-invariance effect.

Recompute or check the deterministic summary with:

```bash
python scripts/summarize_layout_pairs.py
python scripts/summarize_layout_pairs.py --check
```

## Per-part medians at the reported checkpoints

These pool both layouts for each part. All four part queries share the same mixtures, so their scores are not independent.

| Run | Violin 1 | Violin 2 | Viola | Cello |
|---|---:|---:|---:|---:|
| Dense baseline | 2.336473 | -0.808835 | 0.945001 | 2.238691 |
| E1cosine-B (custom, provenance unresolved) | 1.702659 | 0.857860 | 0.262077 | 3.098098 |
| E2 | 1.852161 | -1.487975 | 0.728151 | 1.983418 |
| E3 | 1.838569 | 0.736922 | 0.728485 | 1.872436 |
| P10 | 2.505094 | 0.415632 | 0.811299 | 4.603245 |
| P15 | 2.639378 | 0.948425 | 0.827818 | 5.493811 |
| P50 | 2.164935 | 0.836940 | 1.034435 | 4.673257 |
| E4 | 2.497298 | 0.522183 | 1.308691 | 4.411587 |

## Later final-result files

The pending release also contained final results from later updates. They are retained separately to avoid mixing checkpoints in one comparison.

| Final file / run | Mean | type1 median | type2 median |
|---|---:|---:|---:|
| [PureDrop_E3_300ep_p025_test_results.json](../results/test_results/PureDrop_E3_300ep_p025_test_results.json) | 1.014746 | 0.342062 | 1.686057 |
| [PureNoise_E2_300ep_s010_test_results.json](../results/test_results/PureNoise_E2_300ep_s010_test_results.json) | 0.793954 | 0.428687 | 1.236596 |
| [PureSwap_E4_1000ep_pinv_test_results.json](../results/test_results/PureSwap_E4_1000ep_pinv_test_results.json) | 2.239984 | 1.808942 | 2.132388 |

E3's 1.686057 type2 median belongs to its 300-epoch final result, while the 275-epoch checkpoint yields 1.631006. The earlier report mixed these snapshots. E2/E3 final files correspond to the script's 300-epoch endpoint and E4 to its 1000-epoch endpoint by available run records; these final weights were not among the eight checkpoint files safely inspected in the local audit. No new weight-to-score verification was performed.

## Data and protocol boundaries

The dense baseline uses historical 60/5 selection metadata, exhaustive crop slabs, a different conditioning route and 52,236 progress updates. P10/P15/P50/E2/E3/E4 use 59 type2 training IDs. E2/E3 have constant learning rate; the cosine variants have a different schedule and horizon. Manual activation recomputation in P10/P15/P50/E2/E3 changes BatchNorm running-stat updates relative to E4. These differences prevent treating the whole table as a matched one-factor ablation.

E1cosine-B is an especially important provenance exception. Its safely inspected checkpoint config refers to custom `quartet-data/train` and a custom manifest; its result JSON refers to another server path. The original custom manifests and exact source-audio identity are absent. The script describes oversampling a small custom dataset, but does not establish its actual selected record count. This run is archived as a separate historical observation, not the p=0.25 member of a matched SQR dose sweep.

## Interpretation

P15 has the largest observed mean among the main 425-epoch snapshots listed here. This is a descriptive observation on repeatedly inspected recordings, not proof that p=0.15 is optimal or that shortcut learning was causally removed. No untouched test selection, multi-seed uncertainty interval or matched data/control result is available to support those conclusions. E4's independent dropout masks further confound a position-only regularization interpretation.

The test grid has six source recordings, two rendered layouts and four queried parts, giving 48 correlated rows per file. They are not 48 independent compositions. The spatialized URMP test set also does not directly validate robustness to unseen simultaneous concert performances. Original SQR data and custom MuseScore data require separate provenance and claims.

Use validation for future checkpoint/hyperparameter decisions. Freeze data and numerical protocol, run multiple seeds, and reserve new independent test material for confirmatory conclusions. See [data provenance](DATA_PROVENANCE.md), [design](E1-4_DESIGN.md) and [known limitations](KNOWN_LIMITATIONS.md).

## Verification

Run `python scripts/verify_results.py` from the repository root. It checks all 11 original result files, 528 rows, complete query grids, saved scalar summaries and preserved evidence hashes. Full precision values and row sources are in `results/recomputed_summary.json`; the main table is also available as `results/recomputed_summary.csv`. This command does not perform inference.
