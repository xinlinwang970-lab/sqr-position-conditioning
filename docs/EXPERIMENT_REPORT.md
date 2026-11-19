# Historical experiment results — corrected release report

Scores below are recomputed from preserved per-query JSON, not newly generated from audio/checkpoints. The metric is gain-dependent waveform SDR in dB. Single seed 2026, repeated URMP test inspection, different data/routes/schedules and retained numerical confounds limit interpretation.

## Reported checkpoints

| Run | Completed epochs | Progress counter | Mean | Median | type1 median | type2 median | type2 minus type1 |
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
