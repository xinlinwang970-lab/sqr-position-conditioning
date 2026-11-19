# SQR Position Conditioning: Independent Implementation and Exploratory Experiments

This repository contains an independent implementation for separating four string parts, experiment configurations, data manifests, historical per-query metrics, and tools for recomputing saved summaries. The results support pipeline validation and exploratory comparisons. Single-seed runs, repeated test-set inspection, differences in training protocols, and incomplete data provenance limit the conclusions.

The reference paper is Xu et al., [Position also matters! Separating Same Instruments in String Quartet using Timbral and Positional Cues](https://www.isca-archive.org/interspeech_2025/xu25b_interspeech.html), Interspeech 2025. The original dataset is available at [String Quartet Set with Reverb](https://zenodo.org/records/14793971). This project is independent of the original authors. The `author_release` label in filenames identifies local experiments using the authors' released data; it does not identify an official model or checkpoint.

## Contents and scope

- `src/`, `scripts/`, `configs/`: the model, historical training/evaluation entry points, and portable data-preparation tools.
- `manifests/`: 59 type2 training records, 12 validation records, and 12 test records. The historical baseline's separate 60/10/12 record selection is also retained.
- `results/checkpoint_evals/`, `results/test_results/`: the 11 original result JSON files, with their scores unchanged.
- `results/provenance/`: training logs, available intermediate test evaluations, local inspection checksums for eight checkpoints, original configurations, and file provenance.
- `tests/`: checks for the model, conditioning routes, data loading, evaluation, synthetic training, and release tools.

Audio, SoundFonts, third-party dataset archives, and model weights are not included. The historical SQR experiments consume the authors' released spatialized audio. The paper describes synthetic sources from EnsembleSet/BBCSO. The separate MuseScore data-generation workflow produces a different dataset variant and must be identified separately from the SQR data used here. [Paper dataset description](https://www.isca-archive.org/interspeech_2025/xu25b_interspeech.pdf)

**Read the [known limitations](docs/KNOWN_LIMITATIONS.md) and [data provenance](docs/DATA_PROVENANCE.md) first.** In particular, the E1cosine-B checkpoint points to custom `quartet-data`; this run cannot be treated as a control that uses the same 59 SQR records and changes only the augmentation.

## Installation and checks without dataset audio

From the downloaded repository root, run these commands in your Python environment:

```bash
python -m pip install -e ".[torch,test]"
python -m pytest -q
python scripts/verify_results.py
```

For GPU use, first install a PyTorch build compatible with your platform. The `.[torch,test]` extras install PyTorch and test dependencies; the base package alone does not install PyTorch. The declared Python support range does not mean that every environment combination has been verified. The recorded local release-preparation environment is documented in `results/validation/`. Full GPU training and replication of the historical Linux training environment were not rerun during release preparation; GitHub Actions provides separate Linux CPU checks.

The result-verification command checks saved per-query scores, summary calculations, and evidence-file checksums. It does not run model inference.

## Rebuilding manifests from the SQR archives

Obtain both archives from the dataset page and extract them beneath one data root. You can retain their original directory structure; validation audio does not need to be moved into a separate directory. The builder selects the frozen record IDs bundled with this repository, derives position labels, checks 44.1 kHz stereo audio and matching lengths, and computes SHA-256 hashes of complete WAV file bytes.

```bash
python scripts/build_data_split_manifests.py --data-root data --output-dir manifests/local
```

The validation selection follows the [six-item list published on Zenodo](https://zenodo.org/records/14793971). The training template contains only type2 records; validation and test contain both layouts. The tool does not move, resample, or normalize audio. It stops if the audio differs from the frozen manifests. Explicitly passing `--allow-hash-differences` produces a difference report; any such audio must be reported as a separate dataset variant.

## Preflight validation and training

```bash
python scripts/train_PureSwap_E1cosine-P15_1000ep.py --data-root data --train-manifest manifests/local/sqr_paper_split_train.jsonl --val-manifest manifests/local/sqr_paper_split_val.jsonl --test-manifest manifests/local/sqr_paper_split_test.jsonl --run-root runs/new_P15 --check-only
```

After preflight validation passes, remove `--check-only` to start training under this historical protocol. Its script constants set batch size 4, seed 2026, a 1000-epoch target, and swap probability 0.15. Full training is computationally expensive and was not started during release preparation. Each launch saves `release_launch.json`, recording the entry point, configuration, manifest paths, and actual manual activation-recomputation setting. Automatic resume continues from existing checkpoints in the output directory, so use a new `--run-root` for a new experiment.

All eight training entry points provide `--help`, `--config`, `--data-root`, `--run-root`, three manifest-path options, and `--check-only`. The historical baseline also requires an explicit `--cache-root`. The custom-data E1cosine-B entry point requires explicit train/validation manifests and separate data roots; this package cannot recover its original dataset identity.

## Dense-cache preparation for the historical baseline

The following commands use the historical 60-training-ID/five-validation-ID selection to create a cache that preserves input sample amplitudes. Those selection manifests were recovered from an older normalization workspace. The manifests alone do not establish that the audio or cache matches the historical training inputs.

```bash
python scripts/build_dense_cache.py --data-root data --output-root data/cache_baseline_new --train-manifest manifests/historical_baseline/train.jsonl --val-manifest manifests/historical_baseline/val.jsonl --test-manifest manifests/historical_baseline/test.jsonl
python scripts/train_author_release_12ep_warmup_cosine.py --data-root data --cache-root data/cache_baseline_new --run-root runs/new_baseline --check-only
```

The cache builder expects WAVs at the paths in these historical manifests. If your extraction uses different paths, adjust the relative paths in copies of the manifests and explicitly pass all three manifest options. The tool does not copy audio or reproduce the paper's synthesis and RMS-processing pipeline.

## Evaluating existing weights

Provide a checkpoint matching the relevant run. Checksums from local inspection of the original weights are recorded in `results/provenance/checkpoint_verification.json`.

```bash
python scripts/evaluate_checkpoint.py --config configs/PureSwap_E1cosine-P15_1000ep.yaml --checkpoint weights/P15_checkpoint_025075.pt --manifest manifests/local/sqr_paper_split_test.jsonl --root data --output runs/new_eval/P15.json
```

Evaluation uses the historical full-length sliding-window procedure. The metric is gain-dependent waveform SDR: `10*log10(sum(reference**2)/(sum((estimate-reference)**2)+1e-8))`. It does not perform the optimal scale projection used by SI-SDR and is not interchangeable with other SDR implementations the paper may have used. The legacy Hann overlap-add endpoint and short-input behavior are documented; these scores cannot be claimed as directly comparable to the paper's results.

## Historical results

All values are in dB. Mean is the mean over 48 correlated query rows; type1/type2 values are medians over 24 queries each. A filename ending in `ep274` uses a zero-based epoch index and corresponds to 275 completed epochs. Data selections, conditioning routes, schedulers, and training budgets differ across runs.

| Run | Update counter | Mean | type1 median | type2 median |
|---|---:|---:|---:|---:|
| Historical dense baseline, 12 epochs | 52,236 | 1.3268 | -0.3299 | 1.6361 |
| E1cosine-B, unresolved custom-data provenance, 275 epochs | 16,225 | 1.6187 | 1.3918 | 1.4778 |
| E2, 275-epoch snapshot | 16,225 | 1.0256 | 0.0338 | 1.4814 |
| E3, 275-epoch snapshot | 16,225 | 1.4086 | 0.6771 | 1.6310 |
| P10, 425-epoch snapshot | 25,075 | 2.2077 | 1.7480 | 2.1189 |
| P15, 425-epoch snapshot | 25,075 | 2.5150 | 2.2651 | 2.3007 |
| P50, 425-epoch snapshot | 25,075 | 2.2612 | 1.9960 | 1.9132 |
| E4, 425-epoch snapshot | 25,075 | 2.3167 | 2.0913 | 2.1929 |

Per-part metrics for these snapshots and the later final results are listed in the [experiment report](docs/EXPERIMENT_REPORT.md). The values were recomputed from saved JSON files; release preparation did not regenerate them from checkpoints and full-length audio. P15 has the highest observed mean among the main 425-epoch snapshots shown here. This observation does not establish an optimal probability, confirm the proposed intervention mechanism, or demonstrate robust generalization across independent works or real quartet performances. Future studies should freeze validation-based model selection, match experimental protocols, add random seeds, and use new independent test material.

## Reproducibility and citation

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) distinguishes verified calculations from unverified replication layers. The [experiment design](docs/E1-4_DESIGN.md) documents historical settings, and the [release audit](docs/AUDIT_REPORT.md) records the basis for the corrections.

Project code retains the pending release's MIT license. That license does not automatically grant rights to third-party data or assets; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Cite the original authors separately when using their paper or dataset.
