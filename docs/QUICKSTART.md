# Quickstart

This guide documents the portable entry points for the historical experiments. Read the [research overview](../README.md), [data provenance](DATA_PROVENANCE.md), and [known limitations](KNOWN_LIMITATIONS.md) before interpreting results.

## Installation and checks without dataset audio

From the repository root, run these commands in your Python environment:

```bash
python -m pip install -e ".[torch,test]"
python -m pytest -q
python scripts/verify_results.py
```

For GPU use, first install a PyTorch build compatible with your platform. The extras install PyTorch and test dependencies; the base package alone does not install PyTorch. The declared Python support range does not mean every combination has been verified. Local release-preparation records are in `results/validation/`, and GitHub Actions provides separate Linux CPU checks. Full GPU training and replication of the historical Linux training environment were not rerun during release preparation.

Result verification checks saved per-query scores, summary calculations, and evidence-file checksums. It does not run inference. Audio, SoundFonts, third-party dataset archives, and model weights are not bundled. The separate MuseScore workflow produces a different dataset variant and is not the source of the historical SQR scores.

## Rebuild manifests from the SQR archives

Obtain both archives from [SQR on Zenodo](https://zenodo.org/records/14793971) and extract them beneath one data root. You can retain the original directory structure; validation audio does not need to be moved into a separate directory.

```bash
python scripts/build_data_split_manifests.py --data-root data --output-dir manifests/local
```

The builder selects the frozen IDs, derives position labels, checks 44.1 kHz stereo audio and matching lengths, and hashes complete WAV file bytes. The main selection uses 59 type2 training items and the six validation IDs listed by the dataset publisher. Validation and test contain both layouts.

No WAV is moved, resampled, or normalized. The command stops if audio differs from the frozen manifests. Explicitly passing `--allow-hash-differences` produces a difference report; report such audio as a separate dataset variant.

## Preflight validation and training

This example launches the historical P15 protocol, without treating its probability as an established optimum:

```bash
python scripts/train_PureSwap_E1cosine-P15_1000ep.py --data-root data --train-manifest manifests/local/sqr_paper_split_train.jsonl --val-manifest manifests/local/sqr_paper_split_val.jsonl --test-manifest manifests/local/sqr_paper_split_test.jsonl --run-root runs/new_P15 --check-only
```

After preflight passes, remove `--check-only` to start training. Script constants set batch size 4, seed 2026, a 1000-epoch target, and swap probability 0.15. Full training is computationally expensive. `release_launch.json` records the entry point, configuration, paths, and actual manual activation-recomputation setting. Automatic resume continues from existing checkpoints, so use a new `--run-root` for a new experiment.

All eight training entry points provide `--help`, `--config`, `--data-root`, `--run-root`, three manifest-path options, and `--check-only`. The dense baseline also requires `--cache-root`. Custom-data E1cosine-B requires explicit train/validation manifests and separate roots; this package cannot recover its original dataset identity.

## Prepare the historical baseline cache

These commands use the separate historical 60-training-ID/five-validation-ID selection. The cache preserves input amplitudes. The recovered manifests alone do not establish identity with the old training audio/cache.

```bash
python scripts/build_dense_cache.py --data-root data --output-root data/cache_baseline_new --train-manifest manifests/historical_baseline/train.jsonl --val-manifest manifests/historical_baseline/val.jsonl --test-manifest manifests/historical_baseline/test.jsonl
python scripts/train_author_release_12ep_warmup_cosine.py --data-root data --cache-root data/cache_baseline_new --run-root runs/new_baseline --check-only
```

The builder expects WAVs at the historical manifest paths. If your extraction differs, adjust paths in copies of the manifests and explicitly pass all three options. The tool does not copy audio or reconstruct the paper's synthesis and RMS-processing pipeline.

## Evaluate existing weights

Provide a checkpoint matching the relevant run. Original checkpoint inspection hashes are in `results/provenance/checkpoint_verification.json`.

```bash
python scripts/evaluate_checkpoint.py --config configs/PureSwap_E1cosine-P15_1000ep.yaml --checkpoint weights/P15_checkpoint_025075.pt --manifest manifests/local/sqr_paper_split_test.jsonl --root data --output runs/new_eval/P15.json
```

The metric is gain-dependent waveform SDR:

```text
10*log10(sum(reference**2)/(sum((estimate-reference)**2)+1e-8))
```

It does not perform SI-SDR scale projection. Legacy full-length inference uses 50% periodic Hann overlap-add with documented endpoint and short-input behavior. See [reproducibility](../REPRODUCIBILITY.md) for historical restart limitations, environment scope, and the distinction between saved-score verification and a new full inference run.
