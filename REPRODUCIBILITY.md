# Reproducibility Guide

## Prerequisites

1. Clone this repository
2. Set up the environment (see `ENVIRONMENT.md`)
3. Obtain the StringQuD reverb two-seating dataset and URMP test set
4. Place data under a root directory with the expected manifest structure

## Data layout

```
<data_root>/
├── manifests/
│   ├── sqr_paper_split_train.jsonl    # 59 type2 works
│   ├── sqr_paper_split_val.jsonl      # 12 records (6 type1 + 6 type2)
│   └── sqr_paper_split_test.jsonl     # 12 URMP records (6 type1 + 6 type2)
├── train/
│   └── stringqud_reverb_two_seating/
│       └── <work_id>/
│           ├── mix_<work_id>.wav
│           ├── 1_Violin_1.wav  (or similar)
│           ├── 2_Viola.wav
│           ├── 3_Violin_2.wav
│           └── 4_Cello.wav
├── val/
│   └── stringqud_reverb_two_seating/
│       └── ...
└── test/
    └── URMP_test_set/
        └── ...
```

Each manifest record contains:
- `item_id`, `composition_id`, `layout_id` (type1/type2)
- `mixture`: relative path to mixture WAV
- `stems`: dict of part → relative path to stem WAV
- `position_ids`: dict of part → seat index (0-3)
- `num_samples`, `sample_rate`

## Running experiments

### 1. Train a model

```bash
export PYTHONPATH=/path/to/src
export CUDA_VISIBLE_DEVICES=0

# g24 B/MDP (best architecture, no intervention)
python scripts/train_growth24_projected_b_v1_mdp_1000ep.py

# g24 B/MDP with position-swap intervention (p=0.15, wcos)
python scripts/train_growth24_projected_b_v1_mdp_swap15.py

# g24 B/MDP with position-swap (p=0.15, const LR — paper-strict)
python scripts/train_growth24_projected_b_v1_mdp_constlr_swap15.py
```

### 2. Evaluate a checkpoint on URMP test

```bash
python scripts/evaluate_checkpoint.py \
    --config configs/<config>.yaml \
    --checkpoint runs/<run_name>/checkpoint_final.pt \
    --manifest <data_root>/manifests/sqr_paper_split_test.jsonl \
    --root <data_root> \
    --tag final
```

### 3. Audit architecture parameters

```bash
python scripts/audit_architecture_params.py configs/<config>.yaml
```

## Evaluation protocol: `legacy_v1`

All checked-in result-bearing configs select `evaluation_protocol: legacy_v1`.
The training entrypoints and `scripts/evaluate_checkpoint.py` call the same
implementation in `src/sqr_repro/evaluation.py`; the protocol is not copied
independently into each script.

| Element | Executed behavior |
|---|---|
| Metric | scale-dependent waveform energy ratio, `10*log10(sum(ref²) / (sum((estimate-ref)²) + 1e-8))` |
| Segment length | `(num_frames - 1) * hop_length` samples |
| Requested overlap | `0.5` |
| Stride | `int(segment_samples * 0.5)`; deliberately not aligned to an STFT-hop multiple |
| Blend | periodic Hann window (`periodic=True`) |
| Blend denominator | `clamp(weight, min=1e-8)` |
| Short input | pad to one segment for inference; score the common estimate/reference length |
| Aggregates | mean, median, type1 median, type2 median, and query count |

`legacy_v1` is a compatibility label for the protocol that generated the
reported results. It is not a claim that this is the paper authors' SDR
implementation, and it does not introduce a corrected alternative protocol.

## Key hyperparameters (all experiments)

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=5e-4, betas=[0.9,0.999], eps=1e-8, weight_decay=0) |
| Batch size | 4 |
| Training length | 1000 epochs (59000 updates) |
| Precision | fp32 |
| Loss | packed_complex_mse |
| STFT | n_fft=2048, hop=1024, hann window |
| Crop | 128 frames (g24) or 176 frames (dense124), random per epoch |
| Seed | 2026 |
| Validation | Every 25 epochs (full-length WaveSDR on 12 val records) |
| Test | Final-only (full-length WaveSDR on 12 URMP records) |

## Auto-resume

Training scripts automatically resume from the latest `checkpoint_*.pt` in the
run directory if interrupted.

## Reproducibility notes

- All experiments use seed=2026 (single seed). Multi-seed validation is left
  as future work.
- Random crop offsets are deterministic: SHA-256(seed, epoch, item_id) →
  crop start. Same seed + epoch + item always produces the same crop.
- DataLoader shuffle uses generator seed = SEED + epoch, ensuring different
  shuffle order per epoch but full reproducibility.
