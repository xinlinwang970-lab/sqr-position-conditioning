# Data Acquisition Guide

This repository does **not** include the audio data (too large for GitHub).
You need to obtain the data separately and place it in the expected layout.

## Required Data

### 1. Training & Validation: StringQuD Reverb Two-Seating

- **Source**: StringQuD dataset (reverb two-seating release)
- **Train split**: 59 works, type2 seating only
- **Val split**: 12 records (6 works × 2 seating layouts: type1 + type2)

### 2. Test: URMP Real Recordings

- **Source**: URMP dataset (real ensemble recordings)
- **Test split**: 12 records (6 works × 2 seating layouts: type1 + type2)

## Expected Directory Layout

```
<data_root>/
├── manifests/
│   ├── sqr_paper_split_train.jsonl    # 59 type2 works (included in this repo)
│   ├── sqr_paper_split_val.jsonl      # 12 records (included in this repo)
│   └── sqr_paper_split_test.jsonl     # 12 URMP records (included in this repo)
├── train/
│   └── stringqud_reverb_two_seating/
│       └── <work_id>/
│           ├── mix_<work_id>.wav
│           └── <stem_files>.wav
├── val/
│   └── stringqud_reverb_two_seating/
│       └── ...
└── test/
    └── URMP_test_set/
        └── ...
```

## Manifests (Included)

The `manifests/` folder in this repo contains the three JSONL manifest files.
Each record specifies:

```json
{
  "item_id": "2-1-aasesdeath",
  "composition_id": "1-aasesdeath",
  "layout_id": "type2",
  "mixture": "train/stringqud_reverb_two_seating/2-1-aasesdeath/mix_1-aasesdeath.wav",
  "stems": {
    "violin1": "train/.../4_Violin_1.wav",
    "violin2": "train/.../3_Violin_2.wav",
    "viola": "train/.../2_Viola.wav",
    "cello": "train/.../1_Cello.wav"
  },
  "position_ids": {"violin1": 3, "violin2": 2, "viola": 1, "cello": 0},
  "num_samples": 9569700,
  "sample_rate": 44100
}
```

## Configuring the Data Path

All configs and scripts reference `data.root`. Update this to point to your
`<data_root>`:

```yaml
# In config YAML
data:
  root: /path/to/your/data_root
```

## Audio Format

- Sample rate: 44100 Hz
- Channels: 2 (stereo)
- Format: WAV (float32)
- Typical duration: ~200s per work (train), ~30-60s per URMP test item
