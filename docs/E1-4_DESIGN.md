# Ablation Matrix Design: Anti-Shortcut Interventions

This document describes the design rationale and protocol for the eight
experiments shipped in this repository. It is the reference for the
experiment matrix referenced by
[`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) and audited by
[`AUDIT_REPORT.md`](AUDIT_REPORT.md).

---

## 1. Background

A position-conditioned string quartet source separation model accepts a
discrete seating index `p ∈ {0,1,2,3}` and produces the isolated stem for
the instrument seated at that position. Internally, a controller MLP maps
a fused timbre–position embedding `e_m` to GPoCM gate parameters
`W(e_m), b(e_m)` that modulate the decoder.

When training data uses a deterministic seat→part mapping `L`, one risk
is that the model may rely too heavily on the condition labels and underuse
audio input `x`. We call this the **position-shortcut hypothesis**; it is
not implied by the architecture. In the `mid_dual_paper` experiments, the
decoder controller receives `e_m = 0.5(e_t + e_p)`, and GPoCM multiplies
the generated gate with an audio activation `h`. The route therefore does
not reduce to a position-only lookup table. Layout and domain results can be
consistent with a shortcut, but do not directly measure audio dependence.

The evaluation probes whether performance changes when:

- **Combinatorial generalization**: the test layout `L'` differs from
  training.
- **Forward transfer**: the test domain shifts from synthetic
  (MuseScore/StringQuD) to real recordings (URMP).

The ablation matrix asks: **how do these interventions change measured
combinatorial-generalization and forward-transfer scores?** The experiments
do not directly measure whether the network ignores mixture audio.

---

## 2. Common Protocol

All experiments in the matrix share the following fixed elements. Only
the intervention mechanism and its strength vary.

| Element | Value |
|---|---|
| Model | `mid_dual_paper` route, `internal_channels=124`, **30,441,288 params (30.44M; paper reports 46M, rounded)** |
| Condition injection | Decoder-only GPoCM gate modulation |
| Training data | 59 StringQuD works, `type2` seating only |
| Optimizer | Adam, batch=4, 59 updates/epoch |
| Seed | 2026 (unified for data crop, model init, sampler) |
| Audio | 44.1 kHz stereo, n_fft=2048, hop=1024, 128 frames/crop |
| Loss | Packed complex MSE on STFT |
| Inference | Full-length `legacy_v1` 50 %-overlap sliding window, WaveSDR (dB) |
| Test set | URMP StringQuartet, 6 works × 2 seatings × 4 stems = 48 samples |

### Seating layouts

```
type1 (unseen at test): {v2→0, va→1, c→2, v1→3}
type2 (training layout): {c→0,  va→1, v2→2, v1→3}
```

Only `type2` appears in training; the test set contains both, so `type1`
items probe combinatorial generalization.

### Evaluation metrics

- `SDR_test` — overall mean WaveSDR on the 48-sample URMP test set
  (forward transfer, synthetic→real).
- `SDR_type1` — median WaveSDR on the 24 unseen-layout (type1) items
  (combinatorial generalization).
- `SDR_type2` — median WaveSDR on the 24 in-distribution (type2) items.
- `Δ_cg = SDR_type2 − SDR_type1` — generalization gap (smaller is better).

---

## 3. Experiment Matrix

### 3.1 Baseline (no intervention)

| ID | Name | Mechanism | LR schedule | Epochs | Role |
|---|---|---|---|---|---|
| B0 | `author_release_12ep_warmup_cosine` | none | warmup_cosine | ~52 k upd | Author-release configuration baseline |

**Note**: B0 uses the `split_stage_paper` route (encoder receives `e_t`,
middle `e_m`, decoder `e_p`) and a 12-work training set, not the
`mid_dual_paper` + 59-work protocol of the rest of the matrix. Its shipped
config uses width and TFC growth 124, so it is a reference point for that
configuration, not a reconstruction of the paper's stated growth-24 model
or a same-protocol no-intervention control. A same-protocol TruePure
baseline (`mid_dual_paper`, 59 works, `p_swap=0`, warmup_cosine, 1000 ep)
is listed as a known gap in `AUDIT_REPORT.md`.

### 3.2 E1 — Position Swap dose sweep (4 ablations)

The E1 mechanism (position swap) is evaluated as a 4-point dose sweep.
All arms share the `mid_dual_paper` architecture, warmup-cosine LR,
seed 2026, 1000-epoch horizon, and the **59-work training protocol**
(`manifests/sqr_paper_split_train.jsonl`).

| ID | Name | `p_swap` | Epochs | Role |
|---|---|---|---|---|
| P10 | `PureSwap_E1cosine-P10_1000ep` | 0.10 | 1000 | Low dose |
| **P15** | `PureSwap_E1cosine-P15_1000ep` | 0.15 | 1000 | Mid dose (recommended) |
| P25 | `PureSwap_E1cosine-P25_1000ep` | 0.25 | 1000 | Paper-default dose (= original E1cosine) |
| P50 | `PureSwap_E1cosine-P50_1000ep` | 0.50 | 1000 | High dose |

> `P25` is the **original `PureSwap_E1cosine_1000ep`** run — the
> paper-default p=0.25 dose on the 59-work protocol. Its config and
> results are shipped under the P25 name for consistency with the P-series
> dose naming.

**Mechanism**: With probability `p_swap`, the position id `p` is replaced
by `σ(p)` where `σ` is a random permutation of `{0,1,2,3}`. The model
input is perturbed while the sample's target remains fixed. This tests
robustness to an unreliable position condition; it may encourage greater
use of the audio, but does not guarantee that behavior.

**Design rationale**: the dose sweep isolates a single variable
(`p_swap`) while holding architecture, data, LR, seed, and evaluation
epoch fixed. It is the best-controlled comparison in the matrix, and
together with E4 compares an input perturbation with a gate-parameter
regularizer. Changes in SDR and layout gap are measured outcomes; they do
not by themselves establish shortcut suppression.

### 3.3 Embedding-side interventions

| ID | Name | Mechanism | Strength | LR | Epochs | Role |
|---|---|---|---|---|---|---|
| E2 | `PureNoise_E2_300ep_s010` | Add Gaussian noise to `e_p` | σ=0.1 | const 5e-4 | 300 | Blurred hint |
| E3 | `PureDrop_E3_300ep_p025` | Zero out `e_p` | p_drop=0.25 | const 5e-4 | 300 | Missing hint |

**Mechanism (E2)**: a forward hook adds `σ·N(0, I)` to the position
embedding `e_p` per sample, blurring the discrete token into a continuous
region.

**Mechanism (E3)**: with probability `p_drop`, `e_p` is set to zero (so
`e_m = 0.5·e_t`), simulating a missing hint and forcing the model to
learn a no-hint fallback mode.

**Protocol caveat**: E2/E3 use a constant LR and a 300-epoch screening
horizon, whereas the swap family uses warmup-cosine + 1000 epochs. This
is a known protocol misalignment flagged in `AUDIT_REPORT.md` —
cross-family quantitative comparison is confounded by LR schedule and
training horizon.

### 3.4 E4 — Gate-parameter invariance regularization (counterpart of E1)

| ID | Name | Mechanism | Strength | LR | Epochs | Role |
|---|---|---|---|---|---|---|
| E4 | `PureSwap_E4_1000ep` | Penalize `‖W(p) − W(σ(p))‖²` | λ=1.0 | warmup_cosine | 1000 | Gate-side constraint |

**Mechanism**: in addition to the separation loss, E4 adds a
regularization term that penalizes the L2 distance between the GPoCM gate
parameters produced for the true position `p` and for a permuted position
`σ(p)`:

```
L_E4 = ℓ(f(x,t,p), y) + λ · (1/n) Σ_i [ ‖W_i(p) − W_i(σ(p))‖²
                                     + ‖b_i(p) − b_i(σ(p))‖² ]
```

Unlike swap (E1), the model input still uses the correct `p`; the
permutation `σ(p)` only enters the regularization term. E4 therefore
penalizes differences between gate parameters under these paired position
conditions. It does not establish that those differences caused a shortcut.

**Relationship to E1**: E1 (position swap) and E4 (gate invariance) are
compared in the same experiment family. They share the position-permutation
signal `σ(p)` — they are not fully independent paths — but act at
different points: E1 perturbs the model input, E4 constrains gate
parameters. Their measured outcomes do not prove either mechanism removes
a position shortcut.

**Shared signal caveat**: a truly independent control (e.g., gate-magnitude
L2 reg `λ‖W‖²` with no permutation) is listed as future work in
`AUDIT_REPORT.md`.

### 3.5 Excluded experiment: E1cosine-B
### 3.5 Data-ablation variant: E1cosine-B1 (segmented dataset)

| ID | Name | Mechanism | Data | `p_swap` | LR | Epochs | Role |
|---|---|---|---|---|---|---|---|
| E1B1 | `PureSwap_E1cosine-B1_1000ep` | swap | 60 segments (12 works × 5) | 0.25 | warmup_cosine | 1000 | Data-preparation ablation |

E1cosine-B1 tests the effect of data preparation: 12 movements split into
60 equal-length segments (no oversampling), compared to the 59 distinct
works used by the main E1 dose sweep. Same architecture, LR, and dose as
P25. The result (test mean +1.61 vs P25's +2.35) isolates the value of
work-level diversity. See `EXPERIMENT_REPORT.md` § 4.

### 3.6 Excluded experiment: E1cosine-B

The earlier `E1cosine-B` experiment (p=0.25, **12 full works with
crop-slot oversampling**, ~4.92×/epoch) is **not** part of this matrix and
is excluded from evaluation. Its per-epoch cropping generator differs from
both the 59-work protocol (P10/P15/P25/P50) and the 60-segment
no-oversampling E1B1, so its results are not comparable to either family.

---

## 4. Intervention Spectrum

The four mechanisms form a "hint-reliability degradation" spectrum from
light to heavy:

```
B0  no intervention           (fully reliable hint)
 → E2 embedding noise          (blurred: token → continuous region)
 → E1 position swap            (lie: p_swap fraction of steps point to wrong seat)
 → E3 embedding drop           (absent: p_drop fraction of steps have no hint)
 → E4 gate invariance reg      (penalty on gate differences under position permutation)
```

Each cell isolates one mechanism setting. The reported runs show:

> Position-swap runs have near-zero layout gaps at the higher tested doses.
> E4 has similar overall SDR with a wider final layout gap. Noise/drop runs
> score lower at the reported checkpoints, but use different training
> schedules and horizons.

---

## 5. File Index

Each experiment has a matched config / script / results triple.
The E1 dose sweep (P10/P15/P25/P50) and E4 are the two arms of the
anti-shortcut comparison.

| ID | Config | Script | Results |
|---|---|---|---|---|---|
| B0 | `configs/author_release_12ep_warmup_cosine.yaml` | `scripts/train_author_release_12ep_warmup_cosine.py` | `results/test_results/author_release_12ep_warmup_cosine_test_results.json` |
| P10 | `configs/PureSwap_E1cosine-P10_1000ep.yaml` | `scripts/train_PureSwap_E1cosine-P10_1000ep.py` | `results/checkpoint_evals/P10_test_ep424.json`, `results/test_results/PureSwap_E1cosine-P10_1000ep_p010_test_results.json` |
| P15 | `configs/PureSwap_E1cosine-P15_1000ep.yaml` | `scripts/train_PureSwap_E1cosine-P15_1000ep.py` | `results/checkpoint_evals/P15_test_ep424.json`, `results/checkpoint_evals/P15_test_ep274.json`, `results/test_results/PureSwap_E1cosine-P15_1000ep_p015_test_results.json` |
| **P25** | `configs/PureSwap_E1cosine-P25_1000ep.yaml` | `scripts/train_PureSwap_E1cosine-P25_1000ep.py` | `results/test_results/PureSwap_E1cosine-P25_1000ep_p025_test_results.json` |
| P50 | `configs/PureSwap_E1cosine-P50_1000ep.yaml` | `scripts/train_PureSwap_E1cosine-P50_1000ep.py` | `results/checkpoint_evals/P50_test_ep424.json`, `results/test_results/PureSwap_E1cosine-P50_1000ep_p050_test_results.json` |
| E1B1 | `configs/PureSwap_E1cosine-B1_1000ep.yaml` | `scripts/train_PureSwap_E1cosine-B1_1000ep.py` | `results/test_results/PureSwap_E1cosine-B1_1000ep_p025_test_results.json` (data ablation) |
| E2 | `configs/PureNoise_E2_300ep_s010.yaml` | `scripts/train_PureNoise_E2_300ep.py` | `results/checkpoint_evals/E2_test_ep274.json`, `results/test_results/PureNoise_E2_300ep_s010_test_results.json` |
| E3 | `configs/PureDrop_E3_300ep_p025.yaml` | `scripts/train_PureDrop_E3_300ep.py` | `results/checkpoint_evals/E3_test_ep274.json`, `results/test_results/PureDrop_E3_300ep_p025_test_results.json` |
| E4 | `configs/PureSwap_E4_1000ep.yaml` | `scripts/train_PureSwap_E4_1000ep.py` | `results/checkpoint_evals/E4_test_ep424.json`, `results/test_results/PureSwap_E4_1000ep_pinv_test_results.json` |

> `E1cosine-B` (12-work oversampled p=0.25) is excluded from the matrix
> and from evaluation — no config/script/results for it are shipped.
> `PureSwap_E1cosine-P25` is the original `PureSwap_E1cosine_1000ep` run,
> renamed for dose-naming consistency.

The universal evaluation tool `scripts/evaluate_checkpoint.py` can
evaluate any checkpoint on the test manifest; see
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for usage.

---

## 6. Known Gaps and Planned Extensions

Documented in `AUDIT_REPORT.md` § 5; summarized here:

1. **TruePure baseline** (same protocol, `p_swap=0`) — needed for a clean
   shortcut-existence attribution (Argument 1).
2. **E2/E3 protocol realignment** — re-run with warmup_cosine + 1000 ep
   to remove the LR/horizon confound in Argument 2.
3. **Multi-seed repeats** (seed=2027, 2028) — needed for significance
   testing of 0.1–0.3 dB differences (P15/P25/P50 within 0.13 dB).
4. **Independent E4 control** — gate-magnitude L2 reg (`λ‖W‖²`) without
   permutation, plus a λ dose sweep, to test whether a
   permutation-free path works.
5. **Overlap audit** — verify item-level overlap between E1B1's 12 source
   movements and the 59-work train set (Argument 5 attribution).
