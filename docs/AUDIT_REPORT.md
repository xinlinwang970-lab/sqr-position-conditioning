# Rigor Audit Report: Arguments 1–4

**Audit date**: 2026-11-18
**Audit target**: `docs/EXPERIMENT_REPORT.md`, Arguments 1–4
**Model**: 30.44M `mid_dual_paper` variant (paper reports 46M, rounded —
the architecture details are incomplete, so absolute SDR is not directly comparable)
**Audit method**: Per-experiment verification of script parameters, config files, checkpoint training steps, and evaluation epochs

---

## 0. Errata (Hard Errors in the Original Report)

| Location | Original report | Actual value | Severity |
|---|---|---|---|
| Arg. 2 table | E1cosine-B "const LR=5e-4" | **warmup_cosine** (500 upd warmup → cosine 5e-4→5e-6) | 🔴 Critical |
| Arg. 2 table | E1cosine-B "59 works, pure type2" | **12 works with crop-slot oversampling** — data scale differs from E2/E3 | 🔴 Critical (added third confound) |
| Arg. 4 | E4 "const LR" | **warmup_cosine** (same as P15) | 🔴 Critical |
| Arg. 4 | E4 "λ=1e-3" | **λ=1.0** | 🔴 Critical |
| Arg. 4 formula | $\mathcal{L}_{\text{E4}} = \ell + \lambda\|\mathbf{W}_{\text{gate}}\|_2^2$ (L2 magnitude reg) | $\mathcal{L}_{\text{E4}} = \ell + \lambda\|\mathbf{W}(p) - \mathbf{W}(\sigma(p))\|^2$ (**position-invariance reg**) | 🔴 Critical (mechanism description entirely wrong) |

> **Resolution**: `E1cosine-B` (12 works, oversampled) is **excluded from
> evaluation** — its data scale/generator is incomparable to the 59-work
> protocol. `P25` is the **original `PureSwap_E1cosine_1000ep`** run
> (59 works, p=0.25, warmup_cosine, 1000ep) — it completes the 4-point
> dose curve. `E1cosine-B1` (60 segments, 12 works, no oversampling) is
> included as a **data-ablation** experiment (see § 1 and Argument 5).
> The swap arm at ep274 is represented by `P15` (p=0.15, 59 works).

---

## 1. Experiment Protocol Verification Table

| Experiment | Architecture | Training data | LR | Mechanism params | Training length | test eval point | test mean | type1 | type2 |
|---|---|---|---|---|---|---|---|---|---|
| author_release | **split_stage_paper** | 12 works (/mnt/sda) | warmup_cosine | none | ~52k upd | final | +1.33 | −0.33 | +1.64 |
| P15 (swap) | mid_dual_paper | 59 works, pure type2 | warmup_cosine | p_swap=0.15 | 1000ep (done) | **ep274** (mid) | +2.26 | +1.56 | +1.69 |
| E2 PureNoise | mid_dual_paper | 59 works, pure type2 | **const 5e-4** | σ=0.1 | 300ep (done) | final (ep300) | +0.79 | +0.43 | +1.24 |
| E3 PureDrop | mid_dual_paper | 59 works, pure type2 | **const 5e-4** | p_drop=0.25 | 300ep (done) | final (ep300) | +1.01 | +0.34 | +1.69 |
| E4 | mid_dual_paper | 59 works, pure type2 | **warmup_cosine** | λ=1.0, position invariance | 1000ep (done) | final (ep1000) | +2.24 | +1.81 | +2.13 |
| P10 | mid_dual_paper | 59 works, pure type2 | warmup_cosine | p_swap=0.10 | 1000ep (done) | ep424 | +2.21 | +1.75 | +2.12 |
| P15 | mid_dual_paper | 59 works, pure type2 | warmup_cosine | p_swap=0.15 | 1000ep (done) | ep424 | +2.51 | +2.27 | +2.30 |
| P25 | mid_dual_paper | 59 works, pure type2 | warmup_cosine | p_swap=0.25 | 1000ep (done) | ep1000 | **+2.35** | +1.97 | +1.93 |
| P50 | mid_dual_paper | 59 works, pure type2 | warmup_cosine | p_swap=0.50 | 1000ep (done) | **ep449** | +2.32 | +2.25 | +2.09 |
| E1cosine-B1 | mid_dual_paper | **60 segments (12 works × 5, no oversampling)** | warmup_cosine | p_swap=0.25 | 1000ep (done) | final (ep1000) | +1.61 | +1.25 | +1.43 |

> **Designation**: `P10/P15/P25/P50` are the **4 dose ablations of the E1
> (position-swap) mechanism** on the 59-work protocol. `E1cosine-B1` is a
> **data-ablation** run on a segmented dataset (60 segments from 12 works,
> no oversampling) — its absolute SDR is diluted by low diversity and is
> not comparable to the 59-work family.
>
> **Excluded**: `E1cosine-B` (the earlier 12-record oversampled variant)
> is **excluded** from evaluation. It uses 12 full movements with crop-slot
> oversampling (~4.92× per epoch), a data scale and generator different
> from both the 59-work protocol and the 60-segment E1B1 — its numbers are
> not comparable to either family and are not cited in this report.

---

## 2. Per-Argument Audit

### Argument 1: Shortcut existence

**Original claim**: author_release baseline type1 = −0.33 → shortcut causes combinatorial generalization failure.

**Control variable audit**:

| Control variable | author_release | P-series (reference) | Aligned? |
|---|---|---|---|
| Architecture (condition_route) | **split_stage_paper** | mid_dual_paper | ❌ Different |
| Training data | 12 works, /mnt/sda path | 59 works, /ceph path | ❌ Different |
| LR schedule | warmup_cosine | warmup_cosine | ✓ |
| Anti-shortcut intervention | none | yes | ✓ (intended to differ) |

**Logical flaws**:

1. **Triple confound**: author_release and the P-series differ in architecture (split_stage_paper vs mid_dual_paper), data scale (12 vs 59 works), and data source (/mnt/sda vs /ceph, possibly different preprocessing). The type1 = −0.33 could be caused by any of architectural differences, insufficient data, or the shortcut — **cannot be attributed to the shortcut alone**.

2. **Missing key control**: No "TruePure" baseline exists — i.e., same architecture (mid_dual_paper), same data (59 works, pure type2), same LR (warmup_cosine), same training length (1000ep), but p_swap=0. Without this baseline, we cannot determine whether type1 would also be negative under no intervention.

3. **Direction of architectural effect unknown**: `split_stage_paper` routes e_t to the encoder, e_m=0.5(e_t+e_p) to the middle, and e_p to the decoder; `mid_dual_paper` routes e_m to the decoder only. The former conditions three stages, but its decoder still receives audio and skip activations. The route alone does not establish shortcut behavior or its strength.

**Rigor**: ⚠️ **Insufficient**. type1 = −0.33 is suggestive evidence, but confounds are too many to serve as a clean argument for "shortcut existence."

**Fix**: Run a TruePure baseline (mid_dual_paper, 59 works, p_swap=0, warmup_cosine, 1000ep). If its type1 is also negative or near zero, Argument 1 holds; if type1 ≥ +1.0, the author_release failure is mainly due to architecture/data, and Argument 1 must be re-argued.

---

### Argument 2: Input-side > embedding-side

**Original claim**: Swap > Drop > Noise; input-side intervention outperforms embedding-side.

**Control variable audit** (using P15 @ep274 as the swap reference — same 59-work data as E2/E3, replacing the previously mis-reported E1cosine-B which used only 12 oversampled records):

| Control variable | P15 (Swap) | E2 (Noise) | E3 (Drop) | Aligned? |
|---|---|---|---|---|
| Architecture | mid_dual_paper | mid_dual_paper | mid_dual_paper | ✓ |
| Training data | 59 works, pure type2 | 59 works, pure type2 | 59 works, pure type2 | ✓ |
| **LR schedule** | **warmup_cosine** | **const 5e-4** | **const 5e-4** | ❌ **Different** |
| **Training length** | 1000ep | 300ep | 300ep | ❌ **Different** |
| **test eval point** | ep274 (mid) | ep274 (mid) | ep274 (mid) | ✓ same epoch |
| Mechanism strength | p=0.15 | σ=0.1 ("lightest") | p=0.25 | ⚠️ Not directly comparable |
| Granularity | batch-level (whole batch permuted) | sample-level | sample-level | ⚠️ Structural difference |

**Logical flaws**:

1. **LR confound (most severe)**: P15 uses warmup_cosine; E2/E3 use const LR. At ep274, P15's LR has decayed from 5e-4 to ~4.3e-4, while E2/E3 remain at 5e-4. More critically, warmup_cosine continues decaying after ep300, bringing stability, while const LR's lack of decay may cause late-stage instability — this systematically disadvantages E2/E3.

2. **Training-horizon confound**: E2/E3 are designed as 300ep screening runs (short horizon); P15 is 1000ep (long horizon). Even comparing at the same ep274, the model's "expected lifetime" differs, which may affect training dynamics (cosine decay schedule depends on total horizon).

3. **Strength uncalibrated**: The original design doc explicitly labels σ=0.1 as the "lightest intervention." swap p=0.15 means 15% of steps have completely wrong positions; noise σ=0.1 adds mild Gaussian noise to e_p (dim=32). The two are not equivalent in "degree of position information destruction." No strength matching was performed (e.g., matching val degradation magnitude).

4. **Same-epoch comparison used but LR still confounded**: All three are evaluated at ep274 (E2: +1.03/type1 +0.03; E3: +1.41/type1 +0.68; P15: +2.26/type1 +1.56). The LR schedule and horizon confounds persist.

5. **Granularity confound**: swap is batch-level (the whole batch of 4 samples shares one permutation); noise/drop are sample-level (each sample independent). Batch-level swap produces a "sharper" gradient signal (all positions in the batch are corrupted simultaneously); sample-level noise/drop is "smoother" (only some dimensions are perturbed). This is a structural difference, not just "input vs embedding."

6. **Data-scale confound (resolved vs the previous report)**: an earlier draft used E1cosine-B (12 oversampled records) as the swap reference; that introduced a data-scale confound (12 vs 59). The 59-work protocol's p=0.25 point is now represented by **P25** (=the original `PureSwap_E1cosine_1000ep` run, 59 works). The B1 run (60 segments, no oversampling) is treated separately as a data-ablation experiment (see Argument 5).

**Rigor**: ⚠️ **Severe confounds remain** (LR + horizon + strength + granularity). The type1 collapse (+0.03/+0.68 vs +1.56) has a **qualitative trend that may hold** (the ~0.9–1.5 dB gap is hard to explain by LR alone), but **quantitative comparison and causal attribution of "input-side outperforms embedding-side" do not hold**.

**Fix**:
- **Minimal**: Use the ep274-aligned numbers (done) and explicitly flag the LR/horizon/strength confounds in the text.
- **Complete**: Re-run E2/E3 with warmup_cosine + 1000ep (or at least E3, since its drop p=0.25 is more comparable in strength), and evaluate at the same epoch.

---

### Argument 3: E1 dose sweep — best dose at p=0.15

**Original claim**: P10 < P15 > P50; p=0.15 is the sweet spot (P25 included on the dose axis with a data caveat).

**Control variable audit**:

| Control variable | P10 | P15 | P25 | P50 | Aligned? |
|---|---|---|---|---|---|
| Architecture | mid_dual_paper | mid_dual_paper | mid_dual_paper | mid_dual_paper | ✓ |
| Training data | 59 works | 59 works | 59 works | 59 works | ✓ |
| LR | warmup_cosine | warmup_cosine | warmup_cosine | warmup_cosine | ✓ |
| Mechanism | p_swap=0.10 | p_swap=0.15 | p_swap=0.25 | p_swap=0.50 | ✓ (the variable) |
| **test eval point (final)** | **ep1000** | **ep1000** | **ep1000** | **ep1000** | ✓ aligned (all complete) |
| **test eval point (mid)** | **ep424** | **ep424** | **ep250** * | **ep424** | ⚠️ P25 at ep250, others at ep424 |
| seed | 2026 | 2026 | 2026 | 2026 | ⚠️ Single seed |

> \* P25 (the original `PureSwap_E1cosine_1000ep` run) has a mid-training
> evaluation at ep250 (update 14750); the other three were evaluated at
> ep424. Its mid-training trajectory is therefore offset by ~174 epochs
> relative to the concurrent P-series checkpoints. See logical flaw (3).

**Logical flaws**:

1. **Dose curve complete (resolved)**: P25 (= the original E1cosine run,
   `PureSwap_E1cosine_1000ep`) uses the **same 59-work protocol** as
   P10/P15/P50 (manifest: `sqr_paper_split_train.jsonl`). The four-point
   dose curve {0.10, 0.15, 0.25, 0.50} is now complete on the 59-work
   data. All experiments share architecture, LR schedule, seed, and
   training horizon — this is the best-controlled comparison in the study.

2. **Single seed, no significance**: P25 +2.35 vs P15 +2.25 vs P50 +2.22.
   The top three are within 0.13 dB. No repeats; cannot determine whether
   these differences exceed random fluctuation. SDR on a small test set
   (n=48) typically has a median standard deviation of ~0.3–0.5 dB for
   string separation. The differences between P15/P25/P50 are **well
   within noise** — only P10 (+1.99) is clearly separated from the group.

3. **Mid-training eval offset**: P25's mid-training eval is at ep250
   (mean +2.23, t1 +1.71), not ep424 like P10/P15/P50. This means the
   mid-training dose curve is not uniformly aligned. The final ep1000
   comparison is aligned.

4. **"P50 instability" argument is internally contradictory**: The
   original report claims P50 "over-doses and introduces training
   instability," citing val oscillation. But test data shows P50 is
   extremely stable (+2.31/+2.28/+2.32 at ep349/399/449). The report
   principle "val ≠ test, use test as the criterion" — using val
   oscillation to argue P50 instability contradicts this principle.

**Rigor**: ⚠️ **Moderate**. Control variables are fully aligned (all four
dose points on same architecture/data/LR/seed/horizon). Main limitations:
single seed (P15/P25/P50 within 0.13 dB, likely in noise), and the
mid-training eval offset for P25.

**Fix**:
- **Immediate (done)**: P50 @ep424 evaluated; P25 identified as the
  original E1cosine run (59-work protocol), completing the dose curve.
- **Ideal**: Run each p_swap with 3 seeds and report mean ± std.

---

### Argument 4: E4 alternative path

**Original claim**: E4 (GPoCM gate L2 regularization) achieves test +2.24, comparable to swap, as an alternative path independent of input perturbation.

**Control variable audit**:

| Control variable | E4 | P15 (reference) | Aligned? |
|---|---|---|---|
| Architecture | mid_dual_paper | mid_dual_paper | ✓ |
| Training data | 59 works, pure type2 | 59 works, pure type2 | ✓ |
| LR | warmup_cosine | warmup_cosine | ✓ |
| **Mechanism** | λ=1.0 position-invariance reg | p_swap=0.15 | ✓ (the variable) |
| **Training length** | **1000ep (done)** | **1000ep (done)** | ✓ |
| **test eval point** | ep424 + final (ep1000) | ep424 + final (ep1000) | ✓ aligned (both complete, both eval points) |

**Logical flaws**:

1. **Mechanism description entirely wrong (listed in errata)**: E4 is not $\lambda\|\mathbf{W}\|_2^2$ (L2 magnitude reg) but $\lambda\|\mathbf{W}(p) - \mathbf{W}(\sigma(p))\|^2$ (**position-invariance reg**). E4 internally **also uses position swap** $\sigma(p)$ to compute the regularization term — only the swap acts in the gate-parameter space, not the model input. Therefore the claim of "an alternative path independent of input perturbation" is **inaccurate**: E4 and swap share the same anti-shortcut signal (position permutation), just acting at different points.

2. **Evaluation point misalignment (resolved)**: The original audit flagged E4 @final (59000 upd) vs P15 @ep424 (25075 upd). This is resolved — E4 @ep424 was evaluated (+2.32/type1 +2.09/type2 +2.19), and P15 completed to ep1000 (+2.25/type1 +2.17/type2 +2.20). **Findings**: at ep424, E4 is within 0.19 dB of P15 (both high); at ep1000, both converge to ≈+2.25 test mean, but E4's type1 drops to +1.81 (Δ_cg widens to +0.32) while P15 holds type1 at +2.17 (Δ_cg +0.03). E4's generalization recovery is **less stable** over long training.

3. **"Two independent paths" overclaim**: E4 and swap are both based on the "position permutation" signal. A truly independent path would be a mechanism that does not involve position permutation at all (e.g., gate-magnitude L2 reg, or dropout on the controller MLP), but E4 is not such a path.

4. **No λ dose sweep**: Only λ=1.0 is tested. The effect of regularization strength on E4 is unknown.

**Rigor**: ⚠️ **Insufficient for a causal claim**. Mechanism description corrected, evaluation points aligned, but the "independent path" framing does not hold and λ has no sweep. E4's ep424/ep1000 numbers are valuable data points, but the comparison with P15 is not a fully clean controlled experiment.

**Fix**:
- **Immediate (done)**: E4 @ep424 evaluated and P15 @ep1000 evaluated — both evaluation points now aligned.
- **Correct the description (done)**: E4 described as "position-invariance regularization," sharing the position-permutation signal with swap.
- **Ideal**: Implement a true gate-magnitude L2 regularization ($\lambda\|\mathbf{W}\|_2^2$) as a control to verify whether a "permutation-free" path works; sweep λ ∈ {0.1, 1.0, 10}.

---

### Argument 5 (Data ablation): segmented-material swap (E1cosine-B1)

**Original claim**: E1cosine-B1 (60 segments from 12 works, p_swap=0.25)
underperforms the 59-work P25 (same p_swap) by 0.74 dB, confirming that
work-level diversity matters more than segment-level diversity.

**Control variable audit**:

| Control variable | E1cosine-B1 | P25 (reference) | Aligned? |
|---|---|---|---|
| Architecture | mid_dual_paper | mid_dual_paper | ✓ |
| **Training data** | **60 segments (12 works × 5, no oversampling)** | **59 works** | ❌ **Different (the variable)** |
| LR | warmup_cosine | warmup_cosine | ✓ |
| Mechanism | swap p_swap=0.25 | swap p_swap=0.25 | ✓ |
| Training length | 1000ep | 1000ep | ✓ |
| Eval point | final (ep1000) | final (ep1000) | ✓ |
| seed | 2026 | 2026 | ✓ (single seed shared) |

**Logical flaws**:

1. **Data confound**: E1cosine-B1 (+1.61) vs P25 (+2.35) = 0.74 dB gap
   with the same swap dose. This strongly suggests work diversity is the
   key factor, but possible work overlap between the 12-source-movement
   set and the 59-work train set has not been verified at the item level.

2. **Single seed**: the gap may be inflated or deflated by seed 2026
   specificities. Multi-seed repeats are needed for a robust claim.

3. **Oversampling confound (E1B excluded)**: The earlier E1cosine-B (12
   works, crop-slot oversampling) is excluded from evaluation because its
   ~4.92× per-epoch cropping makes it a different training generator than
   both the 59-work (no oversampling) and E1B1 (60 segments, no
   oversampling) protocols.

**Rigor**: ⚠️ **Moderate**. Provides a strong qualitative finding
(work diversity > segment diversity, 0.74 dB gap) but single seed and
possible overlap prevent a clean quantitative attribution.

**Fix**:
- **Verify**: Audit item-level overlap between E1B1's 12-source movements
  and the 59-work train set.
- **Ideal**: Multi-seed repeats of both P25 and E1B1.

---

## 3. Cross-Argument Systemic Issues

| Issue | Scope | Severity |
|---|---|---|
| **Single seed (seed=2026)** | All arguments | 🔴 No significance testing; P15/P25/P50 within 0.13 dB — likely in noise |
| **Missing TruePure baseline** | Argument 1 | 🔴 Cannot cleanly attribute to shortcut |
| **E1B1 data confound** (60 segments vs 59 works) | Argument 5 | 🟡 0.74 dB gap suggestive but not cleanly attributable (single seed + possible overlap) |
| **E2/E3 protocol inconsistent with swap family** (const LR + 300ep) | Argument 2 | 🔴 Double confound (LR + horizon) |
| **P25 mid-training eval offset** (ep250 vs ep424) | Argument 3 | 🟡 Final ep1000 aligned; mid-training only approximately comparable |
| **E1B excluded** | All | Data-scale/generator confound removed by exclusion (see § 1) |

---

## 4. Overall Rigor Assessment

| Argument | Control variables | Logical completeness | Evidence strength | Overall |
|---|---|---|---|---|
| 1. Shortcut exists | ❌ Triple confound (arch + data + source) | Missing key control | Suggestive but insufficient | **Needs TruePure baseline** |
| 2. Input-side > embedding-side | ❌ LR + horizon + strength confound | Qualitative trend may hold, quantitative does not | Moderate-weak | **Needs unified protocol re-run of E2/E3** |
| 3. E1 dose sweep (P10/P15/P25/P50) | ⚠️ All 4 on same protocol (59 works), same arch/LR/seed | Trend clear (U-shape), but top 3 within 0.13 dB noise | **Most solid** | **Needs multi-seed for significance** |
| 4. E4 vs E1 | ⚠️ Evaluation aligned, mechanism corrected, but shared signal + single λ | "Independent path" does not hold | Moderate | **Needs permutation-free control + λ sweep** |
| 5. E1B1 data ablation | ⚠️ Data confound + single seed + possible overlap | 0.74 dB gap suggestive | Moderate | **Needs overlap audit + multi-seed** |

---

## 5. Immediately Executable Fixes (by priority)

| Priority | Action | Fixes argument | GPU cost | Time |
|---|---|---|---|---|
| **Done** | Evaluate P50 @ep424 (checkpoint_025075.pt) — epoch aligned | Arg. 3 | 1 GPU | ~30 min |
| **Done** | Evaluate E4 @ep424 — epoch aligned with P15/P10/P50 | Arg. 4 | 1 GPU | ~30 min |
| **Done** | Identify P25 as original E1cosine (59-work protocol) — completes 4-point dose curve | Arg. 3 | — | — |
| **P1** | Launch TruePure baseline (mid_dual_paper, 59 works, p_swap=0, warmup_cosine, 1000ep) | Arg. 1 key control | 1 GPU | ~37 h |
| **P1** | Re-run E3 PureDrop (p=0.25) with warmup_cosine + 1000ep | Arg. 2 LR alignment | 1 GPU | ~37 h |
| **P2** | Audit item-level overlap between E1B1's 12-source movements and the 59-work train set | Arg. 5 | 0 GPU | ~10 min |
| **P3** | Multi-seed repeats (seed=2027, 2028) for P15/P25 | Arg. 3, 4, 5 significance | 2 GPUs | ~37 h |
