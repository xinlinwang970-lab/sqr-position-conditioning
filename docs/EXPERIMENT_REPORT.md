# Position-Shortcut Learning in String Quartet Source Separation: Combinatorial Generalization and Forward Transfer

**Project**: sqr-position-repro (independent reimplementation + ablation)
**Model**: 30.44M-parameter `mid_dual_paper` variant (paper reports 46M, rounded)
**Test set**: URMP StringQuartet, 6 works × 2 seatings × 4 stems = 48 WaveSDR samples
**Metric**: full-length `legacy_v1` 50%-overlap sliding-window WaveSDR (dB)
**Reproduction status**: this is an independent ablation model, not an exact
reproduction. The paper reports 46M, but does not provide enough layer-width
and controller details to reconstruct the reported model uniquely. Absolute
SDR numbers are not directly comparable; findings below compare runs within
the stated model and protocol. See [`ARCHITECTURE.md`](ARCHITECTURE.md) and
[`GROWTH24_46M_CANDIDATES.md`](GROWTH24_46M_CANDIDATES.md).

---

## 1. Motivation

A position-conditioned string quartet source separation model accepts a discrete seating index $p \in \{0,1,2,3\}$ as an explicit hint to extract the instrument at a given physical seat from the mixture $x$. The internal conditioning path of the model is:

$$
e_t = E_t(t), \quad e_p = E_p(p), \quad e_m = \tfrac{1}{2}(e_t + e_p)
$$

$e_m$ is passed through a controller MLP to produce GPoCM gate parameters
$\mathbf{W}(e_m), \mathbf{b}(e_m)$, which are injected into the decoder
(`mid_dual_paper` route). GPoCM applies these gates to audio activations;
it does not replace the audio path.

When the seat→part mapping $L$ is deterministic in the training data, the
model may rely too heavily on condition labels and underuse mixture audio
$x$. We call this the **position-shortcut hypothesis**. In this
`mid_dual_paper` model, however, the gate is generated from the mixed
timbre–position embedding $e_m$ and applied to audio features. The route
alone does not imply a position-only lookup or prove that $x$ is ignored.
The experiments evaluate output performance across:

1. **Combinatorial generalization**: At test time, an unseen seating layout $L'$ appears.
2. **Forward transfer**: Training uses synthetic MuseScore/StringQuD data and evaluation uses real URMP recordings.

These evaluations can be consistent with the shortcut hypothesis, but do not
directly measure whether the model uses audio. The experiments compare how
interventions change layout and domain metrics.

---

## 2. Problem Definition

### 2.1 Data and Layouts

Training distribution $\mathcal{D}_{\text{train}} = \{(x, \mathbf{y}, L_{\text{type2}})\}$ contains only the type2 layout (59 works);
Test distribution $\mathcal{D}_{\text{test}} = \{(x, \mathbf{y}, L) : L \in \{L_{\text{type1}}, L_{\text{type2}}\}\}$ contains both type1 (**unseen**) and type2.

$$
L_{\text{type1}} = \{\text{v2}{\to}0,\ \text{va}{\to}1,\ \text{c}{\to}2,\ \text{v1}{\to}3\}, \quad
L_{\text{type2}} = \{\text{c}{\to}0,\ \text{va}{\to}1,\ \text{v2}{\to}2,\ \text{v1}{\to}3\}
$$

### 2.2 Evaluation Metrics

- **Combinatorial generalization**: Median SDR on unseen layouts $\text{SDR}_{\text{type1}}$, and the generalization gap $\Delta_{\text{cg}} = \text{SDR}_{\text{type2}} - \text{SDR}_{\text{type1}}$ (smaller is better).
- **Forward transfer**: Overall mean WaveSDR on the URMP test set $\text{SDR}_{\text{test}}$ (synthetic→real).

### 2.3 Anti-Shortcut Intervention Mechanisms

| Mechanism | Formula | Acts on | Protocol |
|---|---|---|---|
| **Position Swap** | $p \leftarrow \sigma(p)$ w.p. $p_{\text{swap}}$, $\sigma \in S_4$ | input $p$ | warmup_cosine, 1000ep |
| **Embedding Noise** | $e_p \leftarrow e_p + \sigma_{\text{noise}}\mathcal{N}(0,I)$ | embedding $e_p$ | const LR, 300ep |
| **Embedding Drop** | $e_p \leftarrow 0$ w.p. $p_{\text{drop}}$ | embedding $e_p$ | const LR, 300ep |
| **Gate Invariance Reg** | $\mathcal{L} = \ell + \lambda\|\mathbf{W}(p)-\mathbf{W}(\sigma(p))\|^2$ | gate parameters $\mathbf{W}$ | warmup_cosine, 1000ep |

Note: Swap and Gate Invariance Reg share the position permutation signal $\sigma(p)$, but act at different points — the former perturbs the model input, the latter constrains the sensitivity of gate parameters to position.

---

## 3. Arguments and Evidence

### Argument 1: The author-release run has a large unseen-layout gap

**Evidence**: Author-release baseline (author_release_12ep_warmup_cosine) results on the URMP test set:

| Layout | $\text{SDR}_{\text{type1}}$ | $\text{SDR}_{\text{type2}}$ | $\Delta_{\text{cg}}$ |
|---|---|---|---|
| Unseen (type1) | **−0.33** | +1.64 | **+1.97** |

The reported type1 median WaveSDR is below 0 dB. This score alone does not
identify why the model generalizes poorly or compare it with a zero-output
baseline.

**Control variables and limitations**: This baseline uses `split_stage_paper`: encoder receives $e_t$, middle receives $e_m$, and decoder receives $e_p$. It also uses a 12-work training set, unlike subsequent `mid_dual_paper` experiments with 59 works. The negative type1 value is **consistent with** a shortcut hypothesis but does not establish one; architecture and data scale differ, and a same-architecture, same-data, no-intervention baseline (TruePure) has not been run. The decoder continues to receive audio and skip features.

**Safe claim**: Under the author-release configuration, the position-conditioned model exhibits a severe generalization failure on unseen layouts ($\Delta_{\text{cg}} = +1.97$ dB), consistent with the position shortcut hypothesis.

---

### Argument 2: Intervention runs differ on unseen-layout metrics; comparisons have remaining confounds

**Evidence**: Under the same training data (59 works, pure type2) and same architecture (mid_dual_paper), comparing three intervention mechanisms at the ep274 checkpoint:

| Mechanism | Acts on | LR | test mean | $\text{SDR}_{\text{type1}}$ | $\text{SDR}_{\text{type2}}$ |
|---|---|---|---|---|---|
| Swap ($p{=}0.15$, P15) | input $p$ | warmup_cosine | +2.26 | **+1.56** | +1.69 |
| Drop ($p{=}0.25$, E3) | embedding $e_p$ | const | +1.41 | +0.68 | +1.63 |
| Noise ($\sigma{=}0.1$, E2) | embedding $e_p$ | const | +1.03 | +0.03 | +1.48 |

**Analysis**: The observed type1 scores differ across these runs. Position swap may reduce the reliability of the training association between the supplied position and target, while Noise/Drop alter the embedding. These results do not by themselves identify which internal signal the model uses or show that it verifies the hint against audio.

**Control variable limitations**: All three experiments now share the same 59-work training data (removing a previous data-scale confound). Remaining confounds: (a) Swap uses warmup_cosine LR while Noise/Drop use const LR; (b) Noise/Drop train for only 300ep (vs. 1000ep for Swap), so the models have different "expected lifetimes" at ep274; (c) mechanism strengths are not calibrated — swap $p{=}0.15$ is not directly comparable to drop $p{=}0.25$ or noise $\sigma{=}0.1$ in terms of position-information destruction. However, the type1 differences (+1.56 vs +0.03/+0.68, a 0.9~1.5 dB gap) are large enough that LR schedule differences alone are unlikely to fully explain them — **the qualitative trend (input-side intervention outperforms embedding-side) likely holds, but quantitative comparison remains confounded**.

---

### Argument 3: E1 position-swap dose sweep — P15 performs best on type1 among the final checkpoints

**Evidence**: The E1 mechanism (position swap) is evaluated as a 4-dose sweep
$\{0.10, 0.15, 0.25, 0.50\}$ under fixed architecture (mid_dual_paper), LR
(warmup_cosine), seed (2026), and horizon (1000ep, all complete).

**Final results (ep1000, complete training):**

| ID | $p_{\text{swap}}$ | $\text{SDR}_{\text{test}}$ | $\text{SDR}_{\text{type1}}$ | $\text{SDR}_{\text{type2}}$ | $\Delta_{\text{cg}}$ |
|---|---|---|---|---|---|
| P10 | 0.10 | +1.99 | +1.51 | +2.06 | +0.55 |
| **P15** | **0.15** | **+2.25** | **+2.17** | +2.20 | **+0.03** |
| **P25** | **0.25** | **+2.35** | **+1.97** | +1.93 | **−0.05** |
| P50 | 0.50 | +2.22 | +2.15 | +2.17 | +0.02 |

**Mid-training snapshot (ep424, update 25075) for cross-reference (P25 at ep250):**

| $p_{\text{swap}}$ | $\text{SDR}_{\text{test}}$ | $\text{SDR}_{\text{type1}}$ | $\text{SDR}_{\text{type2}}$ | $\Delta_{\text{cg}}$ |
|---|---|---|---|---|
| 0.10 (P10) | +2.21 | +1.75 | +2.12 | +0.37 |
| **0.15 (P15)** | **+2.51** | **+2.27** | +2.30 | **+0.03** |
| 0.25 (P25) @ep250 | +2.23 | +1.71 | +1.71 | +0.00 |
| 0.50 (P50) | +2.26 | +2.00 | +1.91 | −0.09 |

**Analysis**:

1. **Forward transfer (final)**: P25 achieves the highest $\text{SDR}_{\text{test}}=+2.35$, closely followed by P15 (+2.25) and P50 (+2.22). The dose-response is U-shaped: $p{=}0.10$ underperforms, $p{=}0.15$–$0.50$ cluster near each other with P25 slightly ahead.
2. **Combinatorial generalization (final)**: P15 ($+2.17$) and P50 ($+2.15$) achieve the highest $\text{SDR}_{\text{type1}}$, while P25 is at $+1.97$. All three bring $\Delta_{\text{cg}}$ close to zero ($-0.05$ to $+0.03$). P10's larger $\Delta_{\text{cg}}=+0.55$ is an observed difference; it does not establish a threshold for shortcut removal.
3. **Dose effect**: In these runs, $p{=}0.10$ has a larger $\Delta_{\text{cg}}$ than P15/P25/P50, whose gaps are near zero. This is an observed dose pattern, not a direct measure of shortcut suppression. P15 has the best type1 generalization, P25 the best overall SDR, and P50 the best type2 among the reported checkpoints.

**Control variables**: All four dose points share the same architecture, data (59-work protocol), LR (warmup_cosine), seed (2026), and evaluation point (final ep1000 for all; ep424/ep250 for mid-training). Main limitation: single seed — 0.1–0.2 dB differences cannot be tested for significance.

---

### Argument 4: Gate-parameter invariance regularization achieves comparable performance to Swap at equal training steps

**Evidence**: E4 adds an invariance penalty between gate parameters generated under the original and permuted position conditions:

$$
\mathcal{L}_{\text{E4}} = \ell(f_\theta(x,t,p), y) + \lambda \cdot \frac{1}{n}\sum_{i=1}^{n}\Big[\|\mathbf{W}_i(p) - \mathbf{W}_i(\sigma(p))\|^2 + \|\mathbf{b}_i(p) - \mathbf{b}_i(\sigma(p))\|^2\Big]
$$

It shares the position permutation signal $\sigma(p)$ with Swap, but Swap injects $\sigma(p)$ into the model input so the loss is computed on mismatched positions, while E4 uses $\sigma(p)$ to compute a **position-invariance constraint** on the gate parameters — the model input still uses the correct $p$.

**Final results (ep1000) and mid-training snapshot (ep424):**

| Mechanism | Eval point | $\text{SDR}_{\text{test}}$ | $\text{SDR}_{\text{type1}}$ | $\text{SDR}_{\text{type2}}$ | $\Delta_{\text{cg}}$ |
|---|---|---|---|---|---|
| Swap ($p{=}0.15$, P15) | ep424 | **+2.51** | **+2.27** | +2.30 | +0.03 |
| Swap ($p{=}0.15$, P15) | ep1000 | +2.25 | +2.17 | +2.20 | +0.03 |
| Gate Invariance Reg ($\lambda{=}1.0$, E4) | ep424 | +2.32 | +2.09 | +2.19 | +0.10 |
| Gate Invariance Reg ($\lambda{=}1.0$, E4) | ep1000 | +2.24 | +1.81 | +2.13 | +0.32 |

**Analysis**: At ep424, E4 achieves $\text{SDR}_{\text{test}}=+2.32$ and $\Delta_{\text{cg}}=+0.10$, within 0.19 dB of P15. At ep1000, both P15 and E4 converge to $\text{SDR}_{\text{test}}\approx+2.25$, though E4's $\Delta_{\text{cg}}$ widens to $+0.32$ (type1 drops to +1.81 while type2 holds at +2.13), suggesting E4's generalization recovery is less stable than swap's over long training.

**Control variables**: E4 and P15 are fully aligned on architecture, data, LR, and evaluation epoch — only the mechanism differs. Limitations: (a) both share the $\sigma(p)$ signal and are not fully independent anti-shortcut paths; (b) E4 tests only a single strength $\lambda=1.0$ with no dose sweep; (c) single seed.

**Safe claim**: E4 and Swap achieve similar forward-transfer SDR at the listed checkpoints, while E4's unseen-layout gap is wider at ep1000. This supports a performance comparison between the two interventions, but does not establish that a shortcut lives in the gates or that either intervention removed one.

---

## 4. Data Ablation: Segmented Dataset (E1cosine-B1)

`E1cosine-B1` (`PureSwap_E1cosine-B1_1000ep`) is a p_swap=0.25 position-swap
run on a **segmented dataset**: 12 StringQuD movements split into 5
equal-length segments each, yielding 60 training records (no oversampling).
Unlike the main E1 dose sweep (P10/P15/P25/P50, all on 59 distinct works),
E1B1 has far less musical diversity. Architecture, LR, horizon, and seed
are identical to the P-series.

| Experiment | Data | $p_{\text{swap}}$ | $\text{SDR}_{\text{test}}$ | $\text{SDR}_{\text{type1}}$ | $\text{SDR}_{\text{type2}}$ | $\Delta_{\text{cg}}$ |
|---|---|---|---|---|---|---|
| E1cosine-B1 | 60 segments (12 works × 5) | 0.25 | +1.61 | +1.25 | +1.43 | +0.17 |
| P25 (for reference) | 59 works | 0.25 | **+2.35** | **+1.97** | +1.93 | −0.05 |

E1B1's test mean (+1.61) is 0.74 dB below P25 (+2.35) despite the same
swap dose, confirming that **work-level diversity (59 distinct movements)
is more effective than segment-level diversity (60 crops from 12
movements)**. This result is also consistent with E1B1's 12 source works
possibly overlapping the 59-work train set, and is a single-seed run.

---

## 5. Summary

| Argument | Core finding | Evidence strength | Main limitation |
|---|---|---|---|---|
| 1. Author-release layout gap | No-intervention baseline type1 = −0.33, $\Delta_{\text{cg}}=+1.97$ | Suggestive | Architecture/data differ from later experiments; no TruePure baseline or direct audio-dependence measurement |
| 2. Intervention comparison | Swap type1 +1.56 vs Noise/Drop +0.03~+0.68 | Descriptive only | LR schedule confound (cosine vs const), strengths uncalibrated |
| 3. E1 dose sweep | P25/P15/P50 all near-zero $\Delta_{\text{cg}}$; P25 highest SDR (+2.35), P15 best type1 (+2.17), P50 matched | **Most solid** | Single seed; differences ≤0.1 dB between top three, may be within noise |
| 4. E4 vs E1 | E4 comparable in $\text{SDR}_{\text{test}}$ (+2.24) at ep1000, but $\Delta_{\text{cg}}$ wider (+0.32 vs −0.05 to +0.03 for E1) | Moderate | Shared $\sigma(p)$ signal, single $\lambda$, E4 combinatorial generalization less stable |
| 5. Data ablation (E1B1) | 60 segments (12 works) underperform 59-work P25 by 0.74 dB | Moderate | Single seed, possible work overlap |

**Core conclusion**: Within this model and protocol, P15/P25/P50 have near-zero $\Delta_{\text{cg}}$ and reported $\text{SDR}_{\text{test}}$ from +2.22 to +2.35. E4 reaches +2.24 overall SDR but has a wider $\Delta_{\text{cg}}=+0.32$ at ep1000. These are empirical comparisons of the evaluated runs. They do not prove that a position shortcut is the bottleneck or that the interventions suppress one. P15 has the best type1 result (+2.17), and P25 the highest overall SDR (+2.35). The E1B1 data-ablation run (60 segments from 12 works) scores 0.74 dB below P25; this single comparison motivates further study of work-level diversity but does not establish a general rule.

---

## 6. Data Provenance

| Experiment | Result file | Eval point | Notes |
|---|---|---|---|
| author_release_12ep_warmup_cosine | `results/test_results/author_release_12ep_warmup_cosine_test_results.json` | final | split_stage_paper, 12 works |
| P15 (for Arg. 2) | `results/checkpoint_evals/P15_test_ep274.json` | ep274 | mid_dual_paper, 59 works, cosine |
| E2 PureNoise | `results/checkpoint_evals/E2_test_ep274.json` | ep274 | const LR, 300ep |
| E3 PureDrop | `results/checkpoint_evals/E3_test_ep274.json` | ep274 | const LR, 300ep |
| P10 (ep424) | `results/checkpoint_evals/P10_test_ep424.json` | ep424 | E1 dose, cosine, 59 works |
| P15 (ep424) | `results/checkpoint_evals/P15_test_ep424.json` | ep424 | E1 dose, cosine, 59 works |
| P25 (ep250) | `results/test_results/PureSwap_E1cosine-P25_1000ep_p025_test_results.json` | ep1000 | E1 dose p=0.25, cosine, 59 works |
| P50 (ep424) | `results/checkpoint_evals/P50_test_ep424.json` | ep424 | E1 dose, cosine, 59 works |
| E4 (ep424) | `results/checkpoint_evals/E4_test_ep424.json` | ep424 | gate invariance, cosine, 59 works |
| P10 (final) | `results/test_results/PureSwap_E1cosine-P10_1000ep_p010_test_results.json` | ep1000 | E1 dose, cosine, 59 works |
| P15 (final) | `results/test_results/PureSwap_E1cosine-P15_1000ep_p015_test_results.json` | ep1000 | E1 dose, cosine, 59 works |
| P25 (final) | `results/test_results/PureSwap_E1cosine-P25_1000ep_p025_test_results.json` | ep1000 | E1 dose p=0.25, cosine, 59 works |
| P50 (final) | `results/test_results/PureSwap_E1cosine-P50_1000ep_p050_test_results.json` | ep1000 | E1 dose, cosine, 59 works |
| E4 (final) | `results/test_results/PureSwap_E4_1000ep_pinv_test_results.json` | ep1000 | gate invariance, cosine, 59 works |
| E1cosine-B1 (data ablation) | `results/test_results/PureSwap_E1cosine-B1_1000ep_p025_test_results.json` | ep1000 | swap p=0.25, 60 segments (12 works), cosine |
