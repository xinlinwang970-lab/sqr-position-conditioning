# Known limitations retained with the historical results

## Data and causal comparison

The dense baseline uses a 60-item/five-validation-ID selection and `split_stage_paper`; the main P10/P15/P50/E2/E3/E4 scripts select 59 type2 items and use `mid_dual_paper`. Baseline sampling is exhaustive dense slabs; the others use one deterministic random crop per item/part/epoch. Baseline, constant-LR and cosine experiments have different update budgets. Consequently, a score difference cannot be assigned to one augmentation alone.

E1cosine-B's saved checkpoint config names a custom `quartet-data` training root and manifest. Its source describes oversampling a small custom set to 236 queries/epoch. The original custom selection was not recovered; its exact record count and music/audio provenance are unresolved. Its result JSON points to another server path. Retain its scores as historical observations, exclude it from a same-SQR-data dose comparison, and do not identify it as the missing p=0.25 SQR control.

Composition IDs are dataset labels. Checks establish disjoint labels, not a verified musicological audit of all arrangements, alternate names or shared movements. Upstream SQR may include arrangements for four string parts; four stems do not establish original string-quartet composition status.

## Actual activation recomputation

P10/P15/P50/E2/E3 install unconditional manual activation-checkpoint wrappers around three encoder blocks, one middle block and three decoder blocks, even though their YAML field is `model.activation_checkpointing: false`. Baseline/E1cosine-B/E4 do not install those wrappers. Backward recomputation updates the wrapped BatchNorm running statistics again. A regression characterization observes `num_batches_tracked` increasing once in forward and again in backward. This is a protocol confound, not just a memory setting.

The release records the actual wrapper setting in launch metadata and preserves historical behavior. Turning wrappers off or altering BatchNorm semantics would define a new experiment; its old scores cannot be reused.

## E4 regularizer and dropout

`compute_position_invariance_reg` compares decoder-controller parameters for original and permuted position labels using two separate `_condition_params` calls while training. Each call uses independent controller dropout masks (historically p=0.1). Even an identity permutation can produce positive regularization loss. In eval mode, the identity-permutation loss is zero. Therefore this implementation does not isolate positional invariance from controller stochasticity. Its score cannot confirm the proposed mechanism.

## Legacy full-length inference

The training scripts and `evaluate_checkpoint.py` use periodic Hann overlap-add with 50% overlap and a clamped denominator. The first Hann coefficient is zero, so the first output sample is zero. For input shorter than one crop, padding changes the returned length. Later metric code truncates to the shorter of estimate/reference lengths. Generic config options advertising a positive blend floor and original-length retention are not implemented by these legacy script functions.

These behaviors are tested and disclosed, not silently repaired beneath previously reported scores. A corrected inference protocol must be versioned and all relevant checkpoints reevaluated with it.

## Checkpoint restoration

Historical scripts save model/optimizer/config/update metadata, and the baseline also saves sampler state. They do not save Python, NumPy and Torch RNG states. On restart, the scripts reset seeds, making future corruption/dropout sequences differ from an uninterrupted run. Auto-resume exists for the main augmentation scripts but does not imply bitwise reproducibility. The separate synthetic trainer has stronger restart tests; that coverage does not transfer to the historical entry points.

Eight original checkpoint files were hashed and safely inspected locally with `weights_only=True`; that evidence is metadata. A result JSON's path/update labels do not cryptographically bind its scores to those weights. This release did not rerun waveform inference to close that link. All release entry points explicitly request safe checkpoint loading; only supply trusted compatible tensor/config files.

## Metrics and inference scope

The reported quantity is gain-dependent waveform SDR, not SI-SDR or a BSS Eval filter-based SDR. Multiplying a perfect reference by 0.5 yields about 6.0206 dB under this metric. The six URMP source recordings are reused across two layouts and four parts; queries are correlated. One seed and repeated test observation do not establish an optimal probability, a causal shortcut mechanism, or robust transfer to independent simultaneous quartet performances.

Audio normalization is not performed by the released WAV loader or cache builder. The upstream synthesis procedure and separate local RMS experiments must be tracked independently; applying per-stem normalization again changes the task and mixture consistency. MuseScore-rendered audio requires its own dataset/version/provenance and is not covered by these historical SQR scores.
