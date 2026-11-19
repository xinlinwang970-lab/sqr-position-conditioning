# Historical experiment design

This document describes executed code, rather than an idealized controlled study. Results are exploratory. See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) before comparing mechanisms.

## Shared interface

The model extracts one target per query from a stereo mixture. Timbre IDs distinguish violin/viola/cello; a position ID identifies a candidate seat. Violin 1 and violin 2 share a timbre ID and require position information to distinguish their queries under this interface.

The augmentation experiments use `mid_dual_paper`: encoder and middle blocks bypass conditioning; the decoder is modulated from the mean of timbre and position embeddings. The historical dense baseline uses `split_stage_paper`: timbre controls encoder, the mean controls middle, position controls decoder. These are materially different routes.

The main configurations use seven blocks, channel/growth value 124, five TFC layers, controller dropout 0.1, layer normalization in the controller, FP32, complex STFT MSE and Adam (base LR 0.0005, batch 4). Architecture labels containing `paper` do not establish exact equivalence with the original model. The implementation treats base/growth channels together and runs in explicit `w24_engineering_pilot` mode.

## Experiments

| Run | Data selection / sampling | Intervention | Scheduler / planned budget | Manual activation wrappers |
|---|---|---|---|---|
| Dense baseline | Historical 60 type2, all complete 179200-sample slabs x four queries | None | Warmup/cosine, 12 dense epochs, 52,236 updates | No |
| E1cosine-B | Custom quartet manifests, identities unresolved, oversampled to 236 queries/epoch | Batch-level position-label permutation probability 0.25 | Warmup/cosine, 1000 target epochs | No |
| P10/P15/P50 | Main SQR 59 type2, one crop/item/part/epoch | Batch-level permutation probability 0.10/0.15/0.50 | Warmup/cosine, 59,000 planned updates | Yes |
| E2 | Main SQR 59 type2 | Add Gaussian noise, standard deviation 0.10, to position embedding during training | Constant LR, 17,700 planned updates | Yes |
| E3 | Main SQR 59 type2 | Zero position embedding for selected samples, probability 0.25 | Constant LR, 17,700 planned updates | Yes |
| E4 | Main SQR 59 type2 | Add decoder parameter consistency penalty, lambda=1 | Warmup/cosine, 59,000 planned updates | No |

For swap, one random permutation of four seat labels is applied to a selected batch. Mixture audio, target audio and timbre labels remain unchanged. This is label corruption, not physical seat rerendering or target permutation. Random permutations can leave positions unchanged; the configured probability is not the exact per-query probability of a different label.

For E4, the original/permuted controller calls use separate dropout masks while training. Its penalty includes this stochastic difference. No causal claim that it exclusively enforces position invariance is supported.

The cosine scripts use 500 warmup updates and a 0.000005 endpoint over their full planned horizon. A 425-epoch checkpoint is partway through that horizon, not the endpoint of a new 425-epoch cosine schedule. Non-finite-loss handling advances counters without an optimizer step, so counter labels describe script progress rather than a separately audited count of effective parameter updates.

## Selection and follow-up

The original code evaluates validation during training and final test at the end, but separate evaluation scripts were also used to inspect test snapshots repeatedly. The retained records therefore expose test-set development feedback. Existing scores must not be presented as an untouched final evaluation.

For a future controlled study, freeze data/audio hashes, route, activation/BatchNorm behavior, inference version, budget and validation selection rule first. Run multiple seeds, separate custom MuseScore data, and use new independent test recordings. Correct E4 dropout control and any overlap-add issue in a newly named protocol before producing new metrics. Do not merge the custom E1cosine-B run into a P25 SQR dose curve.
