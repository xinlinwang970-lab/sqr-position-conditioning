# Release preparation audit — 2026-11-19

The starting point was the 99-file pending release associated with Git commit `0f6f63c587672f68086c6a1a1dff3d018d81d179`. Its tracked bytes matched that commit; exFAT executable-mode differences were not substantive code modifications. Its 41 Python source files matched the corresponding local research source files.

The release was prepared in a separate copy. The original research directory, pending release and original checkpoint/audio files were not modified. No GitHub upload, commit or push was performed.

## Corrections made

- Relabeled the metric as gain-dependent waveform SDR and documented its formula.
- Distinguished completed epochs from zero-based filename labels and from update budgets.
- Separated the 59/6 main selection, historical 60/5 baseline metadata and unresolved E1cosine-B custom data.
- Corrected the E3 comparison: type2 median 1.6310055 at the 275-epoch snapshot; 1.6860574 belongs to the later final result.
- Removed claims of an optimal swap probability, proven shortcut mechanism, fully matched interventions, guaranteed reproduction tolerance, or established population-level transfer.
- Corrected the data/paper links and distinguished SQR/BBCSO from the separate MuseScore generation workflow.
- Replaced server-dependent setup paths with explicit CLI options and launch metadata. Numerical helper functions and model-core files were preserved.
- Added a manifest rebuilder for unseparated extractions and a portable, unnormalized dense-cache builder.
- Restored core model, dataset, evaluator and synthetic training tests, and added release-tool plus legacy-behavior checks.
- Included available logs, intermediate evaluations, original configs and checkpoint checksum metadata. Preserved all 11 pending-release result files byte-for-byte.
- Removed the invented GitHub organization URL from citation metadata. Existing project-team attribution and MIT license were retained; actual contributor identity was not inferred.

## Verified and remaining boundaries

`results/validation/` holds the exact local verification environment, JUnit results, entrypoint smoke checks and saved-score recomputation. The main acceptance checks are path portability, fixture-based data preparation/cache/loading, model forward/backward and condition routing, full saved-result arithmetic, original-result hashes and clean packaging. Read the evidence for exact counts and exit statuses.

During preparation, a private Windows validation wrapper initially lacked the multiprocessing main guard and recursively launched pytest workers. That wrapper was corrected and the whole suite rerun; this was a validation-harness error, not a production-model fix. The local preparation record retains the failed run. Public tests are not skipped to conceal it.

Fresh full-song inference, long GPU training and remote Linux CI were not rerun. Weights/audio are excluded, and original baseline cache identity plus E1cosine-B manifests are not recovered. These gaps limit scientific reproducibility even when the software checks pass.

The earlier local audit missed the custom-data root in E1cosine-B. Rechecking the saved checkpoint configuration exposed it. This report and the experiment table supersede the previous same-data interpretation; the archived numeric scores remain intact.

## Source traceability

`results/provenance/release_origin_files.json` records the starting file inventory. `imported_files.json` describes recovered evidence with byte hashes and source locations. `checkpoint_verification.json` records the eight original checkpoint inspections. `numerical_function_preservation.json` records AST-level equivalence of non-main training functions/classes. Full package hashes are provided separately with the prepared release.

Logs and historical configs retain original server path strings as provenance. They are not active defaults. No audio redistribution or new licensing determination is implied by the presence of metadata. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
