# Contributing

Install `python -m pip install -e ".[torch,test]"` and run `python -m pytest -q` plus `python scripts/verify_results.py` before proposing changes.

Keep historical score JSONs immutable. Submit new runs under distinct names with configuration, dataset selection/hash, seed, update count, effective runtime settings and per-recording results. Numerical changes to inference, augmentation, conditioning, checkpoint recomputation or normalization need a new protocol version and fresh evaluation; old metrics must not be relabeled as new results.

Use validation data for hyperparameter/checkpoint decisions and disclose any test-set development feedback. Compare interventions with the same source data, route, optimizer/scheduler, batch size, update budget and inference version. Keep custom MuseScore-generated data separate from the original SQR data and document source-score identities across all splits.

Do not commit audio, model checkpoints, credentials, source archives or derived caches. Data and third-party assets keep their own terms. Include concise verification evidence and disclose environments or full training checks that were not run.
