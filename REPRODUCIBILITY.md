# Reproducibility status

This release supports verification at several distinct levels.

| Level | Evidence and scope | Status at release preparation |
|---|---|---|
| Saved-score arithmetic | `python scripts/verify_results.py`; 11 original result JSONs, 528 query rows, complete six-recording grids | Verified locally |
| Preserved evidence bytes | SHA-256 for imported logs/configs/selections and original result files | Verified locally |
| Model/data/tool behavior | CPU pytest suite, real WAV fixtures, forward/backward, routing and cache checks | See `results/validation/` for exact run |
| Checkpoint identity | Eight original files were hashed and safely loaded during the local audit; metadata is bundled | Local inspection only; weights excluded |
| Fresh full-song inference | Requires matching weights and upstream audio | Not rerun in this preparation |
| Full GPU training | Requires original audio/cache identity, training environment and budget | Not rerun |
| Bitwise historical restart | Historical scripts do not save RNG states | Not supported |
| Original-paper reproduction | Architecture/data/metric differences remain | Not established |

## Environment

The local release checks use Windows 11, Python 3.14.3, PyTorch 2.14.0+cu126 on CPU, NumPy 2.4.3, SoundFile 0.13.1, PyYAML 6.0.3 and pytest 9.0.3. The actual environment and JUnit results are retained under `results/validation/`. CUDA availability does not imply that GPU training was tested.

`pyproject.toml` declares Python >=3.10,<3.15. The CI workflow runs CPU checks on Linux with Python 3.10 and 3.11, but its presence is not evidence that remote CI has run. Dependencies are not a lockfile for historical training. A future replication should pin its complete environment before training and record GPU, driver, CUDA and library versions.

Install with `python -m pip install -e ".[torch,test]"`, then run `python -m pytest -q` and `python scripts/verify_results.py` from the repository root. Bundled relative data/run defaults resolve from the repository; explicitly supplied CLI paths resolve from the caller's current directory.

## Data identity

`manifests/sqr_paper_split_*.jsonl` preserve the 59/12/12 release selection and file hashes. They are not a guarantee that the extracted files on a new machine match. Rebuild into a new directory and inspect `build_report.json`. The builder reads shared extracted trees, preserves split membership, and never moves or normalizes WAVs.

Historical baseline selection uses 60 training items and five validation composition IDs (10 layouts). The metadata was recovered from a normalization workspace. Its original training configuration points to a different cache under the original data workspace. The actual original cache index and its WAV-byte provenance were not bundled. Rebuilding the cache with current inputs is a new run, not proof of exact baseline identity.

E1cosine-B is separately archived. Its original checkpoint configuration names `quartet-data/train`, while the result JSON names a checkpoint on a different server path. The original custom train/val manifests are unavailable here. Neither matching a run name nor matching update counts proves that this was the same 59-item SQR experiment. The command therefore requires explicit custom roots and manifests, and the report excludes it from same-data causal comparisons.

## Historical numerical behavior

No model-core numerical changes were made for release preparation. Non-main functions/classes in all eight training scripts were compared with the source snapshot at the Python AST level; the inventory is under `results/provenance/numerical_function_preservation.json`. This check covers training helpers and inference functions, not an end-to-end trajectory. Main-function changes are path setup, validation, launch metadata and safe checkpoint loading.

Important retained differences are described in [KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md): actual activation checkpoint wrappers differ across experiments; E4's two controller calls use independent dropout masks; the legacy overlap-add has a zero first-sample endpoint and pads short recordings; historical restarts do not restore RNG state. Fixing these for a new protocol requires newly named runs and fresh results.

Training script constants control seed, batch size, target epochs and interventions. The config alone is not the effective protocol. Some generic `inference` YAML fields (positive blend floor, original length) are ignored by legacy script inference. `release_launch.json` records the actual wrapper setting and selected paths. Do not edit a YAML field and assume every historical script consumes it.

## Result selection

Files ending in `ep274` and `ep424` use zero-based epoch labels: they correspond to 275 and 425 completed epochs. Tables must also report update counts. Dense baseline epochs enumerate 17,412 queries and 4,353 full batches; random-crop SQR epochs enumerate 236 queries and 59 full batches. Their epoch numbers are not directly comparable.

The URMP release was evaluated repeatedly during development. Available intermediate evaluations are bundled rather than hidden. These are development observations, not an untouched final test. Future hyperparameter/checkpoint selection must use validation data, with independent new test material for a confirmatory study. Four parts and two layouts from the same recording are correlated; 48 scores are not 48 independent works.

No numerical tolerance such as "within 0.05 dB" is promised for a new training run. No cross-work or synthetic-to-real population conclusion follows from this release alone.
