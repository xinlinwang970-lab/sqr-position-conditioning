# Data provenance and interfaces

The primary source is [String Quartet Set with Reverb, Zenodo record 14793971](https://zenodo.org/records/14793971). The record supplies the train/validation archive and the URMP-based test archive and explicitly lists six held-out validation IDs. The release builder uses those frozen IDs without reorganizing audio on disk.

The [paper, Section 4.1](https://www.isca-archive.org/interspeech_2025/xu25b_interspeech.pdf) describes EnsembleSet/BBCSO synthetic stems for development and isolated URMP recordings for testing, processed with hall reverberation. This repository consumes the distributed stereo WAVs. It neither redistributes those WAVs nor reconstructs the original B-format rotation/decoding pipeline.

## Local selections

| Selection | Training records | Validation records | Test records | Role |
|---|---:|---:|---:|---|
| `sqr_paper_split_*` | 59 type2 | 12 (6 IDs x 2 layouts) | 12 (6 IDs x 2 layouts) | Main local SQR experiments |
| `historical_baseline/*` | 60 type2 | 10 (5 IDs x 2 layouts) | 12 | Recovered historical selection metadata |
| E1cosine-B custom quartet data | Unresolved | Unresolved | URMP score grid recorded | Separate provenance; not same-data SQR control |

Validation IDs in the main selection are `8-CaraSposa`, `16-contrapunctusVI`, `27-k464`, `31-MiseroPargoletto`, `51-RM-C015`, `59-SentiSeLIngoanai`. Main test IDs are `24-Pirates`, `26-King`, `32-Fugue`, `36-Rondeau`, `38-Jerusalem`, `44-K515`.

The paper's prose says 65 training songs plus six validation songs, while the local release selection used here contains 59 training IDs plus six validation IDs. Preserve this discrepancy instead of silently expanding or relabeling the local set. Frozen manifests are the execution interface for these experiments, not a corrected edition of the paper.

Historical baseline manifests were recovered from `data_rms_normalized/manifests`, whereas the baseline's saved configuration points to `data/cache_dense_author_release/cache_index.json`. Those metadata do not establish that baseline training used RMS-normalized audio. They also do not prove that a newly generated cache matches the old one. Do not mix this historical 60/5 split into a 59/6 comparison without labeling it.

## Audio and labels

- WAV, sample rate 44,100 Hz, two channels, aligned mixture/targets with equal frame counts.
- Canonical target order: `violin1`, `violin2`, `viola`, `cello`.
- Timbre IDs: `0`, `0`, `1`, `2` respectively.
- type1 position IDs: `3`, `0`, `1`, `2`; type2: `3`, `2`, `1`, `0`.
- Physical seat prefixes in distributed stems are 1-based; model position IDs are prefix minus one. The two violin identities follow the part name, not seat sorting.
- Loader/cache preserve float sample amplitudes. No independent stem normalization, resampling, trim or gain is added by these tools.
- STFT: FFT/window 2048, hop 1024, periodic Hann, centered, reflect padding, unnormalized. Training crop is `(176-1)*1024=179200` samples, about 4.06349 s.
- Packed spectrogram order is `[real_left, real_right, imag_left, imag_right]`, with shape `[batch, 4, frames, frequency_bins]`.

Each manifest record contains identifiers, split, layout, mixture/stem relative paths, labels, sample rate/channel/frame count and fixture status. Main frozen manifests also include whole-file hashes. New builder output uses forward slashes on all platforms. Recovered baseline manifests have fewer metadata fields and no original audio hashes. The historical baseline calls its validation split `validation`; setup and cache tools accept this alias for `val` without rewriting the source metadata.

## Preparing a new extraction

`scripts/build_data_split_manifests.py --data-root ... --output-dir ...` finds exactly one directory for each selected item below the chosen root. It permits a shared train/validation extraction and rejects missing/ambiguous directories, wrong labels, wrong headers, changed hashes by default, and existing output files. No audio file is moved. `--allow-hash-differences` explicitly marks a new audio variant in `build_report.json`.

The training `--check-only` boundary verifies manifest schema, selected record counts, nonempty IDs, file existence and WAV headers. It is not full acoustic QA: it does not listen, detect musical errors, verify every sample's finiteness or enforce full-song mixture summation. The synthetic integration tests verify the loader/cache contract on generated WAVs. Real-dataset waveform QA remains a separate task.

Composition-ID disjointness does not automatically detect the same composition under different names, movement excerpts, arrangements or source-score overlaps. Claims of independent works require the original score-level grouping and a dedicated audit. No such broad musicological claim is made here.
