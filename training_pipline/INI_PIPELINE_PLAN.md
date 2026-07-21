# INI-Driven Modular Wake-Word Pipeline

## Goal

Replace the JSON end-to-end experiment configuration with an INI file read by
Python's `configparser.ExtendedInterpolation`.  Dataset JSONL paths live in
the relevant augmentation, feature, or testing block; no separate `data.json`
registry is used.

## Configuration Contract

- `[main]` holds experiment-wide paths, sample settings, and pipeline state.
- `[steps]` contains the exact comma/newline-separated execution order.
- `augment.*`, `feature.*`, and `testing.*` are independently runnable
  sub-blocks.  Common sections are reused only through explicit interpolation.
- A feature block defines its output NPY, label, and split.  `[train]` names
  feature blocks for train/dev/false-positive validation and supplies one
  `batch.<feature-block> = N` line for every training source.
- `[export]` converts the final PyTorch training artifact to ONNX.
- Each `testing.*` block scores its set once and writes a Markdown FA/hour plus
  FA-rate table for negatives or an FR-rate table for positives over its own
  inclusive threshold range.

## Modular Layout

`src/wuw_pipeline.py` remains a thin CLI compatibility entrypoint.  The new
`src/wuw_training/` package owns:

- INI loading and validation;
- ordered pipeline dispatch and status reporting;
- durable stage completion checkpoints;
- separate augment, feature, train, export, and per-set testing stages;
- manifest normalization and artifact helpers.

The legacy JSON commands remain available temporarily and emit a deprecation
warning for end-to-end runs.  Existing low-level audio helpers are reused by
the new stages rather than duplicated.

## Checkpoints and Resume

Pipeline completion records live under `[main] pipeline_checkpoint_dir` and
are written only after a stage's declared outputs validate.  Their fingerprints
include resolved stage configuration, input signatures, and upstream checkpoint
digests, so stale stages and their consumers rerun automatically.

Model-training checkpoints are separate.  They save model and optimizer state,
phase/global-step position, phase schedule state, retained validation models,
history, sampler state, Python/NumPy/PyTorch random states, and AMP scaler state
when enabled.  The trainer resumes from the newest valid checkpoint; it starts
fresh when none is compatible.

## Training and Evaluation

The three-phase policy from the existing trainer is retained but made explicit
through `phase_step_ratios` and `phase_learning_rates`.  Training writes a
PyTorch state artifact only; export is a separate ONNX stage with optional
ONNX Runtime parity verification.

The trainer supports `model_type = dnn`, `rnn`, `cnn`, and `attention`.
CNN and convolution-attention settings are supplied through documented
`model.*` options in `[train]`, saved into the training checkpoint, and reused
by the export stage when it rebuilds the model. Attention export uses ONNX
opset 14 or newer automatically.

Each `testing.*` block writes its own Markdown threshold summary and raw,
threshold-independent sliding-window details. It reports FR rate for positive
sets and both FA/hour and per-clip FA rate for negative sets without
abnormal-case output. Thresholds such as `0.1` through `0.9` at `0.2`
increments are configured independently for every test block.

## Verification

Tests cover INI validation, ordered dispatch, checkpoint skipping and
invalidation, tiny feature training/resume, ONNX export, and threshold
aggregation.  A small isolated subset of `training/expts5/data` is used for a
CPU smoke test; full experiment datasets are never consumed by the smoke test.
