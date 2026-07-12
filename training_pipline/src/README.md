# WUW openWakeWord Training Pipeline

Reusable tools for preparing JSONL-described audio data, training, and
evaluating a local openWakeWord model.

## INI Pipeline (recommended for new experiments)

The modular pipeline uses Python's `configparser` and runs the exact ordered
list in `[steps]`. JSONL paths are specified directly in `augment.*`,
`feature.*`, and `testing.*` blocks; `data.json` is not required.

Start from [wakeword_pipeline.ini.example.conf](../examples/wakeword_pipeline.ini.example.conf)
and the design notes in [INI_PIPELINE_PLAN.md](../INI_PIPELINE_PLAN.md).

```bash
PYTHON=/home/alicekenway/miniconda3/envs/openwake/bin/python
PIPELINE=/home/alicekenway/Dev/project/WUW/openWakeWord/training_pipline/src/wuw_pipeline.py
CONFIG=/path/to/config.ini

"$PYTHON" "$PIPELINE" validate --config "$CONFIG"
"$PYTHON" "$PIPELINE" run --config "$CONFIG"
"$PYTHON" "$PIPELINE" run-step --config "$CONFIG" feature.positive_train
"$PYTHON" "$PIPELINE" status --config "$CONFIG"
```

Use `--from`, `--to`, or repeat `--force <stage>` with `run` to control a
partial rerun. Pipeline completion files and model-training checkpoints are
separate. The former JSON end-to-end commands remain below for compatibility
but are deprecated for new work.

### INI section contracts

- `[main]` defines the experiment directory, model name, feature-model assets,
  sample rate, clip length, seed, and pipeline checkpoint directory.
- `[steps] steps = ...` is the single source of execution order. Use named
  substeps such as `augment.positive_train`, `feature.negative_cv_train`, and
  `testing.background`.
- An `augment.*` block accepts `input_jsonl`, optional `audio_base_dir`, one or
  more `noise_jsonl`/`noise_dir` sources, augmentation settings, and an output
  WAV directory plus output manifest.
- A `feature.*` block accepts one JSONL path or a JSON list of paths, emits one
  NPY file, and declares `label` plus `split`. Relative audio paths use
  `audio_base_dir` when present, otherwise the JSONL file's parent directory.
- `[train]` names feature blocks with `train`, `dev`, and `false_positive`.
  Each named training block needs a matching line such as
  `batch.feature.negative_cv_train = 128`. The stage saves resumable model
  checkpoints separately from its final `.pt` model artifact.
- `[export]` reads the final training `.pt` artifact and writes ONNX. It can
  verify PyTorch/ONNX Runtime inference parity.
- Every `testing.*` block evaluates one positive or negative set once, sweeps
  its configured inclusive threshold range over the resulting window scores,
  and writes `threshold_summary.md` plus threshold-independent score details.
  Positive reports show false-reject rate; negative reports show false accepts
  per hour. No abnormal-case file or JSON summary is produced.

Configure a per-set threshold sweep explicitly or interpolate common values:

```ini
[testing.common]
threshold_range = [0.1, 0.9]
threshold_step = 0.2
debounce_seconds = 1.0

[testing.positive]
threshold_range = ${testing.common:threshold_range}
threshold_step = ${testing.common:threshold_step}
output_report = ${main:experiment_dir}/evaluation/positive/threshold_summary.md
```

`threshold_step` must reach the inclusive end of `threshold_range` exactly.

Only genuine lists use JSON syntax within the INI, for example
`phase_learning_rates = [0.0001, 0.00001, 0.000001]` or
`input_jsonl = ["original.jsonl", "${augment.positive_train:output_manifest}"]`.
Shared values are referenced explicitly with `${section:option}`; INI sections
do not inherit values implicitly.

### CNN and attention model choices

`model_type` now accepts `dnn`, `rnn`, `cnn`, or `attention`. The standalone
CNN and convolution-attention heads return logits as normal PyTorch modules;
the training wrapper applies their final sigmoid so exported wake-word ONNX
models continue to return score values between zero and one.

For `model_type = cnn`, set these optional `[train]` options:

```ini
model.channels = 128
model.expansion = 1
model.dropout = 0.05
model.classifier_hidden = 64
model.cnn_kernels = [3, 5, 3, 3]
model.cnn_dilations = [1, 1, 2, 4]
model.cnn_use_se = [false, false, true, true]
```

For `model_type = attention`, use the convolution-attention head:

```ini
model.channels = 128
model.expansion = 1
model.dropout = 0.05
model.classifier_hidden = 64
model.attention_time_steps = 16
model.attention_num_heads = 4
model.attention_ff_multiplier = 2
model.attention_local_kernels = [3, 3]
model.attention_local_dilations = [1, 2]
model.attention_local_use_se = [false, false]
```

The attention channel count must divide evenly by `model.attention_num_heads`,
and `model.attention_time_steps` must match the generated NPY feature time
dimension. Both are validated before training starts. The export stage uses
ONNX opset 14 automatically for attention models when the configured opset is
lower, because PyTorch attention export requires it.

### Checkpoint behavior

`pipeline_state/*.done.json` marks an entire successful stage. It is valid only
when the resolved stage configuration, inputs, upstream checkpoints, and output
signatures still match. A changed upstream stage therefore invalidates its
consumers automatically.

`[train] model_checkpoint_dir` stores model/optimizer/schedule/RNG state during
training. With `resume = yes`, the latest compatible checkpoint resumes the
three-phase training policy. It is distinct from the pipeline completion marker
and from the final exported ONNX model.

Training prints loss and learning-rate progress during the run and appends the
same events to JSONL. Configure the cadence and file in `[train]`:

```ini
log_interval_steps = 100
validation_interval_steps = 500
training_log_file = ${main:experiment_dir}/trained_model/training.jsonl
```

The JSONL includes `run_start`, `phase_start`, `train_step`, `validation`,
`checkpoint`, and `run_complete` events. Resumed runs append a new session ID
to the same file.

Use this Python environment:

```bash
export PYTHON=/home/alicekenway/miniconda3/envs/openwake/bin/python
export PIPELINE=/home/alicekenway/Dev/project/WUW/openWakeWord/training_pipline/src/wuw_pipeline.py
export EXP=/home/alicekenway/Dev/project/WUW/training/expts2
```

## Data Input Contract

All source datasets enter the pipeline as JSONL files. Each JSONL row needs one
audio path field:

```json
{"path": "wav/000000000.wav", "text": "Hey Siri"}
```

New output uses exactly one canonical `path` field. Readers also accept the
legacy keys `audiofile_path`, `audio_file`, `audio_path`, `file`, and
`filename` so older source datasets remain usable.

In a config file, each dataset can be either a single JSONL path:

```json
"positive_jsonl": "/mnt/d/wuw_data/eng/wuw_audio/metadata.jsonl"
```

or a list of dataset objects:

```json
"negative_jsonl": [
  {
    "jsonl_path": "/mnt/d/wuw_data/eng/non_wuw_audio/metadata.jsonl",
    "audio_base_path": "/mnt/d/wuw_data/eng/non_wuw_audio"
  },
  {
    "jsonl_path": "/mnt/d/wuw_data/eng/cv_audio/metadata.jsonl",
    "audio_base_path": "/mnt/d/wuw_data/eng/cv_audio"
  }
]
```

`audio_base_path` is optional. If it is omitted, relative audio paths are
resolved relative to the JSONL file's parent directory.

For plain audio folders with no metadata JSONL, create one first:

```bash
"$PYTHON" "$PIPELINE" index-audio \
  --audio-dir /mnt/d/wuw_data/background \
  --output-jsonl /mnt/d/wuw_data/background/metadata.jsonl \
  --source background \
  --label 0
```

## Experiment Config

Example:

```json
{
  "experiment_dir": "/home/alicekenway/Dev/project/WUW/training/expts2",
  "model_name": "wuw_expts2",
  "positive_jsonl": {
    "jsonl_path": "/mnt/d/wuw_data/eng/wuw_audio/metadata.jsonl",
    "audio_base_path": "/mnt/d/wuw_data/eng/wuw_audio"
  },
  "negative_jsonl": [
    {
      "jsonl_path": "/mnt/d/wuw_data/eng/non_wuw_audio/metadata.jsonl",
      "audio_base_path": "/mnt/d/wuw_data/eng/non_wuw_audio"
    }
  ],
  "background_jsonl": {
    "jsonl_path": "/mnt/d/wuw_data/background/metadata.jsonl",
    "audio_base_path": "/mnt/d/wuw_data/background"
  },
  "positive_dev_count": 500,
  "positive_test_count": 500,
  "negative_train_count": 50000,
  "negative_dev_count": 500,
  "negative_test_seconds": 3600,
  "background_train_count": 5000,
  "background_dev_count": 300,
  "background_test_seconds": 3600,
  "steps": 5000,
  "device": "auto",
  "skip_download": true
}
```

## Run The Whole Pipeline

Run through manifests, conversion, augmentation, feature generation, training,
and evaluation:

```bash
"$PYTHON" "$PIPELINE" run-experiment \
  --config /home/alicekenway/Dev/project/WUW/training/expts2/experiment_config.json
```

Run through the shell wrapper:

```bash
CONFIG=/home/alicekenway/Dev/project/WUW/training/expts2/experiment_config.json \
  /home/alicekenway/Dev/project/WUW/openWakeWord/training_pipline/src/run_experiment.sh
```

Run a small smoke test:

```bash
"$PYTHON" "$PIPELINE" run-experiment \
  --config /home/alicekenway/Dev/project/WUW/training/expts2/experiment_config.json \
  --experiment-dir /home/alicekenway/Dev/project/WUW/training/smoke \
  --quick \
  --device auto
```

## Run From Existing Split JSONL

If train/dev/test JSONL files already exist, skip manifest preparation and run
augmentation, feature extraction, training, and evaluation directly:

```bash
"$PYTHON" "$PIPELINE" run-from-splits \
  --config /home/alicekenway/Dev/project/WUW/training/expts2/experiment_config.json
```

The split config uses `split_manifests.positive`, `split_manifests.negative`,
and `split_manifests.background`. Negative splits may be a list of JSONL files,
for example Common Voice plus local non-WUW audio. Background augmentation uses
only the `background.train` JSONL as the noise source.

Config values can be overridden from the CLI:

```bash
"$PYTHON" "$PIPELINE" run-experiment \
  --config /home/alicekenway/Dev/project/WUW/training/expts2/experiment_config.json \
  --threshold 0.65 \
  --debounce-seconds 1.0
```

## Run Stages Individually

Download or verify the feature/pretrained model assets:

```bash
"$PYTHON" "$PIPELINE" download-models \
  --output-dir "$EXP/models" \
  --models all
```

Prepare train/dev/test manifests:

```bash
"$PYTHON" "$PIPELINE" prepare-manifests \
  --config /home/alicekenway/Dev/project/WUW/training/expts2/experiment_config.json \
  --output-dir /home/alicekenway/Dev/project/WUW/training/expts2 \
  --negative-train-count 200000 \
  --negative-dev-count 300 \
  --negative-test-seconds 3600
```

Convert one manifest to fixed-length 16 kHz WAV:

```bash
"$PYTHON" "$PIPELINE" convert-manifest \
  --manifest "$EXP/manifests/positive_train.jsonl" \
  --output-dir "$EXP/audio/converted/positive_train" \
  --output-manifest "$EXP/audio/converted/positive_train.jsonl" \
  --placement end \
  --workers 16
```

Augment a converted manifest with background noise:

```bash
"$PYTHON" "$PIPELINE" augment-audio \
  --input-manifest "$EXP/audio/converted/positive_train.jsonl" \
  --noise-dir "$EXP/audio/converted/background_train" \
  --output-dir "$EXP/audio/augmented/positive_train" \
  --output-manifest "$EXP/audio/augmented/positive_train.jsonl" \
  --rounds 1 \
  --snr-low -5 \
  --snr-high 15 \
  --workers 16
```

Generate feature arrays:

```bash
"$PYTHON" "$PIPELINE" generate-features \
  --audio-manifest "$EXP/audio/converted/positive_train.jsonl" "$EXP/audio/augmented/positive_train.jsonl" \
  --output-file "$EXP/features/positive_train.npy" \
  --model-dir "$EXP/models" \
  --device auto \
  --batch-size 64 \
  --placement end
```

For large JSONL runs, feature extraction can prefetch audio loading while the
GPU computes the previous batch:

```bash
  --batch-size 256 \
  --audio-loader-workers 8 \
  --prefetch-batches 2
```

Train a model from generated feature arrays:

```bash
"$PYTHON" "$PIPELINE" train-model \
  --positive-train-features "$EXP/features/positive_train.npy" \
  --negative-train-features "$EXP/features/negative_train.npy" "$EXP/features/background_train.npy" \
  --positive-dev-features "$EXP/features/positive_dev.npy" \
  --negative-dev-features "$EXP/features/negative_dev.npy" \
  --false-positive-features "$EXP/features/background_dev.npy" \
  --output-dir "$EXP/trained_model" \
  --model-name turn_on_the_office_lights \
  --steps 2000
```

## Legacy JSON Evaluation CLI

The commands below document the deprecated legacy evaluator. New INI
`testing.*` stages use the per-set Markdown threshold reports described above
and do not create these JSON/abnormal artifacts.

Evaluate directly from CLI values:

```bash
"$PYTHON" "$PIPELINE" evaluate \
  --model "$EXP/trained_model/turn_on_the_office_lights.onnx" \
  --model-dir "$EXP/models" \
  --positive-manifest "$EXP/audio/converted/positive_test.jsonl" \
  --negative-manifest "$EXP/audio/converted/negative_test.jsonl" \
  --background-manifest "$EXP/audio/converted/background_test.jsonl" \
  --output-json "$EXP/evaluation/eval_summary.json" \
  --threshold 0.5 \
  --debounce-seconds 1.0
```

Evaluate from a config file:

```json
{
  "model": "/home/alicekenway/Dev/project/WUW/training/expts1/trained_model/turn_on_the_office_lights.onnx",
  "model_dir": "/home/alicekenway/Dev/project/WUW/training/expts1/models",
  "positive_manifest": "/home/alicekenway/Dev/project/WUW/training/expts1/audio/converted/positive_test.jsonl",
  "negative_manifest": "/home/alicekenway/Dev/project/WUW/training/expts1/audio/converted/negative_test.jsonl",
  "background_manifest": "/home/alicekenway/Dev/project/WUW/training/expts1/audio/converted/background_test.jsonl",
  "output_json": "/home/alicekenway/Dev/project/WUW/training/expts1/evaluation/eval_summary_threshold_050.json",
  "threshold": 0.5,
  "debounce_seconds": 1.0,
  "positive_padding": 1,
  "negative_padding": 0,
  "chunk_size": 1280,
  "model_window_seconds": 2.0,
  "record_window_scores": true
}
```

```bash
"$PYTHON" "$PIPELINE" evaluate \
  --evaluation-config "$EXP/evaluation/evaluation_config.json"
```

Rerun only evaluation with a new threshold:

```bash
"$PYTHON" "$PIPELINE" evaluate \
  --evaluation-config "$EXP/evaluation/evaluation_config.json" \
  --output-json "$EXP/evaluation/eval_summary_threshold_065.json" \
  --threshold 0.65
```

Limit evaluation for a quick check:

```bash
"$PYTHON" "$PIPELINE" evaluate \
  --evaluation-config "$EXP/evaluation/evaluation_config.json" \
  --output-json "$EXP/evaluation/eval_summary_debug.json" \
  --limit-positive 20 \
  --limit-negative-seconds 120
```

## Legacy Evaluation Outputs

Each evaluation writes:

- `eval_summary.json`: aggregate positive FR and negative FA metrics.
- `evaluation_config.json`: the resolved testing config, including threshold.
- `eval_details.jsonl`: one row per tested clip.
- `eval_abnormal.jsonl`: only abnormal rows: false rejects, false accepts, and
  evaluation errors.

Each detail row records:

- set name, clip id, source path, duration, expected label
- threshold, debounce, padding, chunk size, model window seconds
- max score, average score, median score, and best scoring window
- sliding-window score rows with start time, end time, raw padded times, and score
- threshold crossings and debounced events
- `false_reject`, `false_accept`, `abnormal`, and `abnormal_type`

For positive clips, abnormal means no sliding window crossed the threshold. For
negative or background clips, abnormal means at least one debounced event
crossed the threshold.

## Output Layout

The end-to-end run writes:

- `experiment_config.json`: run settings.
- `manifests/`: selected source files.
- `audio/converted/`: fixed-length WAV clips.
- `audio/augmented/`: augmented training clips.
- `features/`: openWakeWord `.npy` feature arrays plus summary JSON.
- `trained_model/`: ONNX model, PyTorch checkpoint, and training summary.
- `evaluation/`: independent per-set Markdown threshold reports and raw score
  details.
- `REPORT.md`: human-readable experiment notes and conclusions.

## CUDA Behavior

Feature generation uses CUDA automatically when both PyTorch CUDA and ONNX
Runtime CUDA are visible. The default is `--device auto`; use `--device gpu`
only when you want the command to fail if CUDA is not visible.

Training uses PyTorch CUDA when available. `--require-cuda` is optional and is
only a fail-fast guard for debugging GPU visibility.

Full negative runs spend most of their time decoding audio and writing
WAV files before CUDA is used. Use `--convert-workers` and `--augment-workers`
on `run-experiment` to parallelize those CPU/file IO stages.
