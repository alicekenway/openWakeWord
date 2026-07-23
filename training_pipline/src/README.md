# WUW openWakeWord Training Pipeline

Reusable tools for preparing JSONL-described audio data, training, and
evaluating a local openWakeWord model.

## INI Pipeline (recommended for new experiments)

The modular pipeline uses Python's `configparser` and runs the exact ordered
list in `[steps]`. JSONL paths are specified directly in `augment.*`,
`feature.*`, and `testing.*` blocks; `data.json` is not required.

Choose the example that matches your model and execution environment:

- [local_wuw.ini.example.conf](../examples/local_wuw.ini.example.conf):
  standard openWakeWord model on one machine.
- [slurm_wuw.ini.example.conf](../examples/slurm_wuw.ini.example.conf):
  standard openWakeWord model with Slurm arrays.
- [local_ctc_wuw.ini.example.conf](../examples/local_ctc_wuw.ini.example.conf):
  WeNet CTC + WAC two-stage model on one machine.
- [slurm_ctc_wuw.ini.example.conf](../examples/slurm_ctc_wuw.ini.example.conf):
  WeNet CTC + WAC two-stage model with Slurm arrays.

See also the design notes in [INI_PIPELINE_PLAN.md](../INI_PIPELINE_PLAN.md).

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

### Slurm mode

Set `[main] execution_mode = slurm` to submit the same INI pipeline to Slurm.
The submission host prepares each stage, waits for it to finish, and only then
starts the next one. `augment.*`, `feature.*`, and `testing.*` use a Slurm job
array; `train`, `export`, and `summary` use one Slurm job.

Each listed stage needs a matching resource section. `sbatch_args` is passed
as arguments to `sbatch`, so normal settings such as `--mem`, `--gres`, and
`--partition` work directly.

```ini
[main]
execution_mode = slurm

[slurm]
sbatch_command = sbatch
squeue_command = squeue
python_executable = /path/to/openwake/bin/python
setup_commands =
    module load cuda

[slurm.feature.positive_train]
tasks = 16
sbatch_args = --partition=gpu --mem=24G --gres=gpu:1 --cpus-per-task=4

[slurm.train]
sbatch_args = --partition=gpu --mem=48G --gres=gpu:1
```

The controller writes generated scripts, logs, and per-task state beneath
`<experiment_dir>/.pipeline_work/<stage>/slurm/`. If one array task fails, it
waits for all sibling tasks, reports every failed/missing task, and stops the
pipeline. On the next run, completed shard outputs are reused and only failed
or missing tasks are submitted. Once a stage merges successfully, its large
temporary shard outputs are removed while final artifacts, task metadata, and
logs remain.

If `tasks` is greater than the number of input records, the pipeline submits
one task per record instead and records both requested and actual counts in
the stage result.

Slurm workers must be able to see the repository, the source/configuration
paths, model assets, and experiment directory. The default worker Python is
the Python used to start the controller; set `python_executable` and optional
trusted `setup_commands` when compute nodes need a different environment.

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
- `[train]` names labeled feature blocks with `train` and `dev`. Each named
  training block needs a matching line such as
  `batch.feature.negative_cv_train = 128`. The stage saves resumable model
  checkpoints separately from its final `.pt` model artifact. Validation is
  threshold-free: it reports BCE loss independently for every dev block, plus
  aggregate, positive-label, and negative-label dev losses. Older configs with
  `false_positive` remain readable and are merged into the labeled dev list.
- `[export]` reads the final training `.pt` artifact and writes ONNX. It can
  verify PyTorch/ONNX Runtime inference parity.
- Every `testing.*` block evaluates one positive or negative set once, sweeps
  its configured inclusive threshold range over the resulting window scores,
  and writes `threshold_summary.md` plus threshold-independent score details.
  Positive reports show false-reject rate; negative reports show false accepts
  per hour and false-accept rate (the share of evaluated clips with at least
  one false accept). No abnormal-case file or JSON summary is produced.

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

### Optional: WeNet CTC + WAC two-stage wake word

There is an opt-in pipeline for the design we discussed:

1. A frozen WeNet CTC model is stage 1. It returns the complete encoder
   matrix `[T, D]` and CTC log-probability matrix `[T, V]` for each augmented
   waveform. The pipeline scores every keyword, finds the best normalized CTC
   candidate, and records its non-blank start/end frames.
2. A small WAC-like model is stage 2. It receives only the cropped encoder
   candidate, its length-normalized CTC score, the top-one minus top-two
   margin, and a one-hot value saying which wake word won in stage 1. Candidate
   lengths remain variable: training groups nearby lengths, tail-pads a batch,
   and supplies a frame mask to masked pooling.

Stage 1 and stage 2 are trained separately. This repository does **not** train
the CTC model; fine-tune and export it with WeNet. The feature stage retains
every valid best candidate in a ragged bundle, plus its score, margin, keyword
winner, crop boundaries, and (for positives) the expected keyword ID. It also
compares the selected keyword against the strongest non-keyword CTC sequence
found by prefix beam search on exactly the same candidate frames.
`[stage1_report]` writes threshold-by-keyword tables before any filtering.
Each row has one selected stage-1 keyword: the highest-scoring keyword for
that candidate. Positive tables give each keyword's FR using only that
keyword's expected examples as its denominator. Stage 1 is treated as a binary
candidate detector: if any selected keyword passes, the positive clip is a hit
even when the winning keyword differs from the expected keyword. Negative
tables give total `FA/h` and total FA rate over all input clips, followed by
per-winning-keyword false-accept counts and `FA/h`; a negative clip contributes
to at most one keyword. Positive FR includes examples where no complete CTC
alignment was found or whose selected candidate is below threshold. The
`[train] structure = ctc_wac` step applies each wake word's manual stage-1
threshold later, immediately before training the WAC model. That means you can
choose a threshold and retrain stage 2 without rerunning the expensive CTC
model. The default gate is `stage1_gate_score = normalized_ctc_score`, the
older negative CTC-score scale stored in `top_score`. Thresholds must use the
same scale as the selected gate. For the `[0, 1]` keyword-versus-filler values
reported below, set `stage1_gate_score = normalized_confidence` in `[train]`.

To teach an existing Stage-2 model additional hard negatives without starting
again from random weights, set `initialize_from_model` to its PyTorch `.pt`
artifact (not the exported `.onnx` file):

```ini
[train]
structure = ctc_wac
initialize_from_model = /path/to/existing_stage2.pt
```

The keyword IDs/order, feature dimension, and WAC architecture must match.
Fine-tuning keeps the source model's score normalization and evaluates the
unchanged source model as the step-zero validation baseline. If every later
checkpoint has worse validation loss, the exported model retains the source
weights. Normal checkpoint `resume` still takes priority when a compatible
checkpoint for the current run exists.

### Keyword-versus-filler CTC confidence

For each selected candidate segment, the feature stage runs CTC prefix beam
search and chooses the highest-scoring collapsed token sequence that is not
exactly the selected keyword. It recomputes both the keyword and this
competitor with the standard CTC **forward** algorithm, which sums all legal
CTC paths. The bundle row and adjacent NPY files store `keyword_score`,
`filler_score`, `raw_score`, `normalized_raw_score`, `confidence`,
`normalized_confidence`, and `segment_length`. `confidence` is
`sigmoid(keyword_score - filler_score)`; `normalized_confidence` applies the
same sigmoid after dividing the score difference by the selected segment's
encoder-frame length. The old `top_score` / `normalized_ctc_score` remains
unchanged for comparison and is still the score used by the existing stage-2
model.

The Stage-2 gate can independently choose one of these stored scalar values:
`normalized_ctc_score` (default), `confidence`, or
`normalized_confidence`. This gate decides which Stage-1 candidates enter
Stage-2 training; the Stage-2 model itself continues to receive the frozen
normalized CTC score, margin, and winner one-hot context.

The feature options below control the approximate competitor search. Their
defaults are shown; `competitor_token_prune` keeps only the best non-blank
tokens at each encoder frame, which keeps large negative-corpus runs practical.

```ini
[feature.negative_dev]
competitor_beam_size = 16
competitor_token_prune = 8
```

The report has three separate threshold ranges, because the old log score and
the two confidence values have different scales:

```ini
[stage1_report]
threshold_start = -5.0
threshold_stop = 0.0
threshold_step = 0.05
confidence_threshold_start = 0.0
confidence_threshold_stop = 1.0
confidence_threshold_step = 0.05
normalized_confidence_threshold_start = 0.0
normalized_confidence_threshold_stop = 1.0
normalized_confidence_threshold_step = 0.05
```

### Experimental contextual BPE WUW beam

For a BPE CTC model, a feature block can use an opt-in contextual beam rather
than forcing every input through every keyword spelling. Keep the existing
`keyword_tokens` JSON for the model-facing token IDs and supply a TSV with one
human-readable keyword and completion bonus per line:

```text
# display_text<TAB>bonus
hey siri	5.0
next track	6.0
```

```ini
[feature.positive_dev]
extractor = wenet_ctc_wac
candidate_search = contextual_wuw_beam
wuw_bias_tsv = /path/to/wuw_bias.tsv
contextual_beam_size = 16
contextual_token_prune = 8
contextual_lookahead = yes
```

The decoder retains ordinary BPE paths during normal beam search. A completed
keyword becomes a virtual `<WUW:keyword_id>` only inside the decoder and gains
its TSV bonus once. Partial keyword prefixes use the best reachable bonus only
for beam pruning; they receive no final bonus unless the full spelling is
completed. At finalization, paths with no completed WUW are discarded. If none
remain, the input has confidence `0`, creates no Stage-2 crop, and is counted
as a Stage-1 rejection.

The contextual Stage-1 gate uses normalized confidence:
`sigmoid((boosted WUW score - best non-WUW score) / CTC frame count)`. Set
per-keyword thresholds in the keyword JSON on the `[0, 1]` scale after reading
the contextual section of `stage1_report`. The report uses
`normalized_confidence_threshold_*` for this primary table.

No BPE text table is needed for ordinary speech: all non-keyword tokens are
treated as garbage semantically while retaining their integer AM IDs during
search. As a result, this first experiment does not independently verify a
right word boundary; an exact keyword token sequence inside a longer BPE word
can still match and should be checked in Stage-2 evaluation.

For CTC training, use `ctc_context = yes` and a base `window_seconds` of about
`2.56`. Set `window_count = 2` when a wake word plus natural surrounding speech
needs up to `5.12` seconds. With
`leading_context_seconds_range = [1.0, 2.0]`, a short source is **not** padded
to the 5.12-second maximum: the pipeline samples 1--2 seconds of leading
background context, appends the source, and mixes the whole variable-length
signal with a same-length background crop. If the source is near the maximum,
that context range is automatically capped to what still fits. `filter` drops
only sources longer than the maximum; `start`, `end`, and `random` crop such
sources to the maximum and add no context. The supplied examples filter long
positives and randomly crop long negatives.

Use the same `window_seconds` and `window_count` in every CTC feature and test
section. Feature generation scores the continuous `[T, V]` output with that
bounded horizon and records the best candidate's score, margin, and exact
non-blank encoder crop. Evaluation also keeps the stage-1 ONNX cache
continuous; it does not run independent overlapping audio windows. It simply
allows paths beginning in the most recent configured horizon, then passes
newly rising stage-1 candidates to stage 2. Recalibrate stage-1 and final
thresholds after changing the horizon, because a longer search range changes
the maximum-score distribution on negative audio.

### Optional CTC token-alignment debug log

Normal CTC feature generation does not backtrack through each candidate path,
so it keeps the existing throughput and artifact size. To inspect every input
while debugging, enable the following option in a `wenet_ctc_wac` feature
block:

```ini
[feature.negative_dev]
debug_alignment = yes
```

This writes `<output_file stem>.debug.jsonl` next to the feature bundle. It
contains one row per input with its best stage-1 candidate and, for each token,
the token ID, inclusive encoder-frame range, assigned-frame count, raw log
score, and `normalized_score`. A token's `normalized_score` is the mean CTC
log-probability across the frames assigned to that token, so tokens that span
different numbers of frames remain comparable. Rows without a complete CTC
candidate are retained with an explanatory status. Set `debug_alignment = no`
or omit it for normal runs; it is disabled by default and its backtracking work
is not performed. Slurm feature runs merge the same debug rows into one log.

The intended first-run workflow is:

```bash
"$PYTHON" "$PIPELINE" run --config "$CONFIG" --to stage1_report
# Inspect stage1_report/candidates.md, then set each keyword's threshold.
"$PYTHON" "$PIPELINE" run --config "$CONFIG" --from train
```

Start with
[local_ctc_wuw.ini.example.conf](../examples/local_ctc_wuw.ini.example.conf)
or [slurm_ctc_wuw.ini.example.conf](../examples/slurm_ctc_wuw.ini.example.conf),
[wenet_ctc_stage1_contract.example.json](../examples/wenet_ctc_stage1_contract.example.json),
the stable [wenet_ctc_keyword_tokens.example.json](../examples/wenet_ctc_keyword_tokens.example.json),
and [wenet_ctc_keywords.example.json](../examples/wenet_ctc_keywords.example.json).

Export stage 1 with the WUW exporter. It creates the FP32 model, INT8 model,
and matching contract together:

```bash
python wenet_export/export-onnx-streaming.py \
  --checkpoint /path/to/wenet/final.pt \
  --config /path/to/wenet/train.yaml \
  --output-dir /path/to/stage1-export \
  --output-prefix stage1-wuw \
  --chunk-size 16 \
  --left-chunks 4 \
  --token-file /path/to/token.txt
```

The generated `stage1-wuw.contract.json` is a small description of the ONNX
interface. It tells the pipeline which tensor is the fbank input, encoder
feature, CTC output, attention mask, and streaming cache. It also records the
fbank and overlapping-window settings. Normally, do not write or edit this
file yourself; point the pipeline INI at the file created by the exporter.
The example contract is only there to make the format easy to inspect.
For WeNet models, `fbank.waveform_scale` is `32768`: WeNet converts normalized
floating-point PCM back to signed-16-bit amplitude before Kaldi fbank. The
pipeline defaults to this value for older contracts that do not record it.

The code does not assume BPE or phonemes. Put the correct `token_ids` for each
wake word in the keyword JSON. For a BPE/character CTC model, use its
BPE/character IDs. For a phoneme CTC model, use phone IDs.

The example deliberately has two keyword files. Feature blocks use
`keyword_tokens`, which contains only the fixed ids/text/token IDs. Train and
test blocks use `keywords`, which adds the manual threshold for each wake
word. Keep their ids and token IDs identical. With this split, changing only a
threshold invalidates train/export/test, not the expensive feature stage. A
feature block may instead use `keywords` directly for a simpler first setup,
but then a threshold edit will make the normal pipeline cache re-run feature
generation too.

For a CTC-WAC `[testing.*]` block, set
`stage1_gate_score = normalized_confidence` when the keyword JSON contains
the `[0, 1]` normalized keyword-versus-filler thresholds used for Stage-2
mining. The evaluator then calculates that same confidence on each completed
candidate segment before passing it to Stage 2. The optional
`ctc_proposal_score_floor` can reduce beam-comparison work, but it is disabled
by default because it uses a different raw CTC score domain and can otherwise
silently reject candidates before the configured confidence gate. Set it to a
finite number only after measuring its recall impact; use `none` to disable
it. The default `stage1_gate_score = normalized_ctc_score` is retained only
for compatibility with older keyword files whose thresholds are negative
normalized CTC log scores. The `[testing.*] threshold_range` always sweeps the final Stage-2
classifier probability, independently of the Stage-1 gate.

If you use some other ONNX exporter, you must create its contract manually.
The example is a template, not a universal WeNet interface. For example, a
cache entry looks like this:

```json
{
  "input": "att_cache",
  "output": "new_att_cache",
  "shape": [12, 4, 64, 128],
  "dtype": "float32"
}
```

The `shape` is the initial cache shape used by that specific graph. If an input
such as `required_cache_size` is not in the graph, do not put it in
`constant_inputs`. Run one short feature block first: it will report a clear
missing input/output name rather than silently using the wrong tensor.

An `onnx.int8` stage-1 model is fine if it is a normal QDQ-style ONNX model:
the external fbank input, `encoder` output, and `ctc_log_probs` output must be
floating tensors. Do not expose only an int8 encoder tensor. The stage-2
classifier needs the real dequantized encoder values, and feature extraction
and evaluation should use the same stage-1 ONNX file.

The CTC-WAC evaluator runs both ONNX files. It uses the same bounded CTC
boundary scorer and candidate crop as feature generation, then supplies an
all-one mask for the unpadded crop. It reports the effective rolling CTC
horizon, stage-1 candidate rate, final false-accept/false-reject threshold
sweep, candidate counts by winning wake word, and real-time factor.

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

The JSONL includes `run_start`, `phase_start`, `train_step`, aggregate
`validation`, one `validation_set` event per labeled dev block, `checkpoint`,
and `run_complete` events. Resumed runs append a new session ID to the same
file. Threshold-based FA/FR metrics are intentionally reserved for
`testing.*`, where the configured threshold range is swept.

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

New output uses exactly one canonical `path` field. Readers also accept
`audio_filepath` (the NeMo/GigaSpeech manifest convention) and the legacy
keys `audiofile_path`, `audio_file`, `audio_path`, `file`, and `filename` so
older source datasets remain usable.

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
