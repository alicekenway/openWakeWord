# WUW Data Processing Tools

Small tools for normalizing source datasets into the shared metadata JSONL
format:

```json
{"path": "wav/000000000.wav", "text": "Home pay"}
```

New manifests always contain exactly one audio path field named `path`.
Readers still accept legacy aliases such as `audiofile_path`, but any tool that
creates new audio (conversion, VAD trimming, or augmentation) replaces the old
audio reference instead of retaining source-path copies.

## Common Voice TSV To JSONL

Convert all rows:

```bash
python /home/alicekenway/Dev/project/WUW/openWakeWord/data_processing_tools/cv_tsv_to_jsonl.py \
  --tsv /mnt/d/wuw_data/eng/cv-corpus-26.0-2026-06-12/en/train.tsv \
  --output-jsonl /home/alicekenway/Dev/project/WUW/training/expts2/common_voice_train_metadata.jsonl
```

Sample a fixed number of rows:

```bash
python /home/alicekenway/Dev/project/WUW/openWakeWord/data_processing_tools/cv_tsv_to_jsonl.py \
  --tsv /mnt/d/wuw_data/eng/cv-corpus-26.0-2026-06-12/en/train.tsv \
  --output-jsonl /home/alicekenway/Dev/project/WUW/training/expts2/common_voice_train_200k_metadata.jsonl \
  --sample-size 200000 \
  --seed 1337
```

By default, the output `path` is `clips/<filename from TSV>`, so use
the Common Voice `en` directory as `audio_base_path` in training configs:

```json
{
  "jsonl_path": "/home/alicekenway/Dev/project/WUW/training/expts2/common_voice_train_200k_metadata.jsonl",
  "audio_base_path": "/mnt/d/wuw_data/eng/cv-corpus-26.0-2026-06-12/en"
}
```

If you want the JSONL to contain only the raw filename from the TSV, pass:

```bash
--audio-prefix ''
```

## WAV Directory To JSONL

Create metadata JSONL from a directory of background WAV files:

```bash
python /home/alicekenway/Dev/project/WUW/openWakeWord/data_processing_tools/wav_dir_to_jsonl.py \
  --wav-dir /mnt/d/wuw_data/background \
  --output-jsonl /home/alicekenway/Dev/project/WUW/training/expts2/background_metadata.jsonl
```

Each output row is:

```json
{"path": "fma_sample/000002.wav", "text": ""}
```

By default, paths are relative to `--wav-dir`. In the training config, set
`audio_base_path` to the same directory:

```json
{
  "jsonl_path": "/home/alicekenway/Dev/project/WUW/training/expts2/background_metadata.jsonl",
  "audio_base_path": "/mnt/d/wuw_data/background"
}
```

You can sample a fixed number of files:

```bash
python /home/alicekenway/Dev/project/WUW/openWakeWord/data_processing_tools/wav_dir_to_jsonl.py \
  --wav-dir /mnt/d/wuw_data/background \
  --output-jsonl /home/alicekenway/Dev/project/WUW/training/expts2/background_5000_metadata.jsonl \
  --sample-size 5000 \
  --seed 1337
```

## Group-Safe JSONL Split

Split a metadata JSONL into train/val/test without breaking consecutive groups.
With `--group-size 10`, rows `1-10` are one group, `11-20` are another group,
and only whole groups are shuffled and assigned to outputs.

```bash
python /home/alicekenway/Dev/project/WUW/openWakeWord/data_processing_tools/group_split_jsonl.py \
  --input-jsonl /home/alicekenway/Dev/project/WUW/training/expts2/common_voice_train_200k_metadata.jsonl \
  --output-dir /home/alicekenway/Dev/project/WUW/training/expts2/common_voice_split \
  --set-names train:val:test \
  --group-counts 10000:500:500 \
  --group-size 10 \
  --seed 1337
```

`--group-counts` counts groups, not rows. If `--group-size 10` and the train
count is `10000`, the train JSONL will contain up to `100000` rows.

Use `rest` for the remaining groups:

```bash
--set-names train:val:test --group-counts 10000:500:rest
```

If the requested group counts do not consume all groups, the remaining groups
are written to `unused.jsonl`. A `split_summary.json` is always written in the
output directory.

To write absolute audio paths in the split JSONL files, provide the source
audio base directory:

```bash
python /home/alicekenway/Dev/project/WUW/openWakeWord/data_processing_tools/group_split_jsonl.py \
  --input-jsonl /home/alicekenway/Dev/project/WUW/training/expts2/data/full/background_metadata.jsonl \
  --output-dir /home/alicekenway/Dev/project/WUW/training/expts2/data/split/background \
  --set-names train:dev:test \
  --group-counts rest:300:700 \
  --audio-base-dir /mnt/d/wuw_data/background
```

The tool rewrites legacy audio path fields to one canonical `path`. If
`--audio-base-dir` is supplied, that path is absolute. `--add-path-field` is
retained as a deprecated no-op for older command lines.

## JSONL Audio To WAV

Convert audio referenced by a metadata JSONL file into a `wav/` folder and
write a new JSONL with the audio path replaced by the WAV path. This tool also
resamples input audio to the requested sample rate, so MP3/WAV/other decodable
audio becomes clean 16 kHz mono PCM WAV by default.

```bash
python /home/alicekenway/Dev/project/WUW/openWakeWord/data_processing_tools/jsonl_audio_to_wav.py \
  --input-jsonl /home/alicekenway/Dev/project/WUW/training/expts2/data/negative_cv_train.jsonl \
  --audio-base-dir /mnt/d/wuw_data/eng/cv-corpus-26.0-2026-06-12/en \
  --output-dir /home/alicekenway/Dev/project/WUW/training/expts2/data/negative_cv_train_wav \
  --workers auto
```

Input row:

```json
{"path": "clips/common_voice_en_20379937.mp3", "text": "Senator Nelson is the highest ranking Republican in the Texas Senate."}
```

Output row:

```json
{"path": "wav/00000000_common_voice_en_20379937_abc123def0.wav", "text": "Senator Nelson is the highest ranking Republican in the Texas Senate."}
```

Defaults:

- output WAV directory: `output-dir/wav`
- output JSONL: `output-dir/metadata.jsonl`
- output sample rate: `16000`
- worker count: `auto`, based on CPU count and capped at 16

## Resample JSONL WAV

If your source files are already WAV but may not be 16 kHz, use the resampler
name for the same conversion engine:

```bash
python /home/alicekenway/Dev/project/WUW/openWakeWord/data_processing_tools/resample_jsonl_wav.py \
  --input-jsonl /home/alicekenway/Dev/project/WUW/training/expts2/data/split/positive/train.jsonl \
  --audio-base-dir /mnt/d/wuw_data/eng/wuw_audio \
  --output-dir /home/alicekenway/Dev/project/WUW/training/expts2/data/resampled/positive/train \
  --sample-rate 16000 \
  --workers auto
```

The output JSONL points to `wav/*.wav`, and all output audio is mono PCM WAV at
the selected sample rate.

## Normalize Existing JSONL Paths

This metadata-only command removes duplicate audio path aliases and historical
`source_path` values without touching any WAV files:

```bash
python /home/alicekenway/Dev/project/WUW/openWakeWord/data_processing_tools/normalize_jsonl_audio_paths.py \
  /mnt/d/wuw_data/eng --recursive --relative-to /mnt/d/wuw_data/eng
```

For manifests stored inside an experiment, use absolute paths:

```bash
python /home/alicekenway/Dev/project/WUW/openWakeWord/data_processing_tools/normalize_jsonl_audio_paths.py \
  /home/alicekenway/Dev/project/WUW/training --recursive --absolute \
  --fallback-root /mnt/d/wuw_data/eng
```

## Resample WAV Directory

Resample a plain WAV directory while preserving the relative filename layout:

```bash
python /home/alicekenway/Dev/project/WUW/openWakeWord/data_processing_tools/resample_wav_dir.py \
  --input-dir /mnt/d/wuw_data/background/VISC_Dataset_SON \
  --output-dir /mnt/d/wuw_data/background/VISC_Dataset_SON_16k \
  --sample-rate 16000 \
  --workers auto
```

The output directory contains the resampled WAV files directly, plus
`resample_summary.json`.

For a quick smoke test:

```bash
--limit 5
```
