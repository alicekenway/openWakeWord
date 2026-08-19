# WUW SDK 0.0.1

This directory contains a synchronous C++17 wake-up-word runtime for x86-64 Linux. It accepts explicitly segmented mono PCM16 audio at 16 kHz, runs the streaming WeNet phoneme ONNX encoder, applies per-keyword CTC confidence gates, and runs the second-stage ONNX classifier before returning a detection event.

VAD is intentionally outside this SDK. The owning car application should retain its pre-roll/hangover audio, call `begin_segment`, feed the complete speech segment, and call `end_segment`. This keeps one authoritative VAD and audio bank in the full system and avoids losing wake-word boundaries twice.

## Integration contract

- Callers use the stable C ABI in `include/wuw_sdk/c_api.h`. `wuw.hpp` is a convenience RAII wrapper.
- Input is interleaved-free mono signed 16-bit PCM, exactly 16,000 Hz.
- One `WuwSdkEngine` owns immutable ONNX sessions and configuration. It may create multiple independent streams.
- One `WuwSdkStream` belongs to one logical microphone/audio timeline and must not be called concurrently.
- `first_sample_index` is the caller's absolute sample index. It lets detections map back to the audio bank.
- Read events after `accept_pcm16` and after `end_segment`. The queue holds 64 events and returns `RESOURCE_EXHAUSTED` rather than silently dropping one.
- A bundle-configured maximum segment length (180 seconds in the expts8 bundle) prevents exhausting the encoder's positional range. Normal VAD segments should be far shorter.
- Threshold precedence is bundle defaults, optional JSON override, then explicit API overrides. Values are validated in `[0,1]` when the engine is created.
- There is one global stage-2 threshold. Stage-1 thresholds are per keyword.
- `reset` clears segment state, debounce state, pending events, and statistics.

The engine is safe to keep for the life of the process. Create a stream per mono input channel. Model loading and checksum verification happen once at engine construction.

## Model bundle

The runtime requires these files in one directory:

```
manifest.json
stage1_contract.json
stage1.onnx
stage2.onnx
keywords.json
sdk_defaults.json
SHA256SUMS
```

Create the expts8 bundle with `scripts/package_expts8_model.py`. Checksum verification is enabled by default and should remain enabled in production.

## Build

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DWUWSDK_ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.25.1 \
  -DWUWSDK_REQUIRED_ORT_VERSION=1.25.1
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Release builds support ONNX Runtime 1.16.3 and 1.25.1. Build a separate SDK
package for each version; do not replace the ONNX Runtime library underneath an
already-built `libwuw_sdk.so`.

For a portable build, use ordinary release flags. For a known AVX2 target, add `-DCMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG -mavx2 -mfma'`. Do not ship the AVX2 build to an unknown CPU.

## Minimal C++ use

```cpp
wuw_sdk::Engine engine("model_bundle");
auto stream = engine.CreateStream();
stream.BeginSegment(segment_id, absolute_first_sample);
stream.AcceptPcm16(pcm, sample_count);
stream.EndSegment();
for (const auto& event : stream.ReadEvents()) {
  // event.keyword_id, event.confidence, and absolute sample boundaries
}
```

`tools/wuw_stream_wav` is a runnable example. `tools/wuw_eval_manifest` supports deterministic modulo shards and records the sliding-window index for every accepted event. `scripts/submit_expts8_eval.sh` processes test sets sequentially. For each set it creates up to 100 sbatch-array workers and distributes that set's audio files across them, while each audio is evaluated with 5.12-second windows and a 2.56-second stride. It validates every array before moving to the next set and writes `summary.json` plus `FULL_CONDITION_COMPARISON.md`. Its optional fifth argument changes the workers per test set (default 100). The report uses the single stage-2 threshold in `sdk_defaults.json`; for negative data, FA rate is accepted windows divided by evaluated windows, while FA/hour uses debounced events over source-audio duration.

The preferred interface is the config-driven runner and `experiments/expts8_full_sdk.ini`:

```bash
# Safe preview: validates inputs and prints worker distribution; submits nothing.
python3 scripts/run_sdk_eval.py experiments/expts8_full_sdk.ini plan

# Submit each test set, wait for it, validate its workers, then create the summary.
python3 scripts/run_sdk_eval.py experiments/expts8_full_sdk.ini run

# Inspect recorded Slurm jobs, or regenerate the summary from completed shards.
python3 scripts/run_sdk_eval.py experiments/expts8_full_sdk.ini status
python3 scripts/run_sdk_eval.py experiments/expts8_full_sdk.ini summary
```

On server `u`, invoke these commands inside `srun`. Always choose a fresh `[sdk] output_dir` for a new run; the runner refuses to mix new results with an existing `shards` directory.

## Production checks before release

The acceptance suite must cover frontend parity against `torchaudio.compliance.kaldi.fbank`, stage-1 output/cache parity, CTC trace and candidate parity, stage-2 probability parity, chunk-size invariance, malformed bundles/WAVs, multi-stream stress, and long-run memory stability. The provisional one-core targets are RTF <= 0.25, RSS <= 128 MB, and accepted-event latency <= 800 ms after the final wake-word audio reaches the SDK. Metrics must be measured on the actual car CPU/sysroot; a server result is not a substitute for target-hardware sign-off.
