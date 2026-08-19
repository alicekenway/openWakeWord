# Server validation report — SDK 0.0.1

Date: 2026-08-11  
Server output: `/mnt/users/jinyang_wang/WUW/SDK_test/0.0.1`  
Full merged report: `full_expts8/merged/summary.json`

## Build and functional checks

- Built on server `u` with GCC 9.4.0 and ONNX Runtime 1.21.0.
- Portable and AVX2 packages both load the checksummed expts8 model bundle.
- Both packages detect the same `hey_siri` event on the positive smoke WAV.
- Local CMake build and three unit tests pass.
- Full evaluation completed all 1,100 Slurm array tasks, merged 10,543 records across 22 conditions, and ended with zero record errors.
- The evaluation adapter uses the expts8 5.12-second window and 2.56-second stride. It downmixes PCM16 multi-channel evaluation files; the public SDK input remains mono PCM16 at 16 kHz.

## Performance

- Maximum measured per-dataset RTF: 0.0709.
- One-hour negative smoke RTF: 0.0551.
- One-hour negative smoke peak RSS: 80.0 MB.
- These pass the provisional server targets of RTF <= 0.25 and RSS <= 128 MB.
- This is not target-car sign-off. The release was built against the server's glibc/toolchain, not the car sysroot, and must be rebuilt and measured on the actual x86 car machine.

## Detection results at stage-2 threshold 0.7

- Main positive clean FRR: 2.58%.
- Main positive FRR range: 2.58%–12.42% across clean/FIR/car/ordinary conditions.
- Sampled Hey Siri FRR range: 3.77%–15.09%.
- Negative CosyVoice clean: 123.74 accepted events/hour.
- Negative CosyVoice FIR: 132.69 accepted events/hour.
- Negative GigaSpeech clean/FIR/ordinary: 0.60/0.92/0.64 accepted events/hour.
- Pure ordinary background: 1.31 accepted events/hour.
- Car-related and car-related+FIR negative conditions produced zero accepted events in this run.

The runtime meets the provisional compute and memory goals, but the current model/threshold operating point is not industrially acceptable because several negative conditions have excessive false accepts. Do not present SDK 0.0.1 as a production-qualified WUW model. The next model iteration should calibrate the global stage-2 threshold and/or retrain with the clean/FIR false-positive clusters, then rerun this same fixed suite.

## Python-reference comparison

For the first one-hour negative CosyVoice clean recording, Python expts8 had 150 candidates with stage-2 score >= 0.7 and the C++ streaming SDK emitted 141 events. At 10 ms start-time rounding, 123 unique keyword/start pairs matched. The differences are concentrated near the threshold and in streaming rising-edge versus offline best-window selection. This is good behavioral agreement, but it is not bit-exact frontend/candidate parity.
