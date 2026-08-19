#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "usage: $0 SOURCE_DIR ORT_ROOT MODEL_DIR OUTPUT [portable|avx2] [ORT_VERSION]" >&2
  exit 2
fi
source_dir=$1; ort_root=$2; model_dir=$3; output=$4; variant=${5:-portable}
ort_version=${6:-$(<"$ort_root/VERSION_NUMBER")}
build_dir="${output}.build"
if [[ -e "$output" || -e "$build_dir" ]]; then
  echo "output or build directory already exists: $output" >&2
  exit 2
fi
case "$ort_version" in 1.16.3|1.25.1) ;; *) echo "unsupported ORT version: $ort_version" >&2; exit 2;; esac
case "$variant" in portable) cpu_flags=();; avx2) cpu_flags=(-mavx2 -mfma);; *) echo "variant must be portable or avx2" >&2; exit 2;; esac
cmake -S "$source_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG ${cpu_flags[*]}" \
  -DWUWSDK_ONNXRUNTIME_ROOT="$ort_root" \
  -DWUWSDK_REQUIRED_ORT_VERSION="$ort_version"
cmake --build "$build_dir" -j
ctest --test-dir "$build_dir" --output-on-failure
cmake --install "$build_dir" --prefix "$output"
mkdir -p "$output/lib" "$output/model" "$output/licenses"
cp -a "$ort_root/lib/libonnxruntime.so.$ort_version" "$output/lib/"
ln -sfn "libonnxruntime.so.$ort_version" "$output/lib/libonnxruntime.so.1"
ln -sfn libonnxruntime.so.1 "$output/lib/libonnxruntime.so"
if [[ -f "$ort_root/lib/libonnxruntime_providers_shared.so" ]]; then
  cp -a "$ort_root/lib/libonnxruntime_providers_shared.so" "$output/lib/"
fi
cp -a "$model_dir"/. "$output/model/"
cp "$source_dir/README.md" "$output/README.md"
[[ -f "$source_dir/SERVER_TEST_REPORT.md" ]] && cp "$source_dir/SERVER_TEST_REPORT.md" "$output/SERVER_TEST_REPORT.md"
for file in LICENSE ThirdPartyNotices.txt; do [[ -f "$ort_root/$file" ]] && cp "$ort_root/$file" "$output/licenses/ONNXRuntime_$file"; done
(cd "$output" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
tar -czf "$output.tar.gz" -C "$(dirname "$output")" "$(basename "$output")"
echo "$output.tar.gz"
