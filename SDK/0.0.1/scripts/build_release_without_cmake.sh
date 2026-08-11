#!/usr/bin/env bash
set -euo pipefail
source_dir=$1; ort_root=$2; model_dir=$3; output=$4; variant=${5:-portable}
if [[ -e "$output" ]]; then echo "output already exists: $output" >&2; exit 2; fi
case "$variant" in portable) cpu_flags=();; avx2) cpu_flags=(-mavx2 -mfma);; *) echo "variant must be portable or avx2" >&2; exit 2;; esac
mkdir -p "$output/bin" "$output/lib" "$output/include" "$output/model" "$output/licenses"
g++ -std=c++17 -O3 -DNDEBUG "${cpu_flags[@]}" -fPIC -shared \
  -I"$source_dir/include" -I"$source_dir/src" -I"$ort_root/include" \
  "$source_dir"/src/*.cc -L"$ort_root/lib" -lonnxruntime \
  -Wl,-rpath,'$ORIGIN' -o "$output/lib/libwuw_sdk.so.0.0.1"
ln -s libwuw_sdk.so.0.0.1 "$output/lib/libwuw_sdk.so.1"; ln -s libwuw_sdk.so.1 "$output/lib/libwuw_sdk.so"
cp -a "$ort_root"/lib/libonnxruntime.so* "$ort_root"/lib/libonnxruntime_providers_shared.so "$output/lib/"
for tool in wuw_inspect wuw_stream_wav wuw_eval_manifest wuw_merge_results; do
  g++ -std=c++17 -O3 -DNDEBUG "${cpu_flags[@]}" -I"$source_dir/include" -I"$source_dir/tools" \
    "$source_dir/tools/$tool.cc" -L"$output/lib" -lwuw_sdk -lonnxruntime \
    -Wl,-rpath,'$ORIGIN/../lib' -o "$output/bin/$tool"
done
cp -a "$source_dir/include/wuw_sdk" "$output/include/"; cp -a "$model_dir"/. "$output/model/"
cp "$source_dir/README.md" "$output/README.md"
for file in LICENSE ThirdPartyNotices.txt; do [[ -f "$ort_root/$file" ]] && cp "$ort_root/$file" "$output/licenses/ONNXRuntime_$file"; done
(cd "$output" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
tar -czf "$output.tar.gz" -C "$(dirname "$output")" "$(basename "$output")"
echo "$output.tar.gz"
