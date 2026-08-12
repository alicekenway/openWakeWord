#!/usr/bin/env bash
set -euo pipefail
plan=$1; model=$2; build=$3; output=$4
task=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
readarray -t fields < <(python3 -c 'import json,sys;x=json.loads(open(sys.argv[1]).readlines()[int(sys.argv[2])]);print(x["dataset"]);print(x["manifest"]);print(x["expected_label"]);print(x["shard"]);print(x["shards"]);print(x["window_samples"]);print(x["stride_samples"])' "$plan" "$task")
dataset=${fields[0]}; manifest=${fields[1]}; label=${fields[2]}; shard=${fields[3]}; shards=${fields[4]}; window_samples=${fields[5]}; stride_samples=${fields[6]}
dataset_dir="$output/shards/$dataset"; mkdir -p "$dataset_dir"
runtime_root=$(dirname "$build"); export LD_LIBRARY_PATH="$build:$runtime_root/onnxruntime/lib:${LD_LIBRARY_PATH:-}"
"$build/wuw_eval_manifest" "$model" "$manifest" "$dataset_dir/part-$(printf '%04d' "$shard").jsonl" "$shard" "$shards" "$window_samples" "$stride_samples"
printf '%s\n' "$label" > "$dataset_dir/expected_label"
