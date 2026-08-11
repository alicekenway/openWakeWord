#!/usr/bin/env bash
set -euo pipefail
plan=$1; model=$2; build=$3; output=$4
task=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
readarray -t fields < <(python3 -c 'import json,sys;x=json.loads(open(sys.argv[1]).readlines()[int(sys.argv[2])]);print(x["dataset"]);print(x["manifest"]);print(x["expected_label"]);print(x["shard"]);print(x["shards"])' "$plan" "$task")
dataset=${fields[0]}; manifest=${fields[1]}; label=${fields[2]}; shard=${fields[3]}; shards=${fields[4]}
dataset_dir="$output/shards/$dataset"; mkdir -p "$dataset_dir"
"$build/wuw_eval_manifest" "$model" "$manifest" "$dataset_dir/part-$(printf '%04d' "$shard").jsonl" "$shard" "$shards"
printf '%s\n' "$label" > "$dataset_dir/expected_label"
