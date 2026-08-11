#!/usr/bin/env bash
set -euo pipefail
model=$1; build=$2; output=$3; config=${4:-/mnt/users/jinyang_wang/WUW/test/expts8_full/expts8_full.ini}; shards=${5:-50}
sdk_dir=$(cd "$(dirname "$0")/.." && pwd); mkdir -p "$output/logs"
python3 "$sdk_dir/scripts/make_expts8_eval_plan.py" --config "$config" --output "$output/tasks.jsonl" --shards "$shards" >/dev/null
tasks=$(wc -l < "$output/tasks.jsonl"); job=$(sbatch --parsable --partition=cpu --cpus-per-task=1 --mem=2G --time=02:00:00 --array="0-$((tasks-1))%50" --output="$output/logs/%A_%a.out" --error="$output/logs/%A_%a.err" "$sdk_dir/scripts/expts8_sbatch_entry.sh" "$sdk_dir/scripts/expts8_array_worker.sh" "$output/tasks.jsonl" "$model" "$build" "$output")
echo "$job" > "$output/job_id"; echo "submitted $job ($tasks tasks)"
while squeue -h -j "$job" | grep -q .; do sleep 20; done
if sacct -n -X -j "$job" --format=State | grep -Eq 'FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY'; then echo "array job failed" >&2; exit 1; fi
python3 "$sdk_dir/scripts/analyze_expts8_eval.py" --shards-root "$output/shards" --keywords "$model/keywords.json" --output "$output/merged"
