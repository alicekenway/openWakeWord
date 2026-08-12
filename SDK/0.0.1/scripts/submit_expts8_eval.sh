#!/usr/bin/env bash
set -euo pipefail
model=$1; build=$2; output=$3; config=${4:-/mnt/users/jinyang_wang/WUW/test/expts8_full/config.resolved.ini}; workers_per_test=${5:-100}
sdk_dir=$(cd "$(dirname "$0")/.." && pwd); mkdir -p "$output/logs"
python3 "$sdk_dir/scripts/make_expts8_eval_plan.py" --config "$config" --output "$output/tasks.jsonl" --groups-output "$output/groups.jsonl" --workers-per-test "$workers_per_test" >/dev/null
tasks=$(wc -l < "$output/tasks.jsonl")
if [[ "$tasks" -eq 0 ]]; then echo "evaluation plan is empty; check config path: $config" >&2; exit 2; fi
> "$output/job_ids"
while IFS=$'\t' read -r dataset start end workers records; do
  job=$(sbatch --parsable --partition=cpu --cpus-per-task=1 --mem=2G --time=02:00:00 --array="$start-$end%$workers" --output="$output/logs/%A_%a.out" --error="$output/logs/%A_%a.err" "$sdk_dir/scripts/expts8_sbatch_entry.sh" "$sdk_dir/scripts/expts8_array_worker.sh" "$output/tasks.jsonl" "$model" "$build" "$output")
  printf '%s\t%s\n' "$dataset" "$job" >> "$output/job_ids"
  echo "submitted $dataset: job $job ($workers workers for $records audio files)"
  while squeue -h -j "$job" | grep -q .; do sleep 20; done
  if sacct -n -X -j "$job" --format=State | grep -Eq 'FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY'; then echo "$dataset array job failed: $job" >&2; exit 1; fi
done < <(python3 -c 'import json,sys
for line in open(sys.argv[1]):
 x=json.loads(line);print(x["dataset"],x["start"],x["end"],x["workers"],x["records"],sep="\t")' "$output/groups.jsonl")
completed=$(find "$output/shards" -name 'part-*.jsonl' | wc -l)
if [[ "$completed" -ne "$tasks" ]]; then echo "expected $tasks shard files, found $completed" >&2; exit 1; fi
python3 "$sdk_dir/scripts/analyze_expts8_eval.py" --shards-root "$output/shards" --keywords "$model/keywords.json" --model-dir "$model" --output "$output/merged"
