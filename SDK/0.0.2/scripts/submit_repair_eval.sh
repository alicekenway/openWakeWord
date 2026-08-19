#!/usr/bin/env bash
set -euo pipefail
model=$1; build=$2; output=$3
sdk_dir=$(cd "$(dirname "$0")/.." && pwd)
repairs=$(python3 "$sdk_dir/scripts/make_repair_plan.py" --tasks "$output/tasks.jsonl" --shards-root "$output/shards" --output "$output/repair_tasks.jsonl")
if [[ "$repairs" -gt 0 ]]; then
  job=$(sbatch --parsable --partition=cpu --cpus-per-task=1 --mem=2G --time=02:00:00 --array="0-$((repairs-1))%50" --output="$output/logs/repair_%A_%a.out" --error="$output/logs/repair_%A_%a.err" "$sdk_dir/scripts/expts8_sbatch_entry.sh" "$sdk_dir/scripts/expts8_array_worker.sh" "$output/repair_tasks.jsonl" "$model" "$build" "$output")
  echo "$job" > "$output/repair_job_id"; echo "submitted repair $job ($repairs tasks)"
  while squeue -h -j "$job" | grep -q .; do sleep 20; done
  if sacct -n -X -j "$job" --format=State | grep -Eq 'FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY'; then echo "repair array failed" >&2; exit 1; fi
fi
errors=$(grep -l '"error"' "$output"/shards/*/part-*.jsonl | wc -l || true)
if [[ "$errors" -ne 0 ]]; then echo "$errors shard files still contain errors" >&2; exit 1; fi
python3 "$sdk_dir/scripts/analyze_expts8_eval.py" --shards-root "$output/shards" --keywords "$model/keywords.json" --output "$output/merged"
