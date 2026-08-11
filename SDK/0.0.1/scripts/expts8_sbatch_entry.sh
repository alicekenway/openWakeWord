#!/usr/bin/env bash
set -euo pipefail
srun --ntasks=1 --cpus-per-task=1 "$@"
