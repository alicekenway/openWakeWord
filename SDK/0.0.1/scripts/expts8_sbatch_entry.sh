#!/usr/bin/env bash
set -euo pipefail
srun --cpu-bind=none --ntasks=1 "$@"
