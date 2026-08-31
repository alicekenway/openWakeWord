#!/usr/bin/env bash
set -euo pipefail
exec /mnt/users/jinyang_wang/WUW/.wuw/bin/python /mnt/users/jinyang_wang/WUW/openWakeWord/training_pipline/src/wuw_pipeline.py run --config /mnt/users/jinyang_wang/WUW/train/features/batch_11/batch_11.ini "$@"
