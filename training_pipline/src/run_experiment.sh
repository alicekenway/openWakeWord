#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/alicekenway/miniconda3/envs/openwake/bin/python}
PIPELINE=${PIPELINE:-/home/alicekenway/Dev/project/WUW/openWakeWord/training_pipline/src/wuw_pipeline.py}
CONFIG=${CONFIG:-}

if [[ -n "${CONFIG}" && "${CONFIG}" == *.ini ]]; then
  "${PYTHON}" "${PIPELINE}" run --config "${CONFIG}" "$@"
elif [[ -n "${CONFIG}" ]]; then
  "${PYTHON}" "${PIPELINE}" run-experiment --config "${CONFIG}" "$@"
else
  "${PYTHON}" "${PIPELINE}" run-experiment "$@"
fi
