#!/bin/bash

#
# Download the GDPVal dataset from HuggingFace and convert it to the benchmark
# JSONL format consumed by ``ng_e2e_collect_rollouts``.
#
# Output: benchmarks/gdpval/data/gdpval_benchmark.jsonl (220 tasks).
#
# Requires:
#   - Active nemo-gym venv (uv sync --extra dev)
#   - HF_TOKEN env var if the dataset requires authentication

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

ng_prepare_benchmark "+config_paths=[benchmarks/gdpval/config.yaml]"

echo "Done. Output: benchmarks/gdpval/data/gdpval_benchmark.jsonl"
