#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${root_dir}"

# ensure log directory exists for the SQLite DB
mkdir -p "${root_dir}/log"

# Override with MAHR_PYTHON=/path/to/python when a specific environment is needed.
PY="${MAHR_PYTHON:-python}"

${PY} "${SCRIPT_DIR}/spatial_relationship_preservation_across_robots_motions_methods.py" \
    --mode analysis \
    --db_path ${root_dir}/log/metrics.sqlite \
    --analysis_db_path ${root_dir}/log/analysis.sqlite
