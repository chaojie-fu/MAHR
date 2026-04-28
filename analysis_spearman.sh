#!/bin/bash

root_dir="/home/cjfu/Retarget_SOTA/MAHR"
cd ${root_dir}

# ensure log directory exists for the SQLite DB
mkdir -p "${root_dir}/log"

# Use the project virtualenv Python explicitly so each job runs in the right interpreter
PY="/home/cjfu/Envs/MAHR/bin/python"

${PY} ${root_dir}/spatial_relationship_preservation_across_robots_motions_methods.py \
    --mode analysis \
    --db_path ${root_dir}/log/metrics.sqlite \
    --analysis_db_path ${root_dir}/log/analysis.sqlite