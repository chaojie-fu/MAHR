#!/bin/bash

# Activate environment if available (non-fatal if missing)
if [[ -f /home/cjfu/Envs/MAHR/bin/activate ]]; then
  source /home/cjfu/Envs/MAHR/bin/activate
else
  echo "[INFO] Virtualenv /home/cjfu/Envs/MAHR not found; continuing without activation." >&2
fi

which python || true

python ./comparing_optimization_methods_continuous_lbfgsb.py \
  --bvh "mahr/rsc/amass_bvh/ACCAD/G4-spinning_back_kick_poses.bvh" \
  --norm-pose t_pose \
  --norm-pose-dir "mahr/rsc/amass_bvh/norm_pose" \
  --no-filter-edges \
  --db-path "log/optimizations_continuous.sqlite"
