#!/bin/bash

# A batch run script to compare multiple robots, motions, and methods
robots=(
    "AgibotX1"
    "BerkeleyHumanoid"
    "BostonDynamicsAtlasHD"
    "FourierN1"
    "FourierGR1"
    "UnitreeG1"
    "UnitreeH1"
    "iCub"
)

motions=(
    "80_43_poses"
    "82_05_poses"
    "90_28_poses"
    "A10-lie_to_crouch_poses"
    "Extended_1_poses"
    "Extended_3_poses"
    "G4-spinning_back_kick_poses"
    "G7-capoera_poses"
)

methods=(
#    "mahr"
#    "gmr"
#    "phc"
    "ik"
    "jm"
)

root_dir="/home/cjfu/Retarget_SOTA/MAHR"
cd ${root_dir}

# ensure log directory exists for the SQLite DB
mkdir -p "${root_dir}/log"

# Use the project virtualenv Python explicitly so each job runs in the right interpreter
PY="/home/cjfu/Envs/MAHR/bin/python"

# Run one job per combination using GNU Parallel's argument expansion
parallel --jobs 64 \
    echo "Processing Robot: {1}, Motion: {2}, Method: {3}" '&&' ${PY} ${root_dir}/spatial_relationship_preservation_across_robots_motions_methods.py \
        --robot_name {1} \
        --motion_name {2} \
        --method_name {3} \
        --write_db \
        --db_path ${root_dir}/log/metrics.sqlite \
    ::: "${robots[@]}" ::: "${motions[@]}" ::: "${methods[@]}"
