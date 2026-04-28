#!/bin/bash

source /home/cjfu/Envs/MAHR/bin/activate

# a dict storing the robot type and its corresponding urdf path
declare -A robot_urdfs

terrain_idx=(
  "0"
  "1"
  "2"
  "3"
)

motion_names=(
  "A10-lie_to_crouch_poses"
  "Extended_1_poses"
)

for terrain in "${terrain_idx[@]}"; do
  for motion_name in "${motion_names[@]}"; do
    # iterate over json files in the motion path
      echo "Visualizing ${motion_name} for terrain index ${terrain}..."
      # only run if "${motion_path}/$(basename "${json_file%.*}")_visualization.mp4" does not exist
      output_video="${motion_path}/$(basename "${json_file%.*}")_visualization.mp4"
      if [ -f "${output_video}" ]; then
        echo "Visualization already exists, skipping..."
        continue
      fi
      python visualize_retargeted_motion.py \
        --trajectory \
        "retargeted_motions/MAHR_Terrain/GR1/Terrain${terrain}/${motion_name}_gd_v2.json" \
        --urdf \
        "mahr/rsc/assets/urdf/Fourier_GR1/GR1.urdf" \
        --no-loop \
        --no-viewer \
        --record \
        "retargeted_motions/MAHR_Terrain/GR1/Terrain${terrain}/${motion_name}_gd_v2_visualization.mp4" \
        --height-map \
        "height_map/height_map${terrain}.png"
  done
done