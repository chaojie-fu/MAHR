#!/bin/bash

source /home/cjfu/Envs/MAHR/bin/activate

robot_urdf_root="/home/cjfu/Retarget_SOTA/MAHR/misc/visualization/compare_different_robots/robots/parsed_humanoid"

motion_root="/home/cjfu/Retarget_SOTA/MAHR/retargeted_motions/ParameterizedHumanoid"

# iterate over robot size specifications
# find all sub-directories in motion_root
robot_names=()
for dir in "${motion_root}"/*/; do
  if [ -d "${dir}" ]; then
    robot_name=$(basename "${dir}")
    robot_names+=("${robot_name}")
  fi
done

motions=(
  "80_43_poses"
  "90_28_poses"
  "Extended_1_poses"
  "G4-spinning_back_kick_poses"
  "82_05_poses"
  "A10-lie_to_crouch_poses"
  "Extended_3_poses"
  "G7-capoera_poses"
)

for robot_name in "${robot_names[@]}"; do
  echo "Processing robot: ${robot_name}"
  urdf_file="${robot_urdf_root}/${robot_name}.urdf"

  for motion in "${motions[@]}"; do
    echo "  Processing motion: ${motion}"
    json_dir=${motion_root}/${robot_name}/${motion}

    json_file="${json_dir}/${motion}_gd_v2.json"
    # iterate over all json files in motion_path
    if [ -f "${json_file}" ]; then
      echo "    Visualizing ${json_file} for robot ${robot_name}"

      # only run if output video does not exist
      output_video="${json_dir}/$(basename "${json_file%.*}")_visualization.mp4"
      if [ -f "${output_video}" ]; then
        echo "    Visualization already exists, skipping..."
        continue
      fi

      python visualize_retargeted_motion.py \
        --trajectory \
        "${json_file}" \
        --urdf \
        "${urdf_file}" \
        --no-loop \
        --no-viewer \
        --record \
        "${output_video}"
    fi
  done
done