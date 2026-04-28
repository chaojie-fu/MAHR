#!/bin/bash

source /home/cjfu/Envs/MAHR/bin/activate

# a dict storing the robot type and its corresponding urdf path
declare -A robot_urdfs
robot_urdfs["agibot_x1"]="mahr/rsc/assets/urdf/Agibot_X1/urdf/x1_stl.urdf"
robot_urdfs["berkeley_humanoid_lite"]="mahr/rsc/assets/urdf/Berkeley_Humanoid_Lite/berkeley_humanoid_lite_stl.urdf"
robot_urdfs["boston_dynamics_atlas_hd"]="mahr/rsc/assets/urdf/Boston_Dynamics_Atlas/urdf/atlas_v4_with_multisense_stl.urdf"
robot_urdfs["fourier_gr1"]="mahr/rsc/assets/urdf/Fourier_GR1/GR1.urdf"
robot_urdfs["fourier_n1"]="mahr/rsc/assets/urdf/Fourier_N1/urdf/N1_raw_stl.urdf"
robot_urdfs["icub"]="mahr/rsc/assets/urdf/iCub/robots/iCubGazeboV2_7/model_stl.urdf"
robot_urdfs["unitree_g1"]="mahr/rsc/assets/urdf/unitree_g1/g1_29dof_stl.urdf"
robot_urdfs["unitree_h1"]="mahr/rsc/assets/urdf/unitree_h1/urdf/h1_stl.urdf"

methods=(
    "GMR"
    "MAHR"
    "MAHR_Wider_Constraint"
    "PHC"
    "IK"
    "JM"
)

# a dict containing the method name and its corresponding available robots
declare -A method_robots
method_robots["GMR"]="berkeley_humanoid_lite fourier_n1 unitree_g1 unitree_h1"
method_robots["MAHR"]="Agibot_X1 Berkeley_Humanoid_Lite Boston_Dynamics_Atlas_HD Fourier_N1 GR1 iCub Unitree_G1 Unitree_H1"
method_robots["MAHR_Wider_Constraint"]="agibot_x1 boston_dynamics_atlas_hd fourier_gr1 gr1 unitree_g1 berkeley_humanoid_lite fourier_n1 icub unitree_h1"
method_robots["PHC"]="Unitree_G1_PROCESSED Unitree_H1_PROCESSED"
method_robots["IK"]="GR1"
method_robots["JM"]="GR1"

# a dict containing mappings from robot dir names to urdf keys
declare -A robot_dir_to_urdf_key
robot_dir_to_urdf_key["Agibot_X1"]="agibot_x1"
robot_dir_to_urdf_key["berkeley_humanoid_lite"]="berkeley_humanoid_lite"
robot_dir_to_urdf_key["fourier_n1"]="fourier_n1"
robot_dir_to_urdf_key["unitree_g1"]="unitree_g1"
robot_dir_to_urdf_key["unitree_h1"]="unitree_h1"
robot_dir_to_urdf_key["Berkeley_Humanoid_Lite"]="berkeley_humanoid_lite"
robot_dir_to_urdf_key["Boston_Dynamics_Atlas_HD"]="boston_dynamics_atlas_hd"
robot_dir_to_urdf_key["Fourier_N1"]="fourier_n1"
robot_dir_to_urdf_key["GR1"]="fourier_gr1"
robot_dir_to_urdf_key["iCub"]="icub"
robot_dir_to_urdf_key["Unitree_G1"]="unitree_g1"
robot_dir_to_urdf_key["Unitree_H1"]="unitree_h1"
robot_dir_to_urdf_key["agibot_x1"]="agibot_x1"
robot_dir_to_urdf_key["boston_dynamics_atlas_hd"]="boston_dynamics_atlas_hd"
robot_dir_to_urdf_key["fourier_gr1"]="fourier_gr1"
robot_dir_to_urdf_key["gr1"]="fourier_gr1"
robot_dir_to_urdf_key["icub"]="icub"
robot_dir_to_urdf_key["Unitree_G1_PROCESSED"]="unitree_g1"
robot_dir_to_urdf_key["Unitree_H1_PROCESSED"]="unitree_h1"

for method in "${methods[@]}"; do
  robots_string=${method_robots[$method]}
  for robot_dir in $robots_string; do
    # get motion root path
    motion_path="retargeted_motions/${method}/${robot_dir}"
    # set urdf file path
    urdf_key=${robot_dir_to_urdf_key["${robot_dir}"]}
    urdf_file=${robot_urdfs["${urdf_key}"]}
    # iterate over json files in the motion path
    for json_file in ${motion_path}/*.json; do
      echo "Visualizing ${json_file} for robot ${robot_dir} using method ${method}"
      # only run if "${motion_path}/$(basename "${json_file%.*}")_visualization.mp4" does not exist
      output_video="${motion_path}/$(basename "${json_file%.*}")_visualization.mp4"
      if [ -f "${output_video}" ]; then
        echo "Visualization already exists, skipping..."
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
    done
  done
done