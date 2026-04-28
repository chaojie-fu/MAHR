#!/bin/bash

declare -A gr1_dirs
gr1_dirs=(
  ["IK"]="retargeted_motions/IK/GR1"
  ["MAHR"]="retargeted_motions/MAHR_Wider_Constraint/fourier_gr1"
  ["JM"]="retargeted_motions/JM/GR1"
  ["Source"]="retargeted_motions/MAHR_Wider_Constraint/bvh"
)

motion_names=(
#  "144_32"
  "80_43"
#  "82_05"
#  "90_28"
#  "A10-lie_to_crouch"
#  "Extended_1"
#  "Extended_3"
#  "G4-spinning_back_kick"
#  "G7-capoera"
)

# a dict containing GMR mapping from motion name to file name
declare -A ik_mapping
ik_mapping=(
  ["80_43"]="80_43_poses_ik_v2_visualization.mp4"
  ["82_05"]="82_05_poses_ik_v2_visualization.mp4"
  ["90_28"]="90_28_poses_ik_v2_visualization.mp4"
  ["A10-lie_to_crouch"]="A10-lie_to_crouch_poses_ik_v2_visualization.mp4"
  ["Extended_1"]="Extended_1_poses_ik_v2_visualization.mp4"
  ["Extended_3"]="Extended_3_poses_ik_v2_visualization.mp4"
  ["G4-spinning_back_kick"]="G4-spinning_back_kick_poses_ik_v2_visualization.mp4"
  ["G7-capoera"]="G7-capoera_poses_ik_v2_visualization.mp4"
)

# a dict containing MAHR mapping from motion name to file name
declare -A mahr_mapping
mahr_mapping=(
  ["80_43"]="80_43_poses_gd_v2_visualization.mp4"
  ["82_05"]="82_05_poses_gd_v2_visualization.mp4"
  ["90_28"]="90_28_poses_gd_v2_visualization.mp4"
  ["A10-lie_to_crouch"]="A10-lie_to_crouch_poses_gd_v2_visualization.mp4"
  ["Extended_1"]="Extended_1_poses_gd_v2_visualization.mp4"
  ["Extended_3"]="Extended_3_poses_gd_v2_visualization.mp4"
  ["G4-spinning_back_kick"]="G4-spinning_back_kick_poses_gd_v2_visualization.mp4"
  ["G7-capoera"]="G7-capoera_poses_gd_v2_visualization.mp4"
)
# a dict containing PHC mapping from motion name to file name
declare -A jm_mapping
jm_mapping=(
  ["80_43"]="80_43_poses_jm_v2_visualization.mp4"
  ["82_05"]="82_05_poses_jm_v2_visualization.mp4"
  ["90_28"]="90_28_poses_jm_v2_visualization.mp4"
  ["A10-lie_to_crouch"]="A10-lie_to_crouch_poses_jm_v2_visualization.mp4"
  ["Extended_1"]="Extended_1_poses_jm_v2_visualization.mp4"
  ["Extended_3"]="Extended_3_poses_jm_v2_visualization.mp4"
  ["G4-spinning_back_kick"]="G4-spinning_back_kick_poses_jm_v2_visualization.mp4"
  ["G7-capoera"]="G7-capoera_poses_jm_v2_visualization.mp4"
)


# iterate over motion names
for motion_name in "${motion_names[@]}"; do
  echo "Concatenating G1 videos for motion: ${motion_name}..."
  ffmpeg -y \
         -i "${gr1_dirs["Source"]}/${motion_name}.mp4" \
         -i "${gr1_dirs["MAHR"]}/${mahr_mapping[$motion_name]}" \
         -i "${gr1_dirs["IK"]}/${ik_mapping[$motion_name]}" \
         -i "${gr1_dirs["JM"]}/${jm_mapping[$motion_name]}" \
         -filter_complex "\
         nullsrc=size=1920x1080 [base]; \
         [0:v] setpts=PTS-STARTPTS, scale=960x540, \
             drawtext=font='Arial':text='Source':x=20:y=20:fontsize=40:fontcolor=black [upperleft]; \
         [1:v] setpts=PTS-STARTPTS, scale=960x540, \
             drawtext=font='Arial':text='GD':x=20:y=20:fontsize=40:fontcolor=black [upperright]; \
         [2:v] setpts=PTS-STARTPTS, scale=960x540, \
             drawtext=font='Arial':text='IK':x=20:y=20:fontsize=40:fontcolor=black [lowerleft]; \
         [3:v] setpts=PTS-STARTPTS, scale=960x540, \
             drawtext=font='Arial':text='JM':x=20:y=20:fontsize=40:fontcolor=black [lowerright]; \
         [base][upperleft] overlay=shortest=1:x=0:y=0 [tmp1]; \
         [tmp1][upperright] overlay=shortest=1:x=960:y=0 [tmp2]; \
         [tmp2][lowerleft] overlay=shortest=1:x=0:y=540 [tmp3]; \
         [tmp3][lowerright] overlay=shortest=1:x=960:y=540 \
         " \
         -c:v libx264 -crf 18 -preset veryfast \
         "retargeted_motions/ComparingMethods/fourier_gr1_${motion_name}_poses_comparison_visualization.mp4"
done