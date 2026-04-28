#!/bin/bash

declare -A g1_dirs
g1_dirs=(
  ["GMR"]="retargeted_motions/GMR/unitree_g1"
  ["MAHR"]="retargeted_motions/MAHR_Wider_Constraint/unitree_g1"
  ["PHC"]="retargeted_motions/PHC/Unitree_G1_PROCESSED"
  ["Source"]="retargeted_motions/MAHR_Wider_Constraint/bvh"
)

motion_names=(
  "144_32"
  "80_43"
  "82_05"
  "90_28"
  "A10-lie_to_crouch"
  "Extended_1"
  "Extended_3"
  "G4-spinning_back_kick"
  "G7-capoera"
)

# a dict containing GMR mapping from motion name to file name
declare -A gmr_mapping
gmr_mapping=(
  ["144_32"]="144_32_stageii.npz_visualization.mp4"
  ["80_43"]="80_43_stageii.npz_visualization.mp4"
  ["82_05"]="82_05_stageii.npz_visualization.mp4"
  ["90_28"]="90_28_stageii.npz_visualization.mp4"
  ["A10-lie_to_crouch"]="A10_-_lie_to_crouch_stageii.npz_visualization.mp4"
  ["Extended_1"]="Extended_1_stageii.npz_visualization.mp4"
  ["Extended_3"]="Extended_3_stageii.npz_visualization.mp4"
  ["G4-spinning_back_kick"]="G4_-spinning_back_kick_stageii.npz_visualization.mp4"
  ["G7-capoera"]="G7_-__capoera_stageii.npz_visualization.mp4"
)

# a dict containing MAHR mapping from motion name to file name
declare -A mahr_mapping
mahr_mapping=(
  ["144_32"]="144_32_poses_gd_v2_visualization.mp4"
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
declare -A phc_mapping
phc_mapping=(
  ["144_32"]="0-selected_data_144_32_stageii_visualization.mp4"
  ["80_43"]="0-selected_data_80_43_stageii_visualization.mp4"
  ["82_05"]="0-selected_data_82_05_stageii_visualization.mp4"
  ["90_28"]="0-selected_data_90_28_stageii_visualization.mp4"
  ["A10-lie_to_crouch"]="0-selected_data_A10_-_lie_to_crouch_stageii_visualization.mp4"
  ["Extended_1"]="0-selected_data_Extended_1_stageii_visualization.mp4"
  ["Extended_3"]="0-selected_data_Extended_3_stageii_visualization.mp4"
  ["G4-spinning_back_kick"]="0-selected_data_G4_-spinning_back_kick_stageii_visualization.mp4"
  ["G7-capoera"]="0-selected_data_G7_-__capoera_stageii_visualization.mp4"
)


# iterate over motion names
for motion_name in "${motion_names[@]}"; do
  echo "Concatenating G1 videos for motion: ${motion_name}..."
  ffmpeg -y \
         -i "${g1_dirs["Source"]}/${motion_name}.mp4" \
         -i "${g1_dirs["MAHR"]}/${mahr_mapping[$motion_name]}" \
         -i "${g1_dirs["GMR"]}/${gmr_mapping[$motion_name]}" \
         -i "${g1_dirs["PHC"]}/${phc_mapping[$motion_name]}" \
         -filter_complex "\
         nullsrc=size=1920x1080 [base]; \
         [0:v] setpts=PTS-STARTPTS, scale=960x540, \
             drawtext=font='Arial':text='Source':x=20:y=20:fontsize=40:fontcolor=black [upperleft]; \
         [1:v] setpts=PTS-STARTPTS, scale=960x540, \
             drawtext=font='Arial':text='GD':x=20:y=20:fontsize=40:fontcolor=black [upperright]; \
         [2:v] setpts=PTS-STARTPTS, scale=960x540, \
             drawtext=font='Arial':text='GMR':x=20:y=20:fontsize=40:fontcolor=black [lowerleft]; \
         [3:v] setpts=PTS-STARTPTS, scale=960x540, \
             drawtext=font='Arial':text='PHC':x=20:y=20:fontsize=40:fontcolor=black [lowerright]; \
         [base][upperleft] overlay=shortest=1:x=0:y=0 [tmp1]; \
         [tmp1][upperright] overlay=shortest=1:x=960:y=0 [tmp2]; \
         [tmp2][lowerleft] overlay=shortest=1:x=0:y=540 [tmp3]; \
         [tmp3][lowerright] overlay=shortest=1:x=960:y=540 \
         " \
         -c:v libx264 -crf 18 -preset veryfast \
         "retargeted_motions/ComparingMethods/unitree_g1_${motion_name}_poses_comparison_visualization.mp4"
done