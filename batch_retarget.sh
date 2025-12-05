source /home/cjfu/Envs/MAHR/bin/activate

which python

# a list of source mocap files
mocap_files=(
  # normal actions
  "mahr/rsc/amass_bvh/ACCAD/A10-lie_to_crouch_poses.bvh"
  "mahr/rsc/amass_bvh/ACCAD/Extended_1_poses.bvh"
  "mahr/rsc/amass_bvh/ACCAD/Extended_3_poses.bvh"
  "mahr/rsc/amass_bvh/ACCAD/G4-spinning_back_kick_poses.bvh"
  "mahr/rsc/amass_bvh/ACCAD/G7-capoera_poses.bvh"
  "mahr/rsc/amass_bvh/CMU/80_43_poses.bvh"
  "mahr/rsc/amass_bvh/CMU/82_05_poses.bvh"
  "mahr/rsc/amass_bvh/CMU/90_28_poses.bvh"
  "mahr/rsc/amass_bvh/CMU/144_32_poses.bvh"
  "mahr/rsc/amass_bvh/DanceDB_SMPLHG/Stefanos_Zeibekiko_CY_C3D_poses.bvh"
)
robots=(
  "fourier_gr1"
#  "unitree_g1"
#  "unitree_h1"
#  "berkeley_humanoid_lite"
#  "fourier_n1"
#  "boston_dynamics_atlas_hd"
#  "icub"
#  "agibot_x1"
)

height_map_indices=(
  "0"
  "1"
  "2"
  "3"
)

# iterate over mocap files and robots
# iterate over mocap files
# Use GNU parallel to run the Cartesian product of mocap_files and robots (simple and concise)
parallel python motion_retarget_graph_distance.py \
  --dataset amass \
  --bvh '{1}' \
  --norm-pose-dir "mahr/rsc/amass_bvh/norm_pose" \
  --out-dir "./graph_distance_AMASS_to_{2}_Terrain{3}" \
  --robot '{2}' \
  --no-filter-edges \
  --height-map \
  "height_map/height_map{3}.png" \
  ::: "${mocap_files[@]}" ::: "${robots[@]}" ::: "${height_map_indices[@]}"
