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
)
robots=(
  "parameterized_humanoid"
)

robot_urdfs=(
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/00_00.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/00_05.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/00_10.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/00_15.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/00_20.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/05_00.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/05_05.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/05_10.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/05_15.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/05_20.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/10_00.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/10_05.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/10_10.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/10_15.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/10_20.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/15_00.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/15_05.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/15_10.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/15_15.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/15_20.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/20_00.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/20_05.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/20_10.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/20_15.urdf"
  "misc/visualization/compare_different_robots/robots/parsed_humanoid/20_20.urdf"
)

# iterate over mocap files and robots
# iterate over mocap files
# Use GNU parallel to run the Cartesian product of mocap_files and robots (simple and concise)
parallel python motion_retarget_graph_distance.py \
  --dataset amass \
  --bvh '{1}' \
  --norm-pose-dir "mahr/rsc/amass_bvh/norm_pose" \
  --robot '{2}' \
  --no-filter-edges \
  --robot-urdf '{3}' \
  ::: "${mocap_files[@]}" ::: "${robots[@]}" ::: "${robot_urdfs[@]}"
