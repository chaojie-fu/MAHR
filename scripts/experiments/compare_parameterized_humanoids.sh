#!/bin/bash
set -euo pipefail

root_dir="/home/cjfu/Retarget_SOTA/MAHR/retargeted_motions/ParameterizedHumanoid"

motion_names=(
  "80_43_poses"
  "90_28_poses"
  "Extended_1_poses"
  "G4-spinning_back_kick_poses"
  "82_05_poses"
  "A10-lie_to_crouch_poses"
  "Extended_3_poses"
  "G7-capoera_poses"
)

vals=(00 05 10 15 20)

declare -A val_labels
val_labels=(
  ["00"]="1\:1.75"
  ["05"]="1\:1.375"
  ["10"]="1\:1"
  ["15"]="1.375\:1"
  ["20"]="1.75\:1"
)

# define canvas size
# define columns and rows based on the number of elements in vals
cols=${#vals[@]}
rows=${#vals[@]}
out_w=1920
out_h=1080
tile_w=$((out_w / cols))
tile_h=$((out_h / rows))


# iterate over motion names
for motion_name in "${motion_names[@]}"; do
  echo "Concatenating G1 videos for motion: ${motion_name}..."

  inputs=()
  labels=()
  for WW in "${vals[@]}"; do
    for LL in "${vals[@]}"; do
      f="${root_dir}/${WW}_${LL}/${motion_name}/${motion_name}_gd_v2_visualization.mp4"
      inputs+=("$f")
#      labels+=("${WW}_${LL}")
      labels+=("Arm\:Body ${val_labels[${WW}]}; Leg\:Body ${val_labels[${LL}]}")
    done
  done

  # build ffmpeg input arguments
  args=(-y)
  for f in "${inputs[@]}"; do
    args+=(-i "$f")
  done

  # define filter complex
  filter="nullsrc=size=${out_w}x${out_h} [base]; "
  for i in "${!inputs[@]}"; do
    label="${labels[$i]}"
    filter+="[${i}:v] setpts=PTS-STARTPTS, scale=${tile_w}x${tile_h}, "
    # add a border around each grid tile
    filter+="drawbox=x=0:y=0:w=iw:h=ih:t=2:c=black, "
    filter+="drawtext=font='Arial':text='${label}':x=20:y=10:fontsize=20:fontcolor=black "
    filter+="[v${i}]; "
  done

  # define overlays
  last="[base]"
  for i in "${!inputs[@]}"; do
    row=$((i / cols))
    col=$((i % cols))
    x=$((col * tile_w))
    y=$((row * tile_h))
    tmp="[tmp${i}]"
    filter+="${last}[v${i}] overlay=shortest=1:x=${x}:y=${y} ${tmp}; "
    last="${tmp}"
  done

  filter+="${last} copy [out]"

  out="retargeted_motions/ComparingMethods/parameterized_humanoids_${motion_name}_poses_comparison_visualization.mp4"

  # if out file already exists, skip
  if [ -f "$out" ]; then
    echo "Output file ${out} already exists, skipping..."
    continue
  fi
  mkdir -p "$(dirname "$out")"
  ffmpeg "${args[@]}" \
    -filter_complex "$filter" \
    -map "[out]" \
    -c:v libx264 -crf 18 -preset veryfast \
    "$out"
done
