#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate environment if available (non-fatal if missing)
if [[ -f /home/cjfu/Envs/MAHR/bin/activate ]]; then
  source /home/cjfu/Envs/MAHR/bin/activate
else
  echo "[INFO] Virtualenv /home/cjfu/Envs/MAHR not found; continuing without activation." >&2
fi

which python || true

# Optimizers to compare
optimizers=(
#  "lbfgsb"
  "de"
)

# Map each motion to its own list of frames.
# Supported frame specs per motion:
# - Space-separated or comma-separated integers: "0 5 10" or "0,5,10"
# - Numeric ranges with optional step: "0-10" (0..10) or "0-20:5" (0,5,10,15,20)
# You can also mix them: "0, 3-9:3 25"
# Example:
# frames_by_motion["path/to/foo.bvh"]="0 5 10-30:10 42"
# frames_by_motion["path/to/bar.bvh"]="3,7,11"
lbfgsb_max_iters=(
  "50"
  "100"
  "500"
  "1000"
  "2000"
  "5000"
)

de_max_iters=(
#  "50"
#  "100"
#  "500"
#  "1000"
#  "2000"
#  "5000"
  "10000"
)

declare -A frames_by_motion
frames_by_motion["mahr/rsc/amass_bvh/ACCAD/A10-lie_to_crouch_poses.bvh"]="0-520:30"
frames_by_motion["mahr/rsc/amass_bvh/ACCAD/Extended_1_poses.bvh"]="0-2400:30"
frames_by_motion["mahr/rsc/amass_bvh/ACCAD/Extended_3_poses.bvh"]="0-6360:30"
frames_by_motion["mahr/rsc/amass_bvh/ACCAD/G4-spinning_back_kick_poses.bvh"]="0-320:30"
frames_by_motion["mahr/rsc/amass_bvh/ACCAD/G7-capoera_poses.bvh"]="0-280:30"
frames_by_motion["mahr/rsc/amass_bvh/CMU/80_43_poses.bvh"]="0-980:30"
frames_by_motion["mahr/rsc/amass_bvh/CMU/82_05_poses.bvh"]="0-2240:30"
frames_by_motion["mahr/rsc/amass_bvh/CMU/90_28_poses.bvh"]="0-770:30"
frames_by_motion["mahr/rsc/amass_bvh/CMU/144_32_poses.bvh"]="0-5200:30"
frames_by_motion["mahr/rsc/amass_bvh/DanceDB_SMPLHG/Stefanos_Zeibekiko_CY_C3D_poses.bvh"]="0-9290:30"

# Expand a frame spec string into a space-separated list of integers
expand_frames() {
  local spec="$1"
  local out=()
  # Normalize commas to spaces
  spec="${spec//,/ }"
  for token in $spec; do
    if [[ "$token" =~ ^-?[0-9]+$ ]]; then
      out+=("$token")
    elif [[ "$token" =~ ^(-?[0-9]+)-(-?[0-9]+)(:(-?[0-9]+))?$ ]]; then
      local start="${BASH_REMATCH[1]}"
      local end="${BASH_REMATCH[2]}"
      local step="${BASH_REMATCH[4]}"
      if [[ -z "$step" ]]; then
        if (( start <= end )); then step=1; else step=-1; fi
      fi
      # Guard against zero step
      if (( step == 0 )); then
        echo "Warning: step 0 in token '$token' ignored; defaulting to 1/-1" >&2
        if (( start <= end )); then step=1; else step=-1; fi
      fi
      local i
      if (( step > 0 )); then
        for (( i=start; i<=end; i+=step )); do out+=("$i"); done
      else
        for (( i=start; i>=end; i+=step )); do out+=("$i"); done
      fi
    else
      echo "Warning: unrecognized frame token '$token' in spec '$1'" >&2
    fi
  done
  printf '%s ' "${out[@]}"
}

# Build tab-separated triples: motion\tframe\toptimizer
triples=()
for motion in "${!frames_by_motion[@]}"; do
  frames_spec=${frames_by_motion[$motion]}
  # Expand spec to an array of frames
  read -r -a frames_array <<< "$(expand_frames "$frames_spec")"
  for frame in "${frames_array[@]}"; do
    for opt in "${optimizers[@]}"; do
      # if opt equals to lbfgsb or de, iterate over their respective max iters
      if [[ "$opt" == "lbfgsb" ]]; then
        # Iterate over L-BFGS-B max iters
        for maxiter in "${lbfgsb_max_iters[@]}"; do
          triples+=("${motion}"$'\t'"${frame}"$'\t'"${opt}"$'\t'"${maxiter}"$'\t'"0")
        done
        continue
      elif [[ "$opt" == "de" ]]; then
        # Iterate over DE max iters
        for maxiter in "${de_max_iters[@]}"; do
          triples+=("${motion}"$'\t'"${frame}"$'\t'"${opt}"$'\t'"0"$'\t'"${maxiter}")
        done
        continue
      fi
    done
  done
done

# Fail early if nothing to run
if (( ${#triples[@]} == 0 )); then
  echo "No combinations to run." >&2
  exit 1
fi

# Parallelize over the prepared triples using tab-separated columns
# Columns: {1}=motion, {2}=frame, {3}=optimizer
parallel --jobs 96 --colsep '\t' \
  python "${SCRIPT_DIR}/comparing_optimization_methods.py" \
    --bvh "{1}" \
    --frame "{2}" \
    --optimizer "{3}" \
    --norm-pose t_pose \
    --norm-pose-dir "mahr/rsc/amass_bvh/norm_pose" \
    --no-filter-edges \
    --db-path "log/optimizations.sqlite" \
    --lbfgsb-maxiter "{4}" \
    --de-maxiter "{5}" \
    --de-popsize 15 \
    --de-strategy "best1bin" \
  ::: "${triples[@]}"
