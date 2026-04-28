#!/bin/bash

# Activate environment if available (non-fatal if missing)
if [[ -f /home/cjfu/Envs/MAHR/bin/activate ]]; then
  source /home/cjfu/Envs/MAHR/bin/activate
else
  echo "[INFO] Virtualenv /home/cjfu/Envs/MAHR not found; continuing without activation." >&2
fi

which python || true

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

frame_spec="0-320:4"
de_max_iters=(
  50
  100
  500
  1000
  2000
  5000
  10000
)

# Build tab-separated triples: motion\tframe\toptimizer
triples=()
read -r -a frames_array <<< "$(expand_frames "$frame_spec")"
echo "Expanded frames: ${frames_array[*]}"
for frame in "${frames_array[@]}"; do
    for maxiter in "${de_max_iters[@]}"; do
      triples+=("${frame}"$'\t'"${maxiter}")
    done
done

# Fail early if nothing to run
if (( ${#triples[@]} == 0 )); then
  echo "No combinations to run." >&2
  exit 1
fi

parallel --jobs 96 --colsep '\t' \
  python ./comparing_optimization_methods_continuous_de.py \
    --bvh "mahr/rsc/amass_bvh/ACCAD/G4-spinning_back_kick_poses.bvh" \
    --norm-pose t_pose \
    --norm-pose-dir "mahr/rsc/amass_bvh/norm_pose" \
    --no-filter-edges \
    --frame "{1}" \
    --de-maxiter "{2}" \
    --db-path "log/optimizations_continuous.sqlite" \
  ::: "${triples[@]}"
