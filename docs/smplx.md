# SMPL-X data preparation

[Back to MAHR](../README.md)

Run commands from the repository root with the
[uv environment](../README.md#installation) active.

Prepare two downloads: a **body model**, which defines the person's body shape and
joints, and **motion data**, which supplies the poses over time. AMASS SMPL-X motions
can be used directly after extraction. Use the uv environment linked above for the checks and commands below.

## 1. Download the SMPL-X body models

Register and sign in on the [SMPL-X download page](https://smpl-x.is.tue.mpg.de/download.php),
accept the applicable model terms, and download the SMPL-X models with the full shape
and expression space. This workflow was validated with a full SMPL-X v1.1 model.
Extract the archive and copy its neutral, female, and male model files into:

```text
assets/body_models/smplx/
  SMPLX_NEUTRAL.npz
  SMPLX_FEMALE.npz
  SMPLX_MALE.npz
```

This directory is ignored by Git. Official model `.npz` files are preferred; trusted
model `.pkl` files with the same names are also accepted. Use the full model when a
motion has more than ten `betas`; a compact model with ten shape coefficients cannot represent those
coefficients. Preserve the motion's shape coefficients instead of truncating them.

You can keep them elsewhere and pass their directory or a specific model file through
`--smplx-model`. A directory may contain the files directly or inside `smplx/`.
The motion's `gender` selects the model; absent gender defaults to neutral. Use
`--smplx-gender` only when intentionally overriding that selection, which changes the
body model used to interpret the motion. MAHR reads these files with its existing
dependencies; installing the `smplx` package or editing its source is unnecessary.

## 2. Download and extract AMASS SMPL-X motions

Register and sign in on the [AMASS download page](https://amass.is.tue.mpg.de/download.php),
accept the terms for the datasets you want, and select their **SMPL-X** downloads.
For example, start with the DanceDB subset. The SMPL+H/SMPL-H downloads use a different
parameter layout and cannot be passed to this loader or converted by padding poses
with zeros.

Extract each downloaded archive into a motion directory separate from the body models
and output trajectories. For a tar archive, replacing both example paths:

```bash
mkdir -p /path/to/amass
tar -xf /path/to/downloaded_amass_smplx_archive.tar.bz2 -C /path/to/amass
```

Keep the extracted dataset and subject directories. One DanceDB motion is
`DanceDB/20120807_CliodelaVara/Clio_Haniotikos_C3D_stageii.npz`; adjust the input path
to the actual extraction layout. The examples below use `/path/to/amass/clip_stageii.npz`
as a placeholder for a selected motion file. Keep model files and unrelated NPZ files
outside the motion directory, since batch mode treats NPZ files as motion inputs.

## 3. Check the input before retargeting

Run this check with the downloaded models and one extracted motion:

```bash
python - /path/to/amass/clip_stageii.npz <<'PY'
import sys
from mahr import SMPLXSource

source = SMPLXSource("assets/body_models/smplx")
motion, _ = source.read(sys.argv[1])
print(f"{len(motion.positions)} frames at {motion.fps:g} Hz; "
      f"{len(motion.names)} joints; positions in {motion.units}")
PY
```

A valid input reports 55 joints. Keep AMASS translations in meters, rotations in
radians, and the recorded frame rate. MAHR uses AMASS Z-up coordinates by default
and calibrates the source floor automatically. No BVH conversion, manual ground
offset, or advance resampling is required. Continue with
[Retargeting SMPL-X motions](#retargeting-smpl-x-motions) to produce robot trajectories.

If loading fails, check that the model for the motion's gender exists, that you selected
the SMPL-X dataset variant, and that the model supports all supplied shape coefficients.

## Parameters exported by other tools

For another data source, export a NumPy `.npz` file with numeric parameter arrays and
scalar metadata using the fields below. Raw `.pt` predictions and dataset
pickle containers need to be converted to this layout before loading.

| Parameter | Accepted fields and shape |
| --- | --- |
| Root translation, in meters | `trans` or `transl`: `[frames, 3]` |
| Pose, in axis-angle radians | `poses`: `[frames, 165]`, or the separate fields below |
| Root and body pose | `root_orient`/`global_orient`: `[frames, 3]`; `pose_body`/`body_pose`: `[frames, 63]` |
| Shape | `betas`: `[coefficients]`, `[1, coefficients]`, or `[frames, coefficients]` |
| Frame rate | Positive scalar `mocap_frame_rate`, `mocap_framerate`, or `fps` |

Separate poses can also include `pose_hand` (`[frames, 90]`) or
`left_hand_pose`/`right_hand_pose` (`[frames, 45]` each), plus jaw and eye rotations.
Omitted hand and facial rotations are zero. Optional `expression` coefficients are
supported. A full `poses` array takes precedence over separate fields and follows
SMPL-X order: root (3), body (63), jaw (3), left eye (3), right eye (3), left hand (45),
and right hand (45). All per-frame arrays must have the same frame count.

Store `gender` as `neutral`, `female`, or `male`; if supplied, `surface_model_type` or
`model_type` must be `smplx`. Hand poses use full axis angles; expand PCA hand
coefficients using the exporting tool's matching body model first. Angles are absolute
by default; use `--smplx-hand-mean model` for values relative to the model's hand mean.
For Y-up motion files, pass `--smplx-up-axis y` (or `up_axis="y"` in Python).
Preserve the producer's coordinate and hand conventions when preparing the arrays.

## Retargeting SMPL-X motions

After preparing the models and motion data above,
try the first 120 source frames on G1:

```bash
python -m mahr /path/to/amass/clip_stageii.npz --robot g1 \
  --smplx-model assets/body_models/smplx --frames 120 -o results/g1_smplx.json
```

The command writes `results/g1_smplx.json` and `results/g1_smplx.stats.json`.
Remove `--frames 120` to process the complete clip. `--stride 2` keeps every second
source frame and adjusts the output frame rate to preserve playback speed; it does
not select a fixed output frequency.

Render a small 30 Hz preview after installing the [visualization dependencies](../README.md#visualize-the-result):

```bash
MUJOCO_GL=osmesa python scripts/render_motion.py results/g1_smplx.json \
  --urdf assets/robots/unitree_g1/g1_29dof_stl.urdf \
  --fps 30 --width 320 --height 240 --bitrate 120 \
  -o results/g1_smplx.mp4
```

To process the extracted motion directory on all eight robots:

```bash
python -m mahr /path/to/amass --robot all \
  --smplx-model assets/body_models/smplx -o results/smplx_batch
```

Batch output preserves the dataset and subject directories beneath each robot's
directory. Check `results/smplx_batch/batch_summary.json` for failed or unconverged
tasks. SMPL-X input also accepts the [terrain options](usage.md#retargeting-on-non-flat-terrain).
See [batch processing](usage.md#processing-multiple-motions) for output layout and failure handling.
