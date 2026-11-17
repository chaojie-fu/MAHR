# Retargeting guide

[Back to MAHR](../README.md)

Run commands from the repository root with the
[uv environment](../README.md#installation) active.

## Motion inputs

The built-in mappings use AMASS/SMPL joint names. BVH input uses the Y-up convention.
Two converted motions are included under `assets/motions/`:

| Motion | BVH file | Frames |
| --- | --- | ---: |
| Dance | `DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh` | 500 |
| SMPL example | `smpl_h_small.bvh` | 319 |

For other motions, supply a BVH file with the same joint names, coordinate convention,
and length units as the included examples. BVH files with a different skeleton require
a corresponding mapping. SMPL-X NPZ motions can be loaded directly using the [SMPL-X guide](smplx.md).
SMPL/SMPL-H parameter files and CMU ASF/AMC files require conversion or a separate
input adapter; they are not interchangeable with SMPL-X parameters.

## Robot models

The supported robot models are included under `assets/robots/`.
Use the supplied URDF files together with their referenced meshes. The `--robot`
argument selects the target model for retargeting.

The URDF paths below are relative to `assets/robots/`:

| Robot | `--robot` | URDF path |
| --- | --- | --- |
| Unitree G1 | `g1` | `unitree_g1/g1_29dof_stl.urdf` |
| Unitree H1 | `h1` | `unitree_h1/urdf/h1_stl.urdf` |
| Fourier GR1 | `gr1` | `Fourier_GR1/GR1.urdf` |
| Fourier N1 | `n1` | `Fourier_N1/urdf/N1_raw_stl.urdf` |
| Agibot X1 | `x1` | `Agibot_X1/urdf/x1_stl.urdf` |
| Berkeley Humanoid Lite | `berkeley` | `Berkeley_Humanoid_Lite/berkeley_humanoid_lite_stl.urdf` |
| Boston Dynamics Atlas HD | `atlas` | `Boston_Dynamics_Atlas/urdf/atlas_v4_with_multisense_stl.urdf` |
| iCub | `icub` | `iCub/robots/iCubGazeboV2_7/model_stl.urdf` |

Retargeting uses this checkout's `assets/` directory by default. If you store the
assets elsewhere, pass `--asset-root /path/to/assets` to either `python -m mahr` or
`scripts/retarget_all.py`, preserving the `robots/` and `motions/` subdirectories.

## Selecting frames

Use `--frames 120` to select the first 120 source frames and `--stride 2` to keep
every second frame. Striding adjusts the output frame rate to preserve playback
speed; it does not select a fixed output frequency. Run `python -m mahr --help`
for all options.

## Retargeting on non-flat terrain

Pass a height map to adapt a motion to a different target ground surface. For example,
use the included ramp with an 8 m × 8 m footprint:

```bash
python -m mahr assets/motions/smpl_h_small.bvh --robot g1 \
  --height-map assets/terrains/ramp.png \
  --terrain-half-size 4 4 --terrain-elevation 0.4 \
  -o results/g1_ramp.json
```

Height maps must be 8-bit grayscale PNGs. Black represents zero height and white
represents `--terrain-elevation` meters. `--terrain-half-size X Y` sets half the map's
width and depth; its default is `20 20`. Image columns increase along world X and
rows run from positive to negative world Y. Use `--terrain-origin X Y Z` to move the
map's center and base height; the default is `0 0 0`. Outside the map footprint,
height queries use the nearest boundary. Choose a footprint covering the robot's path.

The source motion still uses a calibrated horizontal floor. The supplied height map
describes the target terrain. Retargeting adjusts joint angles and root height while
refining the height-scaled horizontal path; it does not plan a route around obstacles.
Omit `--height-map` for the default flat ground.

Terrain runs also save `g1_ramp.terrain.npz` next to the trajectory. Keep these files
together when moving results: the renderer loads the saved terrain automatically,
without needing the input PNG or additional terrain arguments.

## Processing multiple motions

Supply a directory to recursively process its BVH and SMPL-X NPZ motions. Select all robots with
`--robot all`, or repeat the option, for example `--robot g1 --robot x1`:

```bash
python -m mahr assets/motions --robot all -o results/motion_batch
```

This command processes both included motions on eight robots. Files and directories
whose names start with `norm_pose` and exported `.terrain.npz` files are excluded from
directory scans. File extensions are case-insensitive. Supply `--smplx-model` when
the directory includes SMPL-X motions; keep body-model files outside the motion
directory. A single input file with multiple target robots also uses batch mode.

In batch mode, `-o` is an output directory. Each robot gets a subdirectory that preserves
the input motion's relative path, for example
`results/motion_batch/g1/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.json`.
This keeps motions with the same filename in different directories separate.
`--frames`, `--stride`, and terrain options apply to every task.

The batch continues after individual failures and records completed tasks in
`batch_progress.jsonl`. It finalizes `batch_summary.json` on completion or a Ctrl-C
interruption. The command exits with a nonzero status if any task fails or any frame
does not converge. Non-converged trajectories remain available for inspection.
Read the summary before using a batch in downstream applications.

See [rendering and reproduction](reproduction.md) to view or compare the outputs.

## Python API

Use `run()` to retarget a motion and `save_result()` to export it:

```python
from mahr import run, save_result

result = run("assets", "assets/motions/smpl_h_small.bvh", robot="g1")
save_result(result, "results/g1.json")
```

For SMPL-X, pass a reusable source object after [preparing the body models](smplx.md):

```python
from mahr import SMPLXSource, run, save_result

source = SMPLXSource("assets/body_models/smplx")
result = run("assets", "/path/to/amass/clip_stageii.npz", robot="g1", smplx=source)
save_result(result, "results/g1_smplx.json")
```
