# MAHR

![MAHR: morphology-agnostic humanoid motion retargeting](./images/mahr-teaser.png)

## About this project

MAHR provides a pipeline for retargeting human motion capture data to humanoid robots,
based on *Morphology-Agnostic Humanoid Retargeting via Perceptual Motivated Graph
Similarity*. It focuses on efficient processing of individual motions across robots
with different body proportions and joint configurations.

The project supports BVH input, eight robot models, height-map terrain, multi-motion
batch processing, trajectory export, and video playback. Robot models, meshes,
reference poses, an example height map, and two example motions are included
in this repository. The commands below reproduce both motions on all supported robots
and compare their results in a single video.

The output consists of kinematic trajectories for motion analysis, data preparation,
and downstream control research. Ground penetration may occur, and dynamic balance,
self-collision, and contact forces are not validated by this pipeline.

## How to use

Run the following commands from the root directory of this repository. The Python
module and command-line entry point are both named `mahr`. The examples below use
Linux and Bash.

### Installation

#### 1. Create the virtual environment with uv

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) to create the virtual
environment and install dependencies. Install uv before running the commands below.
Python 3.10 or higher is required; the examples use Python 3.10 and the tested core
dependency versions:

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e . "numpy==2.2.6" "scipy==1.15.3" "mujoco==3.9.0"
```

Keep this environment active for all commands below. In a new shell, run
`source .venv/bin/activate` again before retargeting, rendering, or running tests.

#### 2. (Optional) Install visualization dependencies

Video rendering requires FFmpeg, including `ffprobe`, and OSMesa. On Debian or Ubuntu:

```bash
sudo apt-get install ffmpeg libosmesa6
```

The examples use OSMesa for rendering without a display. These system dependencies
are only needed when exporting videos.

### Preparing Mocap Data

The built-in mappings use AMASS/SMPL joint names and the Y-up BVH convention.
Two converted motions are included under `assets/motions/`:

| Motion | BVH file | Frames |
| --- | --- | ---: |
| Dance | `DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh` | 500 |
| SMPL example | `smpl_h_small.bvh` | 319 |

For other motions, supply a BVH file with the same joint names, coordinate convention,
and length units as the included examples. BVH files with a different skeleton require
a corresponding mapping. AMASS NPZ and CMU ASF/AMC files require conversion to BVH
before use; this repository does not include a converter for those formats.

### Robot Models

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

### Retargeting Motion Capture Data to Robots

For example, to retarget the Dance motion to Unitree G1:

```bash
python -m mahr \
  assets/motions/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh \
  --robot g1 -o results/g1.json
```

The retargeted joint angles, root positions, root orientations, and frame rate are
saved in `results/g1.json`. Solver statistics are saved in `results/g1.stats.json`.

Replace `g1` with another name from the table to change the target robot, or replace
the input path to process another compatible BVH file. Use `--frames 120` to select
the first 120 source frames and `--stride 2` to sample every second frame.
Run `python -m mahr --help` for the available options.

#### Retargeting on non-flat terrain

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
preserving the scaled horizontal path; it does not plan a route around obstacles.
Omit `--height-map` for the default flat ground.

Terrain runs also save `g1_ramp.terrain.npz` next to the trajectory. Keep these files
together when moving results: the renderer loads the saved terrain automatically,
without needing the input PNG or additional terrain arguments.

#### Processing multiple motions

Supply a directory to recursively process its BVH motions. Select all robots with
`--robot all`, or repeat the option, for example `--robot g1 --robot x1`:

```bash
python -m mahr assets/motions --robot all -o results/motion_batch
```

This command processes both included motions on eight robots. Files and directories
whose names start with `norm_pose` are excluded from directory scans. Both `.bvh` and
`.BVH` extensions are accepted. A single input file with multiple target robots also
uses batch mode.

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

### Visualizing Retargeted Motions

Use `scripts/render_motion.py` to render a trajectory with the corresponding robot
URDF. For the G1 result above:

```bash
MUJOCO_GL=osmesa python scripts/render_motion.py results/g1.json \
  --urdf assets/robots/unitree_g1/g1_29dof_stl.urdf \
  -o results/g1.mp4
```

The first argument points to the trajectory JSON, `--urdf` selects the playback
model, and `-o` specifies the output video. Videos play at the original motion speed.
For another robot, use its URDF path from the model table.

For a low-bandwidth preview at exactly 30 fps, use a smaller frame size and a bitrate
budget in kbit/s. `--ground-grid` makes terrain shape easier to see:

```bash
MUJOCO_GL=osmesa python scripts/render_motion.py results/g1_ramp.json \
  --urdf assets/robots/unitree_g1/g1_29dof_stl.urdf \
  --fps 30 --width 320 --height 240 --bitrate 120 --ground-grid \
  -o results/g1_ramp_small.mp4
```

`--fps` samples the original trajectory at evenly spaced timestamps and preserves
playback speed. It is an alternative to `--stride`. `--bitrate` sets the H.264 target
and maximum rate; short buffering bursts are allowed. MP4 files are prepared for
progressive playback. Use `--camera-distance` in meters to adjust the framing.

For a terrain result, use the same command with `results/g1_ramp.json`; its adjacent
terrain file supplies the scene geometry. To render every completed trajectory in a
multi-motion batch:

```bash
python scripts/render_all.py results/motion_batch
```

Videos are written alongside their trajectory files. Tasks that failed before export
are skipped; non-converged results are rendered for inspection. You can run this command
while a batch is still processing: it reads the completed tasks available at startup.
Run it again to include later results.

### Reproducing the Examples on All Robots

Run both example motions on all eight robots:

```bash
python scripts/retarget_all.py \
  --motion assets/motions/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh \
  --output results/all_robots_dance

python scripts/retarget_all.py \
  --motion assets/motions/smpl_h_small.bvh \
  --output results/all_robots_smpl
```

Together, these runs produce 6552 frames of robot motion. The script checks output
values, joint limits, and solver status. Successful runs report `failed 0` for each
robot and write a `summary.json` file in each output directory.

Render the individual videos and combine them into comparison videos and GIFs:

```bash
for clip in dance smpl; do
  python scripts/render_all.py "results/all_robots_${clip}"
  python scripts/make_gallery.py "results/all_robots_${clip}"
done
```

Batch rendering selects the robot models automatically and uses OSMesa. Each directory
contains the robot trajectories and statistics, individual videos such as `g1.mp4`,
and the comparison files `all_robots.mp4` and `all_robots.gif`. The complete results
are also packaged as `results/all_robots_dance_results.zip` and
`results/all_robots_smpl_results.zip`.

Generated results are stored under `results/` and are not included in Git. The commands
above regenerate them from the supplied assets. You can also check the installation
and basic functionality with:

```bash
python -m unittest discover -s tests -q
```

## Acknowledgements

We would like to thank the following projects:

- [MuJoCo](https://github.com/google-deepmind/mujoco) for robot kinematics and visualization.
- NumPy and SciPy for numerical computation and optimization.
- The robot model authors and motion capture dataset contributors.

## License

The code is released under the [MIT License](LICENSE). Robot assets and motion data
remain subject to their respective licenses; model-specific license files are
included alongside the assets.
