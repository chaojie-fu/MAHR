# MAHR

![MAHR: morphology-agnostic humanoid motion retargeting](./images/mahr-teaser.png)

[Paper](https://doi.org/10.3389/fnbot.2026.1814977) ·
[Quick start](#how-to-use) · [Documentation](#documentation) ·
[Full-size image](./images/mahr-teaser.png)

## About this project

MAHR retargets human motion capture data to humanoid robots with different body
proportions and joint configurations. It is based on
[Morphology-agnostic humanoid retargeting via perception-motivated graph similarity](https://doi.org/10.3389/fnbot.2026.1814977).

- **Eight robots**, with BVH and SMPL-X motion input.
- **Single-motion and batch processing**, including target height-map terrain.
- **Trajectory export and video rendering**, with automatic motion-quality checks
  and local repair attempts.

Robot models, meshes, reference poses, and example motions are included.
**The quick start requires no SMPL-X model or external motion downloads.**
Outputs are kinematic trajectories for analysis and control research; continuity
and physical executability are not guaranteed.

## How to use

The commands below use Linux and Bash. Run them from the repository root.

### Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create
and activate an environment with the tested dependency versions:

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e . "numpy==2.2.6" "scipy==1.15.3" "mujoco==3.9.0" "osqp==1.1.1"
```

Python 3.10 or higher is required. In a new shell, activate the environment again
with `source .venv/bin/activate` before running the commands below.

### Retarget an example

Retarget the included Dance motion to Unitree G1:

```bash
python -m mahr \
  assets/motions/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh \
  --robot g1 -o results/g1.json
```

This writes joint angles, root poses, and the motion frame rate to `results/g1.json`,
with solver and quality diagnostics in `results/g1.stats.json`.
Replace `g1` with another robot ID from the table below to change the target.

### Visualize the result

Video export requires FFmpeg (including `ffprobe`) and OSMesa. On Debian or Ubuntu:

```bash
sudo apt-get install ffmpeg libosmesa6
```

Render a 30 fps preview without a display:

```bash
MUJOCO_GL=osmesa python scripts/render_motion.py results/g1.json \
  --urdf assets/robots/unitree_g1/g1_29dof_stl.urdf \
  --fps 30 --width 640 --height 480 -o results/g1.mp4
```

Open `results/g1.mp4` to view the motion at its original speed. Rendering at 30 fps
changes video sampling, not the trajectory's frame rate.
See [reproduction and diagnostics](docs/reproduction.md) to compare all eight robots
and inspect any remaining motion anomalies.

## Supported robots and inputs

| Robot | `--robot` |
| --- | --- |
| Unitree G1 | `g1` |
| Unitree H1 | `h1` |
| Fourier GR1 | `gr1` |
| Fourier N1 | `n1` |
| Agibot X1 | `x1` |
| Berkeley Humanoid Lite | `berkeley` |
| Boston Dynamics Atlas HD | `atlas` |
| iCub | `icub` |

**BVH:** built-in mappings expect AMASS/SMPL joint names and the Y-up convention.
Use the included examples as a reference; other skeletons require a matching mapping.

**SMPL-X:** AMASS SMPL-X NPZ files are supported with separately downloaded body models.
SMPL/SMPL-H parameters and CMU ASF/AMC files require conversion or another adapter.

To process both bundled motions on all eight robots:

```bash
python -m mahr assets/motions --robot all -o results/motion_batch
python scripts/render_all.py results/motion_batch
```

Batch results and videos are organized by robot and input path under
`results/motion_batch/`. Generated outputs are ignored by Git.

## Documentation

| Guide | Contents |
| --- | --- |
| [SMPL-X data preparation](docs/smplx.md) | Download body models and AMASS motions, check inputs, and retarget them |
| [Retargeting guide](docs/usage.md) | BVH requirements, robot paths, frame selection, terrain, batches, and Python API |
| [Rendering and reproduction](docs/reproduction.md) | Video options, eight-robot comparisons, and motion-quality diagnostics |

Run `python -m mahr --help` for command-line options. Solver convergence does not
imply good motion quality; check `recovery.unresolved_frames` in the statistics
and inspect the output before downstream use.

## Citation

If you use MAHR in your research, please cite:

Fu C, Su C, Jiang L, Lan K, Jin Y and Wang H (2026).
*Morphology-agnostic humanoid retargeting via perception-motivated graph similarity.*
Frontiers in Neurorobotics, 20:1814977.
[doi:10.3389/fnbot.2026.1814977](https://doi.org/10.3389/fnbot.2026.1814977).

## Acknowledgements

We thank MuJoCo for robot kinematics and visualization, NumPy and SciPy for numerical
computation, OSQP for trajectory repair, and the robot model authors and motion
capture dataset contributors.

## License

The code is released under the [MIT License](LICENSE). Robot assets and motion data
remain subject to their respective licenses; model-specific license files are
included alongside the assets.
