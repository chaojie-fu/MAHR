# Rendering, reproduction, and diagnostics

[Back to MAHR](../README.md)

Run commands from the repository root with the
[uv environment](../README.md#installation) active.

Install the [video dependencies](../README.md#visualize-the-result) before rendering.

## Visualizing retargeted motions

Use `scripts/render_motion.py` to render a trajectory with the corresponding robot
URDF. After generating `results/g1.json` with the [quick start](../README.md#retarget-an-example):

```bash
MUJOCO_GL=osmesa python scripts/render_motion.py results/g1.json \
  --urdf assets/robots/unitree_g1/g1_29dof_stl.urdf \
  -o results/g1.mp4
```

The first argument points to the trajectory JSON, `--urdf` selects the playback
model, and `-o` specifies the output video. Videos play at the original motion speed.
The scene uses a white sky and a gray/white checker floor; `--no-ground-grid` selects
a uniform gray floor. For another robot, use its URDF path from the [model table](usage.md#robot-models).

For a low-bandwidth preview of the [ramp example](usage.md#retargeting-on-non-flat-terrain)
at exactly 30 fps, use a smaller frame size and a bitrate budget in kbit/s.
The ground grid also makes terrain shape easier to see:

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
[multi-motion batch](usage.md#processing-multiple-motions):

```bash
python scripts/render_all.py results/motion_batch
```

Videos are written alongside their trajectory files. Tasks that failed before export
are skipped; non-converged results are rendered for inspection. You can run this command
while a batch is still processing: it reads the completed tasks available at startup.
Run it again to include later results.

## Reproducing the examples on all robots

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

## Inspecting motion quality

Solver convergence and motion quality are reported separately. In each trajectory's
`.stats.json`, check `recovery.unresolved_frames` even when `failed 0` is reported.
The `quality` and `recovery_status` entries in `frames_detail` locate remaining
suspect frames; divide the zero-based frame index by the trajectory's `fps` to find
their playback times. `recovery.windows` records accepted and rejected repairs,
and `recovery.window_budget_exhausted` indicates that the repair budget was reached.
Unresolved frames remain in the output for inspection.

The trajectory JSON `fps` is the playback rate; the `.stats.json` top-level `fps`
is processing throughput. Use the trajectory value to convert frame indices to
playback times. Inspect flagged intervals visually before using results downstream.
Ground penetration and abrupt motion may remain; dynamic balance, self-collision,
and contact forces are not validated.
