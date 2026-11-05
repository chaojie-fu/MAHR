# MAHR Maintenance Guide

This file applies to the entire MAHR repository. It records implementation conventions,
maintenance decisions, and validation procedures. README.md is for users: it describes
the project's purpose, features, scope, installation, data preparation, and reproduction
steps. Keep internal architecture, algorithm parameters, ablation conclusions, and agent
working conventions here.

## Documentation and Naming

- Write all project documentation in English, including new documents and updates to existing ones.
- Organize README with an overview under `About this project`, followed
  by `How to use`, installation, data preparation, robot models, retargeting, visualization,
  and acknowledgements. Keep explanations practical and commands reproducible.
- Do not add a `Development` section to README. Keep developer guidance and agent
  maintenance instructions in AGENTS.md.
- Use `MAHR` for the project display name and distribution metadata. Keep the Python module
  directory, import name, and command-line entry point lowercase: `mahr`. Package tools
  normalize the distribution and lockfile entry to `mahr`.
- Keep the repository self-contained: runtime code, robot models, required meshes,
  reference poses, and reproduction motions must be available within the checkout.
  Documentation must describe this project directly, without references to a separate
  predecessor repository or workstation-specific paths.
- Built-in examples, tests, and rendering must work without a neighboring checkout,
  external symlinks, or implicit fallback paths. Explicit user-provided inputs and
  standard Python/system dependencies are allowed. Verify asset and path changes
  from an isolated copy of the repository.
- Run README commands from this repository's root. Update reproduction instructions when
  changing the CLI or output format.
- Require uv for virtual environment creation and dependency installation. Document
  `uv venv --python 3.10 .venv`, activation, and `uv pip install` with the validated
  dependency versions; run subsequent commands with the activated environment's Python.

## Project Goals and Maintenance Principles

- Prioritize the latency of retargeting a single motion while keeping all eight built-in
  robots usable.
- Refactoring does not require a one-to-one correspondence with existing modules or data
  structures. Use Python classes for configuration; JSON is for trajectory and statistics output.
- Refer to the MAHR paper for the method. Where the paper and implemented behavior
  disagree, preserve the established numerical behavior and accepted decisions below.
- Keep the main implementation focused on the accepted default pipeline. Historical
  ablations and removed constraints are available in commit `58be08f`; keep experimental
  branches out of the default runtime.
- Validate numerical behavior after structural or performance changes. Changes to the
  objective, initialization strategy, or prior values require an explicit description of
  the behavioral change and a separate comparison.

## Dependencies and Assets

The runtime depends on NumPy, SciPy, MuJoCo, and Pillow (PNG height-map loading).
Bundled resources live in `assets/`:
`robots/` contains the eight configured URDFs, their referenced meshes, and available
model-specific license files; `motions/` contains the two reproduction clips and standing
reference pose; `terrains/ramp.png` is a generated 33-by-65 grayscale ramp for terrain
reproduction (pixel value `2 * column + 3 * row`). Preserve model-relative mesh paths and license notices when updating
assets. Optimization loads no visual or collision meshes; rendering loads the visual assets.

`DEFAULT_ASSET_ROOT` in `config.py` resolves `assets/` relative to this checkout, independently
of the working directory. Both command-line tools accept `--asset-root` for another directory
with the same layout. The Python API's first argument is `asset_root`. Tests use the bundled
assets by default and accept `MAHR_ASSET_ROOT` for an explicit override. The documented
installation is editable; a non-editable installation needs an explicit asset root.

The validated environment is Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3, and MuJoCo 3.9.0.
README uses these dependency versions for reproduction. The current `uv.lock` selects MuJoCo 3.17.0,
so it does not describe the same validated environment. Revalidate when upgrading
dependencies or aligning the lockfile.

Without installing the project, use `PYTHONPATH=src` from the repository root with
the documented dependency versions. Rendering has been validated with
`MUJOCO_GL=osmesa`; the batch renderer also sets `LP_NUM_THREADS=4`.

## Project Structure and Data Flow

```text
BVH or SMPL-X NPZ + Python robot/source config
        ↓ prepare()
PreparedMotion
        ↓ solve()
RetargetResult (prepared + trajectory + report)
        ↓ save_result()
trajectory JSON + solve statistics JSON
```

`run()` combines `prepare()` and `solve()` and returns an in-memory result. Keep
computation separate from file export.

| Path, relative to `src/mahr/` | Responsibility |
| --- | --- |
| `config.py` | Configuration base class, landmark-pair validation, pose dictionary isolation, and preset lookup |
| `robots/` | Eight robot configuration classes and the `PRESETS` registry |
| `motion.py` | Common motion arrays, BVH parsing, coordinate conversion, and source graph vectors |
| `smplx.py` | SMPL-X parameter validation, shape-dependent joint geometry, forward kinematics, and source reference poses |
| `source_cli.py` | Shared SMPL-X source arguments for retargeting entry points |
| `source_ground.py` | Static ground calibration over the complete input motion |
| `terrain.py` | Target height fields, world coordinates, height derivatives, and portable data export |
| `terrain_cli.py` | Shared terrain arguments for retargeting entry points |
| `rendering.py` | Flat ground or matching MuJoCo height-field scene geometry |
| `normalization.py` | Skeleton-height estimation and edge denominator floors |
| `kinematics.py` | Mesh-free MuJoCo FK, body origins, and landmark Jacobians |
| `graph.py` | Graph topology, ground-height relations, proximity weights, and edge derivatives |
| `objective.py` | Losses, analytic gradients, and a shared `LossBreakdown` |
| `solver.py` | `Settings`, sequential state, dynamic bounds, and L-BFGS-B calls |
| `pipeline.py` | Preparation, solving, result composition, and aggregate statistics |
| `batch.py` | Motion discovery, sequential task execution, output paths, and failure manifests |
| `trajectory.py` | In-memory trajectories, output format, and explicit export |
| `cli.py` | Command-line arguments and execution entry point |

`scripts/` contains four tools for the default pipeline: `retarget_all.py`,
`render_motion.py`, `render_all.py`, and `make_gallery.py`. Tests use the standard-library
unittest framework. Generated artifacts belong in the Git-ignored `results/` directory.

`PreparedMotion` contains the robot, graph, normalization scales, source vectors, motion,
horizontal scale ratio, initial state, posture reference, asset path, and metadata.
Preparation explicitly queries reference positions rather than reading MuJoCo state
that depends on the previous FK call.

Each prepared object owns a mutable MuJoCo workspace and supports sequential reuse.
Independent concurrent tasks must prepare their own workspaces. The kinematics object
does not own the posture prior. Repeated `solve()` calls must start from the declared
initial state, independently of the last FK evaluation.

## Python Configuration Conventions

- Each `landmark_pairs` row declares `(source joint, target body)`. Names on each side
  must be unique. Do not restore two independent lists that rely on positional alignment.
- Ground is a separate height relation controlled by `include_ground`, not an ordinary
  landmark in the mapping.
- `toes` and `source_toes` each specify two distinct calibration markers. These markers
  do not have to belong to the graph's landmarks.
- URDF and source-reference paths are relative to the supplied asset root.
  Pose values use radians in the URDF convention.
- `normalization_pose` defines reference geometry, skeleton height, horizontal scaling,
  and the initial root height.
- `initial_pose` supplies joint initialization; `posture_pose` supplies the posture prior.
  When omitted, each receives an independent copy of `normalization_pose`. Dictionaries
  are independent across instances and across the three pose roles.
- An explicitly supplied pose is a complete pose; omitted joints default to zero.
  Use dictionary expansion when inheriting a pose and overriding individual joints.
- Only clip the optimization initial state to joint bounds. Do not silently change
  reference geometry or the posture prior. iCub retains reference elbow angles and
  priors of 0°, while its URDF lower limit is 15°. Tests preserve this historical
  behavior; changes to it need a separate evaluation.

Example: change only X1's posture prior, preserving its reference geometry and initial pose.

```python
from math import radians
from mahr import run, save_result
from mahr.robots import AgibotX1Config

class CustomX1Config(AgibotX1Config):
    posture_pose = {
        **AgibotX1Config.normalization_pose,
        "left_knee_pitch_joint": radians(10),
    }

result = run("assets", "assets/motions/smpl_h_small.bvh", config=CustomX1Config("assets"))
save_result(result, "results/custom_x1.json")
```

Register a new robot class in `mahr.robots.PRESETS` to expose it through `--robot`.
For inclusion in comparison videos, also update the display-name table in
`scripts/make_gallery.py` and validate its rendering model.

## Default Algorithm and Accepted Decisions

### Input and Normalization

Convert source BVH coordinates from Y-up to Z-up with the `(z, x, y)` permutation.
Preserve source length units. Scale horizontal root displacement by the ratio of target
to source reference-pose horizontal toe separation. Root rotation follows the source;
joint angles and root height are optimized.

- Source ground calibration is enabled by default. Compute the minimum toe height per
  frame across the full input, take its 5th percentile as a static ground plane, then
  apply frame slicing or subsampling.
- Normalize source and target edges by their respective reference lengths, with a floor
  of 5% of each skeleton's height. Height is the maximum reference joint/body-origin z
  minus the lowest toe marker z; do not use the visual mesh bounding box.
- A body-to-ground edge has z equal to negative height. Clamp positive source components
  caused by penetration to zero before normalization. Preserve the sign of target edges.
- Compute proximity weights as `softmax(-5 * length)` using normalized source edge lengths.
  Preserve this convention, which differs from the paper's use of raw distances.
- For the source reference pose, subtract the minimum toe height, then add 5% of the resulting
  pelvis height. Target initial root height is `-1.05` times the lowest reference toe z.

### SMPL-X Input

`SMPLXSource(model_path, up_axis='z', gender=None, flat_hand_mean=True)` is the explicit
source configuration accepted by `prepare`, `run`, and `run_batch` as `smplx=`.
BVH input keeps its existing reference and arithmetic. NPZ input requires this source
object and derives its own reference from the selected body model and subject shape;
it must not use a BVH reference with different body proportions.

- Use the standard 55-joint SMPL-X hierarchy, including finger joints needed by the
  existing robot mappings. Preserve SMPL-X joint names without substituting wrists
  for the middle-finger or thumb landmarks.
- Load only `v_template`, `J_regressor`, `shapedirs`, `kintree_table`, and optional hand
  means. Project template and shape/expression bases through a sparse joint regressor
  once per model, then release vertex arrays. Cache these small, read-only joint
  bases per source instance and gender. No Torch, mesh skinning, or SMPL-X package
  import is required by the runtime.
- Joint locations follow the SMPL-X joint-regression and rigid-transform equations.
  Pose blend shapes affect skinned vertices, not these 55 articulated joint origins.
  Validate against the [official SMPL-X implementation](https://github.com/vchoutas/smplx)
  as well as independent synthetic homogeneous-transform tests.
- Support full 165-value poses and separate root/body/hand/jaw/eye axis angles.
  Do not truncate SMPL or SMPL-H vectors or silently reinterpret PCA hand coefficients.
  Absolute hand angles are the default; `flat_hand_mean=False` adds the model's learned
  hand mean. A full `poses` array takes precedence over separately supplied fields.
- Standard full models have 300 shape plus 100 expression directions; compact models
  have 10 plus 10. Validate coefficient counts instead of silently truncating shape.
  Static and per-frame shape/expression coefficients are supported. The standing
  reference uses the first frame's shape and zero expression.
- The default reference uses the same named joint rotations as the bundled standing
  BVH: both ankle X angles at -20 degrees and shoulder Z angles at -90/+90 degrees.
  `SMPLXSource.normalization_pose` is a Python-class override with axis-angle triples.
- Canonical SMPL-X coordinates are Y-up; let `P` map `(x, y, z)` to `(z, x, y)`.
  Let `C` map input world coordinates to MAHR: identity for AMASS Z-up, `P` for Y-up.
  Convert positions with `C` and root rotations with `C @ R @ P.T`, distinguishing
  the model's local axes from the motion's world axes. Applying `C @ R @ C.T` to
  Z-up AMASS would leave the canonical root tilt in the prescribed robot pose.
- Use the complete motion for ground calibration before slicing. Motion arrays carry
  their units so SMPL-X calibration reports meters while BVH retains its original units.
  Keep source-foot rotations out of the objective; they only enter source FK where
  they affect actual descendant joint positions.
- Body models are explicit user inputs, never downloaded or discovered implicitly at
  runtime. Keep local model files under ignored `assets/body_models/` or an explicitly
  selected external directory. Synthetic unit tests contain no trained model weights.
  NPZ loading disables pickle; model PKL loading is limited to explicitly selected,
  trusted body-model files. Directory model lookup prefers NPZ over PKL.

### Objective and Solver

Each frame directly optimizes graph matching, regularization, temporal terms, and
posture with bounded L-BFGS-B. Initialize the first frame from `initial_pose` and
subsequent frames from the previous accepted solution. Store source joint positions
and root rotations; full joint rotations are temporary FK data, not retained motion
state or optimization targets.

`FrameObjective` defines the numerical objective, analytic gradient, and diagnostics.
Do not duplicate loss formulas in the solver. `LossBreakdown` contains graph matching,
clipped regularization, temporal terms, and the posture prior.

| Component | Default value or behavior |
| --- | --- |
| Clipped-square regularizer | Weight `0.001`, clipping at `±0.78`, including root height |
| Non-saturating posture prior | Weight `0.003`, joints only, relative to the independent `posture_reference` |
| First-order temporal term | Weight `0.02`, scaled by `((1/120) / dt)^2` |
| Second-order temporal term | Weight `0.1`, scaled by `((1/120) / dt)^4` |
| Joint bounds | URDF limits; unbounded scalar joints retain `[-π, π]` |
| Flat-ground root-height bounds | `[0, 4]` meters |
| Inter-frame bounds | Intersection of the `50/120` step limit and velocity limits |
| Joint velocity limit | `8 rad/s` |
| Root-height velocity limit | `2 m/s` |
| Optimizer | L-BFGS-B with analytic value/gradient; declared initialization for the first frame, warm start thereafter |

Other solver defaults are defined by `Settings` in `solver.py`. URDF kinematics supports
independent scalar joints only; reject mimic and multi-DoF joints rather than silently
changing their interpretation.

Target ground-penetration penalties, inverted-foot penalties, source foot-rotation
matching, and multi-start experiments have been removed from the default pipeline.
The user considers source foot rotations potentially unreliable; do not reintroduce
them as targets. Graph ground-height matching remains active and is distinct from the
removed additional ground penalties.

### Target Terrain

`HeightMap` is the Python configuration and data object accepted by `prepare`, `run`,
and `run_batch` through `terrain=`. Omission keeps the flat-ground arithmetic and
default trajectories unchanged. PNG input uses 8-bit grayscale samples; the numerical
API also accepts arbitrary finite height arrays in meters, including negative heights.
The map center XY and base Z are configurable independently of its half extents.

- Query terrain at each ground-edge landmark's world XY, including the scaled root
  translation and root rotation. A target ground edge is still vertical:
  `height_map(world_x, world_y) - landmark_world_z`.
- Use continuous piecewise planar interpolation with the same cell diagonal as MuJoCo
  height fields. In image coordinates, the first triangle has `u + v <= 1`.
  This avoids staircase sampling and provides analytic slopes within each triangle.
  Derivatives can change at triangle boundaries; use points away from those boundaries
  for finite-difference checks. Clamp out-of-map queries and zero the derivative along
  each clamped coordinate. The rendered height field covers only the configured footprint.
- Add `dh/dx * dx/dq + dh/dy * dy/dq` to each ground-edge joint derivative. Do not treat
  sampled terrain heights as constants during optimization. Root XY and rotation remain
  prescribed by the source; only joint angles and root height are optimized.
- Keep source calibration and reference edge denominators on horizontal ground. The
  target terrain changes the target graph, not skeleton normalization or source vectors.
- Initialize root height above the first root XY's terrain sample. Root bounds are
  `[ground_at_root, ground_at_root + 4]`; the root component of clipped regularization
  uses height above that sample. Temporal terms and velocity bounds use world heights.
  Report an error if terrain bounds and inter-frame velocity bounds have no intersection.
- Export the sampled map, half extents, and origin to a compressed `.terrain.npz` sidecar.
  The trajectory JSON's optional `terrain` field is the sidecar basename. Playback reads
  that file relative to the trajectory, preserving the map after moving results. Missing
  or invalid maps must fail explicitly rather than silently render or solve flat ground.

Terrain support does not add penetration, contact, foot orientation, or balance penalties.
Validate both analytic gradients and MuJoCo ray heights on asymmetric non-square maps;
matching only PNG dimensions does not establish coordinate or interpolation consistency.
MuJoCo normalizes nonconstant height-field samples to `[0, 1]`. Preserve their actual
height span, including submillimeter spans; use a positive fallback size only for a
completely constant field, whose normalized samples remain zero.

### Multi-Motion Batches

`run_batch` owns file output and calls the in-memory pipeline one task at a time. Avoid
holding all solved trajectories in memory. Load a requested height map once and share
the read-only data across tasks. Each solve still owns its own kinematic workspace.

Directory discovery is recursive and sorted, accepts case-insensitive BVH and NPZ suffixes, and
excludes relative path components beginning with `norm_pose`. An explicitly selected
file bypasses that exclusion. Exclude `.terrain.npz` sidecars during directory scans.
Mixed-format batches share the supplied `SMPLXSource` model cache across tasks; ordinary
BVH tasks require no body model. Preserve each motion's relative path under its robot's
output directory. Preflight collisions across trajectories, statistics, and terrain
sidecars before writing any task outputs.

`--robot` remains a single-value option and can be repeated; `--robot all` expands the
preset registry. A single file and single robot retain the JSON output interface.
Directory inputs or multiple robots use batch output directories. Task statuses distinguish `succeeded`,
`unconverged` (export retained), and `failed` (exception recorded). Continue after task
exceptions, but do not swallow process interrupts. The CLI exits nonzero for failed or
unconverged tasks. Progress records are not a resume/cache mechanism.

- Each output directory has one batch writer. Only that writer updates batch records;
  rendering reads them without modifying solve metadata or recording a fixed render stride.
- Start `batch_progress.jsonl` by atomically replacing any previous journal with one
  metadata line: the initial summary with zero counters and an empty task list. Append
  and flush one JSON task record per completed task. Keep total serialization and writes
  linear in task count; do not rewrite all prior tasks after each completion.
- Atomically write `batch_summary.json` at startup and in `finally` on completion or
  catchable interruption. It retains the existing summary schema and return value.
  A forced termination can leave this snapshot incomplete; flushed journal records
  remain readable without requiring another full summary write.
- `load_batch_summary(directory)` reads metadata and completed tasks from one journal
  handle, ignoring an incomplete final line but rejecting malformed complete records.
  Never combine a journal with metadata from a separate snapshot: restarting a batch
  could otherwise mix runs. Fall back to `batch_summary.json` for existing outputs
  without a journal. A batch renderer processes the records visible when it loads them.

### Performance and State Ownership

- Reuse error, weighted-error, and edge-Jacobian buffers within each frame, normalizing
  in place. Gradients returned to the optimizer must not alias buffers overwritten by later calls.
- Compute only positional MuJoCo Jacobians and write them directly into destination arrays.
  Avoid unused rotational Jacobians and per-landmark copies of temporary Jacobians.
- Cache ground-edge indices and their source-node indices when constructing the graph.
- Evaluate diagnostics at the state accepted by the optimizer. The last line-search
  evaluation is not necessarily the accepted state.
- `retarget_all.py` validates in-memory results. Do not prepare the input a second time
  or reload just-exported JSON merely to validate it.

## Output Interface

`RetargetResult` contains `prepared`, `trajectory`, and `report`. To solve repeatedly,
call `prepare(...)` once and then `solve(data, settings)`. Export through
`save_result(result, path)`.

`Trajectory.coordinates` has shape `[frames, joints + 1]`, with root height in the last
column. `joint_positions` exposes the joint portion. JSON contains `fps`, `joint_names`,
and per-frame `RootPos`, `RootOrient`, and `JointPos`. Root quaternions use **xyzw**;
the renderer converts them to MuJoCo's wxyz convention. Preserve named joints and the
quaternion convention when exporting.

Rendering samples every fourth frame by default and adjusts playback FPS to preserve
motion speed. It performs kinematic playback without dynamics or trajectory smoothing.
Dark-material enhancement for H1 and GR1 affects display only.
For bandwidth-limited previews, `render_motion.py --fps` selects the nearest source
frame at each output timestamp; it does not interpolate or smooth the trajectory.
The frame count is `ceil(source_frames * output_fps / source_fps)`, so duration is
rounded up by less than one output frame. `--fps` and `--stride` are mutually exclusive.
`--bitrate` is in kbit/s and sets H.264 target/max rate with a two-second VBV buffer;
omission preserves CRF 20. Use even output dimensions for yuv420p and `+faststart`
for progressive MP4 playback. Verify exact frame rate, duration, file size, and bitrate
using FFprobe. Optional ground grids and camera distance affect presentation only.

## Validation and Recorded Results

From the repository root, in the environment documented in README:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests scripts
git diff --check
```

The tests cover FK and objective gradients, loss decomposition, buffer reuse,
configuration and pose isolation, in-memory results and export, temporal velocity bounds,
normalization, source ground calibration, and independence from source foot-rotation
channels, the completeness of bundled model assets, terrain gradients and rendered
surfaces (including submillimeter spans), portable terrain export, batch discovery,
task failure isolation, live and interrupted batch records, linear serialization volume,
and rendering without overwriting concurrent batch progress. SMPL-X tests cover independent
joint regression and homogeneous FK, coordinate conventions, shape-dependent reference
poses, parameter validation, model reuse, mixed-format batches, and all eight robots.
Run checks appropriate to the change.
For documentation changes, verify links,
arguments, and executable examples.

For behavioral changes to configuration, graphs, kinematics, objectives, or solvers, run
both motions on all eight robots as documented in README: 6552 frames in total. To compare
against saved baselines, add `--reference` to the batch command:

```bash
python scripts/retarget_all.py \
  --motion assets/motions/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh \
  --output results/check_dance --reference results/python_config_dance
```

Baseline directories are not distributed test assets. If unavailable, generate them
with the pre-change version in the same environment. Save post-change results in a
new directory to avoid overwriting the baseline.

Orientation initialization was evaluated and removed on 2026-11-05. On the two CMU
subject 85 clips, its final orientation RMS improvements were about 0.01 to 0.13
degrees, while a controlled X1 Dance benchmark approximately doubled solve time.
Graph refinement largely undid the initialization benefit. Keep the runtime focused
on direct graph optimization; do not retain orientation mappings, angular Jacobians,
initialization switches, or stage-specific statistics. Historical comparisons remain
in ignored result artifacts.

Single-stage validation after removal on 2026-11-05 passed all 44 tests. Both bundled
BVHs and the complete CMU `85_03` and `85_12` clips passed on all eight robots:
68504 robot frames converged and matched the saved graph-only trajectories exactly,
including root positions and quaternions. Joint/root bounds and velocity checks
passed; first/middle/last analytic-gradient checks had maximum error `3.06e-9`.
Settings and output statistics contain no orientation-initialization fields.

Local validation recorded on 2026-11-04:

- SMPL-X support passed all 43 tests, compilation, and whitespace checks. Using a
  user-supplied female SMPL-X model and a 5171-frame AMASS DanceDB motion, the first
  55 joint positions matched the official float64 SMPL-X implementation within
  `2.47e-9` meters at seven sampled frames. All eight robots retargeted the first
  500 frames (4000 output frames), with no unconverged frames and all joint/root
  bounds and velocity checks passing. Analytic gradient checks at the first, middle,
  and last output frames had a maximum error of `1.91e-9`.
  The two complete bundled BVHs still matched their preceding baselines exactly
  across 6552 output frames. G1 and X1 SMPL-X previews at 320-by-240 and 30 Hz passed
  FFprobe and full decoding checks. Results are under `results/smplx_validation/`;
  body-model weights and motion inputs are local validation assets, not test dependencies.
- Batch record ownership, incremental progress, and small terrain spans passed all
  34 tests. With solving and file I/O mocked, 500 and 1000 tasks serialized 0.238 MB
  and 0.475 MB respectively, including the journal and snapshots; the preceding
  full-rewrite implementation serialized 33.510 MB and 133.771 MB for the same records.
  These are record-volume checks, not end-to-end retargeting timings.
  G1 and X1 each retargeted eight frames with a submillimeter ramp and exported
  320-by-240 videos. FFprobe and complete decoding passed; both batch record files
  retained identical hashes across rendering. Records are in `review_fixes_smoke`.
- Terrain and multi-motion batch support passed 29 tests using the validated core
  environment and Pillow 12.2.0. Editable installation through uv and the installed CLI
  also passed, including execution outside the repository and frame subsampling.
- `terrain_flat_dance` and `terrain_flat_smpl` matched `python_config_dance` and
  `python_config_smpl` exactly across 6552 frames, with no unconverged frames.
- `terrain_motion_batch` processed both complete motions on all eight robots using
  `assets/terrains/ramp.png`, half extents `(4, 4)`, and elevation `0.4`. All 16 tasks
  and 6552 frames converged; joint and world-velocity bounds passed. Finite differences
  at the first, middle, and last frames had a maximum gradient error of `3.15e-9`.
  All 16 terrain videos passed FFprobe checks for dimensions, frame counts, and source
  playback speed. Records are in that directory's `validation.json` and
  `render_validation.json`.
- With one warmup and seven alternating X1 Dance solves, flat-ground median time was
  1.558 s before terrain support and 1.566 s after it (about 0.5% difference), with
  identical trajectories and no unconverged frames. Video export ran concurrently;
  treat these timings as a local regression check. Raw timings are in
  `results/terrain_flat_performance.json`.
- `structured_api_dance` and `structured_api_smpl`, compared with `python_config_dance`
  and `python_config_smpl`, had a maximum difference of zero across all joints, root
  positions, and root quaternions. All frames converged; bounds, velocity, and analytic
  gradient checks passed.
- For X1 Dance, warm up both versions and alternate seven runs of the 500-frame sequence.
  Median solve time fell from 1.665 s to 1.568 s, a 5.9% reduction. At a fixed first-frame
  state, 10000 objective/gradient calls per round reduced median time per call from
  80.4 μs to 72.4 μs, a 10.0% reduction. Timing excludes export; trajectories were identical.
- After renaming the module and command to `mahr`, installation, CLI, and single-robot
  video-export checks passed. The 6552 frames in `docs_rename_dance` and `docs_rename_smpl`
  still had zero maximum difference from their original baselines.
- With bundled assets, an isolated checkout passed all 19 tests and single-robot CLI
  execution. The 6552 frames in `self_contained_dance` and `self_contained_smpl` had
  zero failed frames and zero difference from the preceding baselines. All eight models
  also passed a three-frame video export and FFprobe check using only that checkout's
  assets; records are in `results/self_contained_render/checks.json`.
- Raw timings are in `results/structured_api_benchmark/summary.json`. These measurements
  describe that local comparison, not a guarantee for other robots or hardware.

Performance changes should be measured against the pre-change version on the same
machine, dependencies, and input, alternating versions. Record warmup, repetitions,
medians, trajectory differences, and failed frames. Changes to floating-point operation
order can affect sequential optimization, so validate complete motions.

The README hero image is the approved MAHR teaser stored at `images/mahr-teaser.png`.
Keep it unchanged unless an image replacement is requested. It is an overview
illustration; generated comparison videos belong in `results/`.
Use Markdown image syntax with a repository-relative path and a full-size image link. Verify
the rendered image loads after markup changes; file existence and matching hashes
alone do not verify preview behavior. Keep machine-specific paths out of README.

## Git Commit Message Conventions

Use Conventional Commits for new commits. Do not rewrite existing commit messages.

```text
type(scope): summary

Optional body explaining why the change is needed and its effect.

Validation: checks actually run and their results.
```

### Subject

- `type` is required and uses one of the lowercase types below. `scope` is optional
  and identifies the main affected module.
- Write a concise English imperative summary, starting with a lowercase word and ending
  without a period. Limit the complete subject to 72 characters.
- Describe the final change, not conversation history or abandoned approaches. Avoid
  vague summaries such as “update code” or “fix stuff”.
- Each commit should have one coherent purpose. Related implementation, documentation,
  and tests may belong in the same commit.

| Type | Purpose |
| --- | --- |
| `feat` | Add functionality or supported use cases |
| `fix` | Correct faulty behavior |
| `refactor` | Restructure code while preserving intended behavior |
| `perf` | Improve performance |
| `docs` | Update documentation, including AGENTS.md |
| `test` | Independently change tests, validation tools, or test data |
| `build` | Change dependencies, lockfiles, packaging, or build configuration |
| `ci` | Change continuous integration workflows |
| `chore` | Repository maintenance outside the categories above |
| `revert` | Revert an earlier commit |

Suggested scopes: `config`, `motion`, `graph`, `kinematics`, `objective`, `solver`,
`pipeline`, `trajectory`, `cli`, and `render`. Omit the scope for cross-module changes
without a clear primary module. Robot-specific changes may use the CLI name, such as
`g1` or `x1`.

### Body, Validation, and Breaking Changes

- Separate the subject and body with a blank line. Simple changes may omit an explanatory
  body. For complex changes, explain the problem, resulting behavior, and necessary
  tradeoffs rather than repeating the file list.
- Use `Validation:` to record checks actually performed. Documentation changes may report
  link and command checks; performance changes should report benchmark scope and results.
  State why a relevant check was not run rather than copying an outdated result.
- For incompatible public API, CLI, or output-format changes, append `!` after the type
  or scope and add a `BREAKING CHANGE:` footer describing the impact and migration.
- Use `revert: ...` for reversions, naming the reverted commit's full hash and the reason
  in the body.

Subject examples:

```text
docs: clarify repository purpose and reproduction steps
refactor(config): pair source and robot landmarks
perf(kinematics): avoid unused rotational Jacobians
build: align runtime dependency versions
```

Breaking-change example:

```text
refactor(pipeline)!: return retargeting results in memory

Separate computation from export so callers can inspect and reuse trajectories.

Validation: 18 tests passed; 6552 regression frames were unchanged.

BREAKING CHANGE: run() returns RetargetResult and no longer accepts an output
path. Call save_result(result, path) explicitly to export files.
```

Validation statements in these examples illustrate the format. Actual commits must
report checks performed for the current change.
