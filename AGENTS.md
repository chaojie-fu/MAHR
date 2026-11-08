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

The runtime depends on NumPy, SciPy, MuJoCo, Pillow (PNG height-map loading), and
OSQP (conditional trajectory repair).
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

The validated environment is Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3, MuJoCo 3.9.0,
and OSQP 1.1.1. README uses these dependency versions for reproduction. The current
`uv.lock` selects MuJoCo 3.17.0 and OSQP 1.1.3,
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
| `rendering.py` | Shared playback environment, matching terrain geometry, and compiled-frame mesh updates |
| `normalization.py` | Skeleton-height estimation and edge denominator floors |
| `kinematics.py` | Mesh-free MuJoCo FK, body origins, and landmark Jacobians |
| `graph.py` | Graph topology, ground-height relations, proximity weights, and edge derivatives |
| `ground.py` | Absolute root correction bounds, position prior, and exact flat-ground elimination |
| `pose.py` | Semantic bone directions and the independent first-stage pose objective |
| `objective.py` | Losses, analytic gradients, and a shared `LossBreakdown` |
| `quality.py` | `RecoverySettings` and source-aware checks of a single output trajectory |
| `recovery.py` | Triggered frame restarts, repair-window selection, and acceptance diagnostics |
| `window.py` | Sparse Gauss–Newton trajectory repair using the shared frame objective |
| `solver.py` | `Settings`, sequential state, optimizer calls, and recovery integration |
| `pipeline.py` | Preparation, solving, result composition, and aggregate statistics |
| `batch.py` | Motion discovery, sequential task execution, output paths, and failure manifests |
| `trajectory.py` | In-memory trajectories, output format, and explicit export |
| `cli.py` | Command-line arguments and execution entry point |

`scripts/` contains four tools for the default pipeline: `retarget_all.py`,
`render_motion.py`, `render_all.py`, and `make_gallery.py`. Tests use the standard-library
unittest framework. Generated artifacts belong in the Git-ignored `results/` directory.

`PreparedMotion` contains the robot, graph, normalization scales, source vectors, motion,
height-based horizontal scale ratio, ground XY targets, pose segments and source direction targets, initial state, posture reference,
asset path, and metadata.
Preparation explicitly queries reference positions rather than reading MuJoCo state
that depends on the previous FK call.

Each prepared object owns a mutable MuJoCo workspace and supports sequential reuse.
Independent concurrent tasks must prepare their own workspaces. The kinematics object
does not own the posture prior. Repeated `solve()` calls must start from the declared
initial state, independently of the last FK evaluation.

## Python Configuration Conventions

- Each `landmark_pairs` row declares `(source joint, target body)`. Names on each side
  must be unique. Do not restore two independent lists that rely on positional alignment.
- Ground relations are controlled by `include_ground`, not ordinary landmarks in
  the mapping. Each relation contributes both vertical and world-XY residuals.
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
Preserve source length units. Use the ratio of target to source reference skeleton
heights to scale both source world root XY and source world landmark XY. Root rotation
follows the source; optimize joint angles, world root height, and a bounded XY
correction to the scaled root path. Toe spacing no longer determines horizontal scale.

- Source ground calibration is enabled by default. Compute the minimum toe height per
  frame across the full input, take its 5th percentile as a static ground plane, then
  apply frame slicing or subsampling.
- Normalize source and target edges by their respective reference lengths, with a floor
  of 5% of each skeleton's height. Height is the maximum reference joint/body-origin z
  minus the lowest toe marker z; do not use the visual mesh bounding box.
- A body-to-ground edge has z equal to negative height. Clamp positive source components
  caused by penetration to zero before normalization. Preserve the sign of target edges.
- Compute proximity weights as `softmax(-beta * length)` using normalized source edge
  lengths. The default `Settings.proximity_coefficient` is 1, reduced from 5 to avoid
  excessive differences between ground and body-edge weights. Zero gives uniform weights.
  Keep normalized lengths rather than replacing them with raw distances.
- For the source reference pose, subtract the minimum toe height, then add 5% of the resulting
  pelvis height. Target initial root height is `-1.05` times the lowest reference toe z.

### Ground Grid and Root XY Correction

The accepted default is height scaling plus fixed-world ground-grid relations and
root XY correction. Four equally weighted fixed lattice corners around each source
landmark reduce exactly to a world-XY point residual: corresponding source and target
anchors cancel. Evaluate that reduction directly; grid density does not affect this
objective, and the renderer's checker size is unrelated to optimization.

- For each existing ground edge, add the XY displacement between its robot landmark
  and the height-scaled source world landmark. Divide by 5% of target height and use
  that edge's existing proximity weight (multiplier 1). Preserve body edges, vertical
  ground residuals, and their normalization. Source drift remains in these targets.
- Root correction has per-axis absolute bounds of 20% of target height and a position
  penalty of `0.1 * ||correction / height||^2`. There are no correction speed bounds,
  velocity penalties, or acceleration penalties. The user selected this relaxation
  after complete-trajectory ablations demonstrated that correction history contributed
  to folded-leg and inverted-torso solutions. These bounds concern the correction,
  not overall locomotion. Joint and root-Z continuity uses soft temporal penalties,
  without hard inter-frame speed or step bounds in the normal frame solve.
  Conditional repair windows apply the quality thresholds described below.
- On flat or constant-height terrain, solve the separable box-constrained XY quadratic
  analytically during each objective evaluation. Use its envelope gradient for joints
  and height, retaining the original L-BFGS-B dimension and one FK per evaluation.
  On varying terrain, jointly optimize XY because terrain heights depend on it.
- `FrameObjective` accepts full internal states `[joints, world Z, correction X/Y]`.
  A shorter `[joints, world Z]` state requests exact elimination on horizontal ground.
  Histories use full states; original temporal terms use joints/Z and the posture prior
  uses joints only.
- Export corrected world root positions. `Trajectory.coordinates` retains its existing
  joint-plus-height layout; XY correction is not a robot joint or a prescribed rotation.
  CLI, batch, and Python entry points share this default without an extra switch.
  Low-level callers of `Retargeter` must provide `body_height`, compiled `pose`,
  `root_xy`, `ground_targets`, and per-frame `pose_targets`; `solve` returns `(coordinates, corrections, stats)`.
  `Trajectory.from_solution` now takes solved world root XY rather than a scale ratio.

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

Each frame first fits source body pose, then refines the graph while retaining the
source-pose objective. The first stage uses bounded L-BFGS-B over URDF joint ranges,
independently of graph-result history. Warm-start it from the previous first-stage
solution; when its objective exceeds `pose_restart_threshold=0.1`, also try the
declared initial pose and retain the lower-loss result. The first frame uses that
initial pose. A `1e-4` squared distance to `posture_reference` breaks joint-coordinate
ambiguities in this stage. The first-stage pose is an optimizer seed, not an exported
trajectory, and does not have inter-frame velocity bounds.

`RobotConfig.pose_pairs` declares twelve directed semantic segments by default: torso,
shoulder axis, two pelvis-to-knee proxies, two shins, two upper arms, two elbow-to-hand
segments, and two ankle-to-toe directions. Both BVH and SMPL-X use actual source joint
positions in the prescribed root frame. Normalize each segment to remove bone length;
minimize the mean squared difference of unit directions. Do not fit away root heading
or use source foot-rotation channels as orientation targets. Missing or degenerate
source segments fail explicitly; custom mappings must supply corresponding pose pairs.

The second stage minimizes graph matching, ground XY relations, root correction,
regularization, temporal terms, the static posture prior, and the same direction
objective with `source_pose_weight=1`. Share its FK and positional Jacobian with
graph evaluation. Use bounded L-BFGS-B with analytic XY elimination on horizontal
ground, and SLSQP with full derivatives on varying terrain. Initialize each frame
from the complete first-stage joint solution and the preceding final root height.
Do not project the pose into a box around the previous graph result. Preserve URDF
joint limits, root-height/terrain-clearance limits, and absolute XY correction bounds.
Continuity uses the original first- and second-order temporal penalties, with their
frame-rate scaling. The normal frame solve has no hard speed or step bounds;
conditional trajectory repair adds source-aware joint bounds within suspect windows.
The user selected this default after the Berkeley 85_04 ground-reaching diagnosis:
velocity-box projection discarded a better first-stage arm pose and perpetuated a
poor local solution. Larger frame-to-frame changes are now possible and must be
measured rather than treated as violations of the removed bounds.
Store source joint positions and root rotations; full source joint rotations remain
temporary FK data, not retained motion state or optimization targets.

`FrameObjective` defines the numerical objective, analytic gradient, and diagnostics.
Do not duplicate loss formulas in the solver. `LossBreakdown` contains graph matching,
clipped regularization, temporal terms, the static posture prior, ground XY, root XY,
and retained source-pose matching. Per-frame reports include first-stage loss,
convergence, evaluation count, timing, and restart flag. A failed
first stage marks the frame unsuccessful even if graph refinement converges.

| Component | Default value or behavior |
| --- | --- |
| Clipped-square regularizer | Weight `0.001`, clipping at `±0.78`, including root height |
| Static non-saturating posture prior | Weight `0.003`, joints only, relative to the independent `posture_reference` |
| Retained source pose | Weight `1`, mean squared unit-direction error across the configured segments |
| Proximity softmax coefficient | `1`, applied to normalized source edge lengths |
| First-order temporal term | Weight `0.02`, scaled by `((1/120) / dt)^2` |
| Second-order temporal term | Weight `0.1`, scaled by `((1/120) / dt)^4` |
| Joint bounds | URDF limits; unbounded scalar joints retain `[-π, π]` |
| Flat-ground root-height bounds | `[0, 4]` meters |
| Inter-frame bounds | None in the normal frame solve; source-aware joint velocity/acceleration bounds during conditional window repair |
| Optimizer | Independent L-BFGS-B pose fit, then pose-guided L-BFGS-B on horizontal ground or SLSQP on varying terrain |

Other solver defaults are defined by `Settings` in `solver.py`. URDF kinematics supports
independent scalar joints only; reject mimic and multi-DoF joints rather than silently
changing their interpretation.

`Settings.step_limit`, `Settings.max_joint_velocity`, and `Settings.max_root_velocity`
were removed with the velocity box. Remove these keywords from Python callers;
`temporal_weight` and `acceleration_weight` control soft continuity. Per-frame
`pose_seed_projection` was removed because the first-stage pose is no longer clipped.
The report's `max_joint_velocity` and `max_root_velocity` are measured speeds, not
enforced limits; the latter measures world-Z speed. Retain these diagnostics when
comparing complete trajectories.

### Automatic Quality Checks and Conditional Recovery

Recovery is enabled through `Settings.recovery`, a `RecoverySettings` dataclass.
Use `Settings(recovery=RecoverySettings(enabled=False))` for a controlled comparison.
Detection must use the unique normal graph result, source motion, accepted history,
and numerical status. Do not solve an alternative merely to decide whether to trigger.
The independent first-stage pose fitting described above remains unchanged.

- Track raw scalar joint differences, not modulo-2π differences, because an apparent
  wrap may still require a large commanded joint motion. The default speed threshold
  is `max(16, 2 * (source_segment_speed + root_angular_speed))` radians/second.
  Source segments relax thresholds only for their ancestor joints in the robot.
  Acceleration uses `max(2000, 2 * (source_direction_second_difference / dt^2
  + root_speed_change / dt))` radians/second². Slide coordinates use body height as
  their unit scale. These are anomaly heuristics, not actuator limits.
- Flag a segment error above 90° when sustained for 0.1 seconds or when its increase
  exceeds 30° plus the source's angular change. Failed optimization and infeasibility
  are independent signals. A flag indicates suspicion, not proof of a bad local minimum;
  smooth inaccurate solutions below these thresholds can remain undetected.
- Only after a trigger, try up to three distinct seeds: the latest reliable output,
  neutral coordinates of the affected chain, and reflection of that chain about its
  joint-range midpoint. Pure persistent-pose triggers have a 0.1-second restart cooldown.
  Keep the original objective, bounds, and independently generated pose-stage history.
- Restart acceptance requires fewer reasons or a meaningful severity reduction,
  no segment error increase above 20° (`pose_regression`), and no graph, ground-XY, or total-loss increase
  above 10% plus an absolute `1e-4` tolerance. A lower total loss alone is insufficient.
- After the sequential pass, prioritize remaining suspicious frames by severity.
  Start with a 0.5-second window, extending unreliable boundaries up to 2 seconds.
  Fix two endpoint frames on each side that joins unchanged data. Optimize joints,
  world Z, and XY correction together, preserving terrain clearance and static bounds.
  Apply the detector's source-aware joint speed and acceleration bounds inside these
  windows. Do not add root speed limits, XY temporal penalties, foot-inversion terms,
  penetration penalties, or automatic changes to objective weights.
- `FrameObjective.linearize` supplies the exact residual/Jacobian for sparse
  Gauss–Newton QPs solved by OSQP. Reuse the frame objective's FK and graph derivatives;
  normal frame solves need not assemble the residual Jacobian. Window temporal terms
  use the same weights and frame-rate scaling as the original sequence objective.
- If pose or loss guards reject a window, try a linear joint bridge between its
  endpoints. If both seeds fail those guards, stop retrying that interval. Expand
  windows for solver failures or absent quality improvement. Limit each trajectory to
  64 window solves and each solve to 20 iterations. An iteration-limited candidate can
  be accepted only after feasibility and quality checks; this is not proof of optimality.
- Window acceptance checks each segment's maximum and mean error (10° allowance,
  controlled independently by `window_pose_regression`),
  average graph and ground-XY losses (10% plus `1e-4`), total original objective,
  fewer suspect frames, and no new trigger reasons outside the window. Window-wide
  pose guards can permit redistribution of error between frames; inspect local motion
  as well as summary statistics when evaluating the method.
- Record initial triggers, restart attempts, reliable-seed frame, every accepted or
  rejected window, time spent, remaining flags, and budget exhaustion. Mark only frames
  that actually changed as repaired. Recompute reported objective components using the
  final repaired history. Numerical convergence and quality are separate: `failed 0`
  can coexist with unresolved suspicious frames. Never silently claim all jumps fixed.

On 2026-11-08 the user selected a 64-window budget and a 20° single-frame restart
pose allowance. Keep the window pose allowance at 10°; the single-frame allowance
is an increase in segment error relative to the original candidate, not a joint-step
limit or an absolute pose-error threshold. Both settings use radians in Python.
Previously `pose_regression` also controlled window acceptance; callers customizing
both stages should now explicitly set both `pose_regression` and
`window_pose_regression`. Earlier validation records below describe the prior defaults.

Target ground-penetration penalties, inverted-foot penalties, and source foot-rotation
matching remain absent from the default pipeline.
The user considers source foot rotations potentially unreliable; do not reintroduce
them as targets. Graph ground-height matching remains active and is distinct from the
removed additional ground penalties.

### Target Terrain

`HeightMap` is the Python configuration and data object accepted by `prepare`, `run`,
and `run_batch` through `terrain=`. Omission uses the default flat-ground solve.
PNG input uses 8-bit grayscale samples; the numerical
API also accepts arbitrary finite height arrays in meters, including negative heights.
The map center XY and base Z are configurable independently of its half extents.

- Query terrain at each ground-edge landmark's world XY, including the scaled root
  translation, optimized XY correction, and prescribed root rotation. Its vertical component is:
  `height_map(world_x, world_y) - landmark_world_z`.
- Use continuous piecewise planar interpolation with the same cell diagonal as MuJoCo
  height fields. In image coordinates, the first triangle has `u + v <= 1`.
  This avoids staircase sampling and provides analytic slopes within each triangle.
  Derivatives can change at triangle boundaries; use points away from those boundaries
  for finite-difference checks. Clamp out-of-map queries and zero the derivative along
  each clamped coordinate. The rendered height field covers only the configured footprint.
- Add `dh/dx * dx/dq + dh/dy * dy/dq` to each ground-edge joint derivative. Do not treat
  sampled terrain heights as constants during optimization. Include terrain slopes in
  the correction-XY derivatives as well. Only root rotation remains prescribed.
- Keep source calibration and reference edge denominators on horizontal ground. The
  target terrain changes the target graph, not skeleton normalization or source vectors.
- Initialize root height above the first root XY's terrain sample. Root bounds are
  `[ground_at_root, ground_at_root + 4]`; the root component of clipped regularization
  uses height above that sample. Temporal terms use world heights.
  For varying terrain, constrain this clearance with analytic nonlinear inequalities
  at the corrected root XY, within the global terrain height bounds. A box at the
  uncorrected root would impose the wrong constraint.
  Reject a returned state that violates clearance; retain optimizer failure diagnostics.
  Constant maps use the same reduced solve as flat ground (zero maps are identical).
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
Use `rendering.add_environment` for a shared white skybox, gray/white checker ground,
and lighting in robot and source previews. The checker is enabled by default;
`--no-ground-grid` selects a uniform gray floor. Keep camera and environment settings
identical across comparison panels.

Animated source meshes must preserve MuJoCo's compiled geom and inertial transforms.
Use `rendering.set_mesh_vertices` to convert body-frame vertices and normals into the
compiled mesh frame, then upload the mesh. Do not reset `geom_pos`/`geom_quat` after
compilation: `geom_sameframe` can still refer to the original inertial frame. This
previously rotated the displayed CMU 85_03 source by approximately 89.75 degrees and
offset it, despite correct pre-render SMPL-X joint positions. Validate final rendered
world vertices and normals, not only the skinning output. Robot trajectories, source
coordinate conversion, and optimization metrics were unaffected by this display bug.

For bandwidth-limited previews, `render_motion.py --fps` selects the nearest source
frame at each output timestamp; it does not interpolate or smooth the trajectory.
The frame count is `ceil(source_frames * output_fps / source_fps)`, so duration is
rounded up by less than one output frame. `--fps` and `--stride` are mutually exclusive.
`--bitrate` is in kbit/s and sets H.264 target/max rate with a two-second VBV buffer;
omission preserves CRF 20. Use even output dimensions for yuv420p and `+faststart`
for progressive MP4 playback. Verify exact frame rate, duration, file size, and bitrate
using FFprobe. Optional ground grids and camera distance affect presentation only.

## Validation and Recorded Results

### Soft continuity default selected on 2026-11-06

The user selected removal of the final-stage inter-frame speed bounds while keeping
temporal penalties and first-stage pose initialization. The Berkeley 85_04 diagnosis
showed that clipping a better arm pose into the previous-frame velocity box preserved
an inferior local solution. The default now seeds graph refinement with the complete
first-stage joint pose. Absolute joint, root-clearance, and XY correction bounds remain.

All 60 unit tests, compilation, and whitespace checks passed. Complete 85_04 on all
eight robots, plus both bundled BVHs on flat ground and the documented ramp, covered
40 trajectories and 33544 frames with no first- or second-stage convergence failures.
Static bounds and first/middle/last analytic gradients passed; maximum sampled
gradient error was `1.72e-9`. Records are in `results/soft_continuity_20261106/`.

In the complete Berkeley 85_04 trajectory, left-hand marker height at 14.5 seconds
changed from 35.28 to 9.87 cm, and mean body-direction RMS changed from 23.75 to
20.68 degrees. Seven of eight robots improved this mean on 85_04; G1 worsened from
9.76 to 10.26 degrees. These are complete-trajectory results, unlike the earlier
single-frame diagnostic. They do not establish improvements on all CMU motions.

Soft penalties do not guarantee small joint changes. Berkeley has a 135-degree
left-shoulder-pitch change at 14.1083 seconds; 36 frame transitions contain a joint
change above 20 degrees. H1, N1, X1, Atlas, and iCub also have large changes on 85_04.
Do not describe this change as eliminating discontinuities or producing executable
robot motion. Preserve these measured tradeoffs when presenting the selected default.

### Initial pose-guided default on 2026-11-06, before velocity-bound removal

The user requested pose initialization with a retained source-pose term, relaxed
root-XY history, and less extreme proximity weights. The selected coefficients are
`source_pose_weight=1` and `proximity_coefficient=1`. Complete X1 85_04, G1 85_12,
and GR1 85_07 comparisons covered softmax coefficients 0.5, 1, 2, and 5 and retained
pose weights 0, 0.3, 1, and 3. Weight zero keeps initialization and tests removal of
the retained second-stage objective. Smaller beta and stronger pose weights generally
improve orientation but can increase support sliding; do not report a uniform gain.

All 14 CMU subject 85 motions on all eight robots completed both stages: 191280
frames, no convergence failures. Joint/Z bounds and speeds, absolute XY correction
bounds, finite outputs, and first/middle/last full gradients passed; maximum sampled
gradient error was `8.97e-9`. Both complete bundled BVHs added 6552 flat-ground and
6552 ramp frames with all eight robots, no failures, and gradient error below `1.56e-8`.
All 59 unit tests, compilation, and whitespace checks passed. Runtime snapshots,
parameter comparisons, complete trajectories, diagnostics, and 14 synchronized
30-Hz comparison clips are under `results/pose_guided_20261106/`.

Every CMU motion/robot pair improved mean body-direction RMS. The frame-weighted
mean changed from 20.84 to 12.79 degrees; frames above 60 degrees changed from 7438
to 284. Ankle-to-toe direction errors above 90 degrees changed from 2981 frames to
zero. These positional metrics cannot detect rotation about the ankle-to-toe axis.
Atlas 85_14 retains 65.52 degrees at its original selected peak, and four trajectories
worsen body-error p95, particularly iCub 85_05 (26.38 to 45.83 degrees).

Source-only support masks match the preceding evaluation definition. Aggregate
toe-marker sliding RMS changed from 0.1458 to 0.1298 m/s, but per-robot aggregates
worsened for GR1, N1, and X1, and sliding increased in 69 motion/robot pairs. Preserve
these tradeoffs when reporting the new default. The result is not a guarantee of
dynamic feasibility or elimination of every local solution.

With one warmup and seven alternating isolated X1 Dance solves (500 frames,
OpenBLAS=1, no concurrent rendering), median solve time changed from 3.793 to
4.449 seconds (+17.3%). This includes both stages but excludes preparation/export.

### Recovery default update validation (2026-11-08)

The 64-window budget and 20° frame-restart allowance passed 72 unit tests. Tests
explicitly accept a 15° frame pose tradeoff while rejecting a 15° window pose
regression under the unchanged 10° window allowance. Complete trajectory validation
covers both bundled BVHs on all eight robots, CMU 85_14 on all eight robots,
Berkeley 85_03, and Atlas 85_12: 26 runs, 38048 frames, no numerical failures,
and finite states within static bounds. The 16 BVH runs have no remaining flags.

Across these runs, flags decrease from 477 under the previous defaults to 60.
Berkeley 85_03 decreases from 16 to 0, Atlas 85_12 from 210 to 0, and Atlas 85_14
from 120 to 0. Remaining flags are in 85_14: Berkeley 35, iCub 20, and N1 5.
None of these runs exhausts the new budget. Rejected windows, including pose-guard
rejections, still prevent recovery; do not interpret the new defaults as a continuity
guarantee or generalize this selected regression to all fourteen CMU 85 clips.

For N1 85_14, a fixed-history replay at zero-based frame 2556 accepts a restart
with a maximum segment-error increase of 11.23° under the new 20° allowance.
The selected joint step decreases from 312.06° to 2.24° and loss from 4.76968 to
0.77055. The full rerun also has a 2.24° step at that frame, but a 308.92° step
remains at frame 2562; Berkeley 85_14 retains a 135° step. Zero numerical failures
and fewer flags do not imply that all severe jumps are fixed.

Evidence is under `results/recovery_defaults_20261108/`: `REPORT.md`, `summary.json`,
`guard_probe.json`, `tests.log`, `runtime_hashes.json`, and complete per-robot
trajectory/statistics files under `motions/`. These are ignored validation artifacts.

### Automatic recovery validation (2026-11-06)

The current recovery implementation passed 71 unit tests, including single-output
triggers, source-motion compensation, restart branch exploration, window rejection
despite lower total loss, terrain clearance, and reported losses after history repair.
Exact residual/Jacobian checks cover all eight robots on flat and varying terrain.
An unflagged clip produces exactly the same trajectory with recovery enabled or disabled.

Full-motion validation covers both bundled BVHs on flat ground and a ramp, plus complete
SMPL-X CMU 85_04, for all eight robots: 40 runs and 33544 output frames. All numerical
solves and static/terrain bounds passed. Reconstructing every exported state reproduced
all quality flags and objective components, with maximum loss discrepancy `4.45e-16`.
All 32 BVH runs have zero remaining flags. CMU 85_04 retains the following results:

| Robot | Flagged frames, before → after | Maximum joint step in degrees, before → after |
| --- | ---: | ---: |
| G1 | 58 → 0 | 24.56 → 18.33 |
| H1 | 126 → 8 | 118.63 → 118.63 |
| GR1 | 34 → 0 | 17.66 → 14.92 |
| N1 | 146 → 59 | 297.42 → 125.85 |
| X1 | 157 → 0 | 140.79 → 18.40 |
| Berkeley Humanoid Lite | 185 → 71 | 135.00 → 135.00 |
| Atlas | 263 → 89 | 63.61 → 28.43 |
| iCub | 106 → 2 | 83.22 → 21.13 |

Berkeley's selected 14.108-second transition changes from 135.00° to 3.98°, while
left-hand height at 14.5 seconds remains 9.87 cm. Windows near 16.97 and 18.98 seconds
are rejected for pose regression, so a later 135° jump remains. N1's mean body-direction
RMS worsens from 15.68° to 16.23° despite fewer discontinuities. The other seven robots
improve this metric. These outcomes show partial recovery, not global continuity or
proof of a globally optimal joint sequence. Zero flags is not zero pose error.

With one warmup and seven alternating isolated G1 Dance solves (500 frames,
OpenBLAS/OMP=1, no concurrent rendering), median solve time is 4.430 seconds before
this change and 4.452 seconds with recovery (+0.5% in this local measurement), with
bit-identical trajectories and no triggered retries. Complete 85_04 solves with
recovery take 36.9–68.8 seconds in the eight-worker validation batch; this is not an
isolated performance comparison. Evidence is under `results/automatic_recovery_20261106/`:
`final.json`, `audit.json`, `benchmark.json`, `runtime_hashes.json`, and per-robot
trajectory/statistics files. Results are ignored artifacts, not runtime dependencies.

### Full CMU subject 85 recovery audit (2026-11-06)

The subsequent audit reran all fourteen CMU subject 85 motions on eight robots, both
with default recovery and with recovery disabled: 224 trajectories, 382560 robot frames,
native 120 Hz, and no numerical failures. Runtime/defaults were not changed. Exported
arrays and all quality reasons reproduced; sampled final-history losses differed by
at most `8.89e-16`. Evidence is under `results/cmu85_recovery_audit_20261106/`.

Default recovery reduces flagged frames from 6563 to 1653 and transitions above a raw
20° joint step from 884 to 157. Flags remain in 37/112 default trajectories. The largest
step does not improve (309.61° to 312.06°). Frame-weighted body RMS improves only from
11.57° to 11.46°, with aggregate regressions for H1 and N1. Of 417 remaining frames
above 45° body RMS, 104 are unflagged. These audit thresholds are indicators, not
ground-truth labels; do not equate zero flags or solver convergence with correct motion.

Controlled causal findings:

- Twenty trajectories exhaust the sixteen-window budget; 895 remaining flags are
  outside any attempted window, all on these budget-exhausted trajectories. Replaying
  the same captured pre-window states with only the budget changed to 64 removes all
  16 Berkeley/85_03 flags and all 210 Atlas/85_12 flags.
- N1/85_03 retains its 110.19° jump with 80 rather than 20 iterations. Starting with a
  two-second window and allowing four seconds removes all eleven flags and lowers
  the largest full-clip step to 12.41°. Longer windows only partly help iCub/85_03.
- N1/85_14 frame 2556 has a 312.06° shoulder step. All three actual restart seeds
  find candidates near loss 0.77055 rather than 4.76968, with a 2.24° maximum step,
  but a segment-error increase of 11.227° fails the 10° guard. Changing only that
  allowance to 12° accepts the candidate in a single-frame replay. This is evidence
  about acceptance, not a validated recommendation to globally change the default.
- X1/85_07 retains an inverted torso branch despite accurate first-stage pose fitting.
  At frame 718, removing temporal history exposes a much better spatial solution.
  A same-window comparison on frames [500, 959), retaining weights, limits, endpoint
  constraints and twenty iterations, gives 267 flags from the existing trajectory
  seed and zero from an independent static multi-start IK trajectory seed. Torso
  error at frame 718 changes from 173.64° to 4.48°, and window loss from 287.28 to
  112.29. The diagnostic candidate passes the existing pose/loss guards. Static seed
  generation costs an additional 33.3 seconds in this experiment; it is not runtime code.
- Berkeley/85_06 is unflagged despite 45.39° body RMS: its largest segment error is
  below 90° and its joint motion is smooth. A sixteen-start pose-only fit finds 42.15°
  RMS within declared limits versus 15.59° after a diagnostic range relaxation that
  violates thirteen original limits. Pelvis/head markers have fixed URDF chains, so
  their direction cannot follow arbitrary source torso motion with prescribed root
  rotation. Best-found IK values are not certified global lower bounds.

Keep these mechanisms distinct: work budget, window boundaries, trajectory seed,
acceptance tradeoffs, and representational limits require different responses.
Detection must still operate on the single normal result; alternative solves are
diagnostic/recovery work after a trigger. Do not silently promote artifact ablations
or weaken guards based on one successful case. The report records thirteen frame
probes, window replays, an independently audited 30-Hz comparison video, and all
112 default plus 112 disabled trajectories. No full-subject videos were generated.

### Ground-grid selection and regression reference

The 2026-11-05 ablation under `results/ground_grid_20261105/` covers both bundled
BVHs and complete CMU 85_03 on all eight robots, with toe-spacing/height scaling
and baseline/fixed-XY/corrected-XY variants: 195072 robot frames, all converged.
The user selected **height scaling + grid + XY correction** as the new default.
Historical graph-only references are the `grid_root` directories within `bvh_dance_height`,
`bvh_smpl_height`, and `85_03_height`. The old prototype and runtime are archived
under that result directory's `implementation/`; keep duplicate experimental
solvers out of the package and repository maintenance tools. These preceding-default
trajectories are comparison baselines, not exact-equality targets for the pose-guided default.

On 85_03 this choice reduced support toe-marker speed RMS by 85.4% to 91.3%.
Source-only support proxies used toe height at most 4% of source height, speed
at most 15% of height per second, and duration at least 0.1 seconds; 10% and 20%
speed thresholds were also checked. These labels do not enter optimization.
Measure mapped toe landmarks separately from calibration bodies (GR1 differs).
This measures marker sliding, not contact-patch friction or dynamic feasibility.
The second bundled clip still has pose regressions for X1 and Berkeley relative
to the preceding default; preserve that limitation when reporting improvements.

With one warmup and seven alternating X1 Dance solves (500 frames, OpenBLAS=1,
no concurrent rendering), the prototype's selected scheme took a median 3.175 s
versus 1.155 s for the preceding default. These are local measurements. Historical
videos use 640-by-480 views at 30 Hz; 107 videos passed timestamp and decode checks.

Promotion to the default passed 53 unit tests and full-trajectory comparisons on
both bundled BVHs plus CMU 85_03 across all eight robots: 32512 frames converged
with exactly zero joint, root-position, or root-quaternion difference from the
selected prototype. Both complete BVHs on a ramp added 6552 converged frames;
a real SMPL-X terrain CLI batch added 960 frames across all eight robots.
Joint/root bounds and velocities passed, and sampled full-state analytic-gradient
error was at most `2.64e-8`. Zero-height maps match the flat solve exactly.
An exported X1 terrain trajectory also passed a 30-Hz rendering smoke check
(125 frames, 320 by 240) and complete video decoding. Records are in
`results/default_grid_20261105/validation.json`.

From the repository root, in the environment documented in README:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests scripts
git diff --check
```

The tests cover FK and objective gradients, loss decomposition, buffer reuse,
configuration and pose isolation, in-memory results and export, static bounds and soft continuity,
normalization, source ground calibration, and independence from source foot-rotation
channels, the completeness of bundled model assets, terrain gradients and rendered
surfaces (including submillimeter spans), portable terrain export, batch discovery,
task failure isolation, live and interrupted batch records, linear serialization volume,
and rendering without overwriting concurrent batch progress. SMPL-X tests cover independent
joint regression and homogeneous FK, coordinate conventions, shape-dependent reference
poses, parameter validation, model reuse, mixed-format batches, and all eight robots.
Recovery tests additionally check source-aware detection, conditional retries,
window boundary preservation, residual Jacobians, pose-regression rejection,
terrain clearance, and diagnostics recomputed from the repaired history.
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
Graph refinement largely undid the initialization benefit. Those results describe the
superseded initialization-only method. On 2026-11-06 the user requested pose
initialization with a retained pose objective during graph refinement. This replaces
the preceding restriction on staged optimization. Historical comparisons remain
in ignored result artifacts; do not restore the removed source SO(3) mappings or
rotational Jacobians.

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
