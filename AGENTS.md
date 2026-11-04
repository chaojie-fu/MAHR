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

The runtime depends on NumPy, SciPy, and MuJoCo. Bundled resources live in `assets/`:
`robots/` contains the eight configured URDFs, their referenced meshes, and available
model-specific license files; `motions/` contains the two reproduction clips and standing
reference pose. Preserve model-relative mesh paths and license notices when updating
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
BVH + Python robot config
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
| `motion.py` | BVH parsing, coordinate conversion, and source graph vectors |
| `source_ground.py` | Static ground calibration over the complete input motion |
| `normalization.py` | Skeleton-height estimation and edge denominator floors |
| `kinematics.py` | Mesh-free MuJoCo FK, body origins, and landmark Jacobians |
| `graph.py` | Graph topology, ground-height relations, proximity weights, and edge derivatives |
| `objective.py` | Losses, analytic gradients, and a shared `LossBreakdown` |
| `solver.py` | `Settings`, sequential state, dynamic bounds, and L-BFGS-B calls |
| `pipeline.py` | Preparation, solving, result composition, and aggregate statistics |
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

### Objective and Solver

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
| Root-height bounds | `[0, 4]` meters |
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
channels, and the completeness of bundled model assets. Run checks appropriate to the change.
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

Local validation recorded on 2026-11-04:

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
