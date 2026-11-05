"""Run the production pipeline for all robots and validate trajectories and derivatives."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from mahr.config import DEFAULT_ASSET_ROOT
from mahr.robots import PRESETS
from mahr.pipeline import run
from mahr.objective import FrameObjective, CLIPPED_POSE_LIMIT
from mahr.solver import Settings
from mahr.trajectory import save_result
from mahr.terrain_cli import add_terrain_arguments, terrain_from_arguments
from mahr.source_cli import add_source_arguments, source_from_arguments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-root', type=Path, default=DEFAULT_ASSET_ROOT,
                        help='Directory containing robots/ and motions/ (default: bundled assets)')
    parser.add_argument('--motion', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/all_robots'))
    parser.add_argument('--reference', type=Path, help='Compare complete trajectories against saved results')
    add_terrain_arguments(parser)
    add_source_arguments(parser)
    args = parser.parse_args()
    try:
        terrain = terrain_from_arguments(args)
        smplx = source_from_arguments(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    motion = args.motion or args.asset_root/'motions/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh'
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    settings = Settings()
    for name in PRESETS:
        output = args.output/f'{name}.json'
        result = run(args.asset_root, motion, robot=name, settings=settings, terrain=terrain, smplx=smplx)
        save_result(result, output)
        data, stats = result.prepared, result.report
        robot, clip = data.robot, data.motion
        trajectory = result.trajectory
        q = trajectory.coordinates
        assert len(q) == len(clip.positions) and np.isfinite(q).all()
        ground = terrain.sample(trajectory.root_positions[:, :2]) if terrain is not None else 0.
        assert np.all(q[:, :-1] >= robot.lower-1e-8)
        assert np.all(q[:, :-1] <= robot.upper+1e-8)
        assert np.all(q[:, -1] >= ground-1e-8)
        assert np.all(q[:, -1] <= ground+4.+1e-8)
        if len(q) > 1:
            assert np.max(abs(np.diff(q[:,:-1], axis=0)))*clip.fps <= settings.max_joint_velocity+1e-7
            assert np.max(abs(np.diff(q[:,-1])))*clip.fps <= settings.max_root_velocity+1e-7
        checked = sorted(set((0, len(q)//2, len(q)-1)))
        max_error = 0.
        for index in checked:
            x = q[index].copy()
            relative = x.copy()
            if terrain is not None:
                relative[-1] -= ground[index]
            x[np.abs(np.abs(relative)-CLIPPED_POSE_LIMIT)<1e-4] += .001
            obj = FrameObjective(robot, data.edges, data.scales, data.source_vectors[index],
                                 clip.root_rotations[index], settings,
                                 posture_reference=data.posture_reference,
                                 previous=q[index-1] if index else None,
                                 older=q[index-2] if index>1 else None, frame_dt=1/clip.fps,
                                 terrain=terrain, root_xy=trajectory.root_positions[index, :2])
            _, gradient = obj.value_gradient(x)
            numeric = approx_derivative(obj.value, x, abs_step=1e-8).ravel()
            max_error = max(max_error, float(np.max(abs(gradient-numeric))))
            assert np.allclose(gradient, numeric, atol=2e-5, rtol=1e-4)
        report[name] = {k:v for k,v in stats.items() if k!='frames_detail'}
        report[name].update(dofs=robot.dofs, gradient_max_error=max_error,
            finite_and_within_bounds=True, source_motion=str(motion.resolve()),
            source_fps=clip.fps, checked_frames=checked,
            urdf=str(data.urdf.resolve()), trajectory=str(output.resolve()))
        if args.reference:
            reference = json.loads((args.reference/f'{name}.json').read_text())
            # Include root translation and orientation as well as every joint coordinate.
            assert reference['fps'] == trajectory.fps
            actual = np.column_stack((trajectory.joint_positions, trajectory.root_positions, trajectory.root_orientations))
            expected = np.array([[f['JointPos'][n] for n in robot.names]+f['RootPos']+f['RootOrient'] for f in reference['trajectory']])
            assert actual.shape == expected.shape
            difference = float(np.max(abs(actual-expected)))
            assert difference < 1e-10, (name, difference)
            report[name]['reference_max_difference'] = difference
        (args.output/'summary.json').write_text(json.dumps(report, indent=2))
        print(name, 'frames', len(q), 'failed', stats['unsuccessful_frames'],
              'reference_error', report[name].get('reference_max_difference'), flush=True)
    if any(item['unsuccessful_frames'] for item in report.values()):
        raise RuntimeError('Some frames did not converge; inspect per-frame diagnostics')


if __name__ == '__main__':
    main()
