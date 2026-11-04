import argparse
import json
from pathlib import Path
from .pipeline import run
from .solver import Settings


def main():
    parser = argparse.ArgumentParser(description='Fast MAHR BVH retargeting with analytic derivatives')
    parser.add_argument('motion', type=Path)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--legacy-project', type=Path, default=Path(__file__).resolve().parents[3] / 'MAHR')
    parser.add_argument('--robot-preset', default='UnitreeG1.py')
    parser.add_argument('--config', type=Path, help='Declarative JSON robot/source mapping; paths relative to JSON')
    parser.add_argument('--pose', choices=['stand', 't_pose'], default='stand')
    parser.add_argument('--frames', type=int, help='Number of source frames, before stride')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--maxiter', type=int, default=2000)
    parser.add_argument('--unweighted', action='store_true')
    parser.add_argument('--no-normalize', action='store_true')
    parser.add_argument('--numerical-gradient', action='store_true', help='Controlled derivative benchmark')
    parser.add_argument('--legacy-temporal', action='store_true', help='Disable temporal costs and velocity limits for legacy comparisons')
    parser.add_argument('--max-joint-velocity', type=float, default=8., help='Coordinate/s velocity bound')
    parser.add_argument('--temporal-weight', type=float, default=.02)
    parser.add_argument('--acceleration-weight', type=float, default=.1)
    parser.add_argument('--min-edge-ratio', type=float, default=.05,
                        help='Normalization denominator floor as fraction of rest skeleton height; 0 restores legacy')
    parser.add_argument('--foot-support-weight', type=float, default=0., help='Position-only near-ground foot-up prior; 0 disables')
    parser.add_argument('--posture-weight', type=float, default=.003, help='Non-saturating quadratic rest-pose prior')
    parser.add_argument('--ground-weight', type=float, default=0., help='Unilateral marker and collision-sphere penetration penalty')
    parser.add_argument('--allow-source-ground-penetration', action='store_true')
    parser.add_argument('--no-foot-surfaces', action='store_true', help='Disable STL foot hull contacts; marker/sphere penalty remains')
    ground = parser.add_mutually_exclusive_group()
    ground.add_argument('--no-source-ground-calibration', action='store_true')
    ground.add_argument('--source-ground-height', type=float, help='Static ground plane in original BVH length units')
    parser.add_argument('--source-ground-quantile', type=float, default=.05)
    args = parser.parse_args()
    if args.stride < 1 or args.frames is not None and args.frames < 1:
        parser.error('frames and stride must be positive')
    config = None
    if args.config:
        config = json.loads(args.config.read_text())
        for key in ('urdf', 'source_rest'):
            if key in config:
                config[key] = str((args.config.resolve().parent / config[key]).resolve())
    report = run(args.legacy_project, args.motion, args.output,
                 settings=Settings(maxiter=args.maxiter, weighted=not args.unweighted, ground_weight=args.ground_weight,
                     posture_weight=args.posture_weight,
                     foot_support_weight=args.foot_support_weight,
                     temporal_weight=0. if args.legacy_temporal else args.temporal_weight,
                     acceleration_weight=0. if args.legacy_temporal else args.acceleration_weight,
                     max_joint_velocity=None if args.legacy_temporal else args.max_joint_velocity,
                     max_root_velocity=None if args.legacy_temporal else 2.),
                 analytic=not args.numerical_gradient, pose=args.pose,
                 robot_file=args.robot_preset, frame_slice=slice(0, args.frames, args.stride),
                 config=config, normalize=not args.no_normalize, min_edge_ratio=args.min_edge_ratio,
                 clamp_source_ground=not args.allow_source_ground_penetration, foot_surfaces=not args.no_foot_surfaces,
                 source_ground_calibration=not args.no_source_ground_calibration,
                 source_ground_height=args.source_ground_height, source_ground_quantile=args.source_ground_quantile)
    print(json.dumps({k: v for k, v in report.items() if k != 'frames_detail'}, indent=2))


if __name__ == '__main__':
    main()
