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
                 settings=Settings(maxiter=args.maxiter, weighted=not args.unweighted),
                 analytic=not args.numerical_gradient, pose=args.pose,
                 robot_file=args.robot_preset, frame_slice=slice(0, args.frames, args.stride),
                 config=config, normalize=not args.no_normalize)
    print(json.dumps({k: v for k, v in report.items() if k != 'frames_detail'}, indent=2))


if __name__ == '__main__':
    main()
