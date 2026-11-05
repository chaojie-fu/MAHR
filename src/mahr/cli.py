"""Command line for the production retargeting pipeline."""
import argparse
import json
from pathlib import Path
from .config import DEFAULT_ASSET_ROOT
from .pipeline import run
from .robots import PRESETS
from .trajectory import save_result
from .batch import run_batch
from .terrain_cli import add_terrain_arguments, terrain_from_arguments
from .source_cli import add_source_arguments, source_from_arguments


def main():
    parser = argparse.ArgumentParser(description='BVH and SMPL-X retargeting with analytic derivatives')
    parser.add_argument('motion', type=Path, help='BVH/SMPL-X NPZ file or directory searched recursively')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='JSON file for one motion/robot; output directory for a batch')
    parser.add_argument('--asset-root', type=Path, default=DEFAULT_ASSET_ROOT,
                        help='Directory containing robots/ and motions/ (default: bundled assets)')
    parser.add_argument('--robot', choices=[*PRESETS, 'all'], action='append',
                        help='Target robot; repeat to select several, or use all (default: g1)')
    parser.add_argument('--frames', type=int)
    parser.add_argument('--stride', type=int, default=1)
    add_terrain_arguments(parser)
    add_source_arguments(parser)
    args = parser.parse_args()
    args.robot = args.robot or ['g1']
    if args.stride < 1 or args.frames is not None and args.frames < 1:
        parser.error('frames and stride must be positive')
    if 'all' in args.robot and args.robot != ['all']:
        parser.error('Use --robot all alone, or list individual robot names')
    robots = list(PRESETS) if args.robot == ['all'] else args.robot
    if len(set(robots)) != len(robots):
        parser.error('Robot names must be unique')
    try:
        terrain = terrain_from_arguments(args)
        smplx = source_from_arguments(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.motion.is_dir() or len(robots) > 1:
        def progress(item):
            detail = item.get('error', f"{item.get('frames', 0)} frames")
            print(f"{item['status']}: {item['robot']} / {item['motion']} ({detail})", flush=True)
        try:
            summary = run_batch(args.asset_root, args.motion, args.output, robots=robots,
                                frame_slice=slice(0, args.frames, args.stride),
                                terrain=terrain, smplx=smplx, progress=progress)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps({k: v for k, v in summary.items() if k != 'tasks'}, indent=2))
        if summary['failed'] or summary['unconverged']:
            raise SystemExit(1)
        return
    try:
        result = run(args.asset_root, args.motion, robot=robots[0],
                     frame_slice=slice(0, args.frames, args.stride), terrain=terrain, smplx=smplx)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    save_result(result, args.output)
    print(json.dumps({k: v for k, v in result.report.items() if k != 'frames_detail'}, indent=2))


if __name__ == '__main__':
    main()
