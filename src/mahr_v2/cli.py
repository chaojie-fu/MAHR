"""Command line for the production retargeting pipeline."""
import argparse
import json
from pathlib import Path
from .config import DEFAULT_ASSET_ROOT
from .pipeline import run
from .robots import PRESETS
from .trajectory import save_result


def main():
    parser = argparse.ArgumentParser(description='BVH retargeting with analytic derivatives')
    parser.add_argument('motion', type=Path)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path, default=DEFAULT_ASSET_ROOT,
                        help='Directory containing robots/ and motions/ (default: bundled assets)')
    parser.add_argument('--robot', choices=PRESETS, default='g1')
    parser.add_argument('--frames', type=int)
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()
    if args.stride < 1 or args.frames is not None and args.frames < 1:
        parser.error('frames and stride must be positive')
    result = run(args.asset_root, args.motion, robot=args.robot,
                 frame_slice=slice(0, args.frames, args.stride))
    save_result(result, args.output)
    print(json.dumps({k: v for k, v in result.report.items() if k != 'frames_detail'}, indent=2))


if __name__ == '__main__':
    main()
