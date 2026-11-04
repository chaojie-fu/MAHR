"""Shared height-map options for the production and validation entry points."""
from pathlib import Path

from .terrain import HeightMap


def add_terrain_arguments(parser):
    group = parser.add_argument_group('target terrain')
    group.add_argument('--height-map', type=Path, help='8-bit grayscale PNG; omitted for flat ground')
    group.add_argument('--terrain-half-size', type=float, nargs=2, default=(20., 20.), metavar=('X', 'Y'),
                       help='Terrain half extents in meters (default: 20 20)')
    group.add_argument('--terrain-elevation', type=float, default=1.,
                       help='Meters represented by white pixels (default: 1)')
    group.add_argument('--terrain-origin', type=float, nargs=3, default=(0., 0., 0.), metavar=('X', 'Y', 'Z'),
                       help='Map center XY and black-pixel Z in world meters (default: 0 0 0)')


def terrain_from_arguments(args):
    if args.height_map is None:
        if (tuple(args.terrain_half_size) != (20., 20.) or args.terrain_elevation != 1.
                or tuple(args.terrain_origin) != (0., 0., 0.)):
            raise ValueError('Terrain coordinates and elevation require --height-map')
        return None
    return HeightMap.from_png(args.height_map, half_size=args.terrain_half_size,
                              elevation=args.terrain_elevation, origin=args.terrain_origin)
