"""SMPL-X source options shared by retargeting entry points."""
from pathlib import Path

from .smplx import SMPLXSource


def add_source_arguments(parser):
    group = parser.add_argument_group('SMPL-X input')
    group.add_argument('--smplx-model', type=Path, help='SMPL-X model file or directory of gender-specific models')
    group.add_argument('--smplx-up-axis', choices=('z', 'y'), default='z', help='Input world up axis (default: z for AMASS)')
    group.add_argument('--smplx-gender', choices=('neutral', 'male', 'female'), help='Override input gender when selecting the body model')
    group.add_argument('--smplx-hand-mean', choices=('flat', 'model'), default='flat',
                       help='Hand axis angles are absolute (flat) or relative to the model mean (default: flat)')


def source_from_arguments(args):
    if args.smplx_model is None:
        if args.smplx_up_axis != 'z' or args.smplx_gender is not None or args.smplx_hand_mean != 'flat':
            raise ValueError('SMPL-X source options require --smplx-model')
        return None
    return SMPLXSource(args.smplx_model, up_axis=args.smplx_up_axis, gender=args.smplx_gender,
                       flat_hand_mean=args.smplx_hand_mean == 'flat')
