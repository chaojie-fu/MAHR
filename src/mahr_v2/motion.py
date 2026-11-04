"""Vectorized BVH channel decoding and source graph preparation."""
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation
from .normalization import DEFAULT_MIN_EDGE_RATIO, edge_denominators, skeleton_height


@dataclass
class Motion:
    names: tuple[str, ...]
    positions: np.ndarray  # [frames, joints, xyz], world coordinates
    root_rotations: np.ndarray  # [frames, 3, 3], root to world
    fps: float


def read_bvh(path, frame_slice=slice(None)):
    """Y-up BVH -> Z-up using legacy (z,x,y) permutation; preserve input units.

    Decode whole channel columns at once; propagate one joint at a time across all
    selected frames. End sites have no semantic landmarks and are skipped.
    """
    hierarchy, body = Path(path).read_text().split('MOTION', 1)
    tokens = iter(re.findall(r'\{|\}|[^\s{}]+', hierarchy))
    names, parents, offsets, channels = [], [], [], []
    cursor = 0

    def joint(parent):
        nonlocal cursor
        name = next(tokens)
        if next(tokens) != '{' or next(tokens) != 'OFFSET':
            raise ValueError('Malformed BVH joint')
        offset = [float(next(tokens)) for _ in range(3)]
        index = len(names)
        names.append(name); parents.append(parent); offsets.append(offset)
        channels.append([])
        while True:
            tag = next(tokens)
            if tag == '}':
                return
            if tag == 'CHANNELS':
                n = int(next(tokens))
                channels[index] = [(next(tokens), cursor + k) for k in range(n)]
                cursor += n
            elif tag == 'JOINT':
                joint(index)
            elif tag == 'End':
                if next(tokens) != 'Site' or next(tokens) != '{' or next(tokens) != 'OFFSET':
                    raise ValueError('Malformed BVH end site')
                for _ in range(3):
                    float(next(tokens))
                if next(tokens) != '}':
                    raise ValueError('Malformed BVH end site')
            else:
                raise ValueError(f'Unknown BVH token {tag}')

    if next(tokens) != 'HIERARCHY' or next(tokens) != 'ROOT':
        raise ValueError('Expected BVH hierarchy and root')
    joint(-1)
    match = re.match(r'\s*Frames:\s*(\d+)\s*Frame\s+Time:\s*([\d.eE+-]+)\s*', body)
    if match is None:
        raise ValueError('Malformed BVH motion header')
    frames, dt = int(match[1]), float(match[2])
    if frames < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid frame count/time')
    values = np.fromstring(body[match.end():], sep=' ')
    if values.size != frames * cursor or not np.isfinite(values).all():
        raise ValueError('BVH channel count mismatch or non-finite values')
    values = values.reshape(frames, cursor)[frame_slice]
    if not len(values):
        raise ValueError('No frames selected')
    positions = np.empty((len(values), len(names), 3))
    rotations = np.empty((len(values), len(names), 3, 3))
    for j, ch in enumerate(channels):
        local_p = np.broadcast_to(offsets[j], (len(values), 3)).copy()
        local_r = np.broadcast_to(np.eye(3), (len(values), 3, 3)).copy()
        for label, column in ch:
            axis = 'XYZ'.index(label[0])
            if label.endswith('position'):
                local_p[:, axis] = values[:, column]
            elif label.endswith('rotation'):
                rv = np.zeros((len(values), 3))
                rv[:, axis] = np.deg2rad(values[:, column])
                local_r = local_r @ Rotation.from_rotvec(rv).as_matrix()
            else:
                raise ValueError(f'Unsupported BVH channel {label}')
        p = parents[j]
        if p < 0:
            positions[:, j], rotations[:, j] = local_p, local_r
        else:
            positions[:, j] = positions[:, p] + np.einsum('fij,fj->fi', rotations[:, p], local_p)
            rotations[:, j] = rotations[:, p] @ local_r
    perm = [2, 0, 1]
    stride = frame_slice.step or 1
    return Motion(tuple(names), positions[:, :, perm], rotations[:, 0][:, perm][:, :, perm], 1. / dt / stride)


def prepare_source(motion, rest, landmark_names, toe_names, edges, normalize=True,
                   min_edge_ratio=DEFAULT_MIN_EDGE_RATIO, body_height=None, clamp_source_ground=True):
    ids = [motion.names.index(n) for n in landmark_names]
    rest_ids = [rest.names.index(n) for n in landmark_names]
    toes = [rest.names.index(n) for n in toe_names]
    # Legacy source rest pose: subtract toe minimum, then add 5% of pelvis height.
    rest_world = rest.positions[0].copy()
    rest_world[:, 2] -= rest_world[toes, 2].min()
    rest_world[:, 2] += .05 * rest_world[0, 2]
    rest_local = (rest_world[rest_ids] - rest_world[0]) @ rest.root_rotations[0]
    # Rest body edges use rest root orientation; ground edges remain world vertical.
    rest_vectors = edges.vectors(rest_local, rest.root_rotations[0], rest_world[0, 2])
    if normalize:
        if body_height is None:
            body_height = skeleton_height(rest.positions[0], rest.positions[0, toes])
        scale = edge_denominators(np.linalg.norm(rest_vectors, axis=1), body_height,
                                  min_edge_ratio, source=True)
    else:
        scale = np.ones(len(edges.pairs))
    local = np.einsum('fkj,fji->fki', motion.positions[:, ids] - motion.positions[:, :1], motion.root_rotations)
    vectors = np.array([edges.vectors(p, r, h)
                        for p, r, h in zip(local, motion.root_rotations, motion.positions[:, 0, 2])])
    if clamp_source_ground:
        # Body -> ground edges use -height. Positive z means source penetration.
        vectors[:, edges.ground, 2] = np.minimum(vectors[:, edges.ground, 2], 0.)
    vectors /= scale[None, :, None]
    foot_distance = np.linalg.norm(rest.positions[0, toes[0], :2] - rest.positions[0, toes[1], :2])
    if foot_distance <= 0:
        raise ValueError('Source rest toe distance must be positive')
    return vectors, float(foot_distance)
