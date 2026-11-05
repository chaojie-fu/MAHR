"""SMPL-X parameter loading and joint-only forward kinematics in NumPy."""
from math import radians
from pathlib import Path
import pickle

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation

from .motion import Motion


JOINT_NAMES = (
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'jaw', 'left_eye_smplhf', 'right_eye_smplhf',
) + tuple(f'{side}_{finger}{joint}' for side in ('left', 'right')
          for finger in ('index', 'middle', 'pinky', 'ring', 'thumb') for joint in (1, 2, 3))

# Canonical SMPL-X is Y-up, with X pointing left and Z pointing forward.
CANONICAL_TO_MAHR = np.eye(3)[[2, 0, 1]]


def _numeric(value, name):
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f'SMPL-X {name} must be finite')
    return array


def _field(data, *names):
    for name in names:
        if name in data:
            return data[name]
    raise ValueError(f'SMPL-X input requires {" or ".join(names)}')


def _text(value, name):
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError(f'SMPL-X {name} must be a scalar string')
    item = value.item()
    return (item.decode('utf-8') if isinstance(item, bytes) else str(item)).lower()


def _pose(value, frames, joints, name):
    array = _numeric(value, name)
    if array.shape not in ((frames, joints * 3), (frames, joints, 3)):
        raise ValueError(f'SMPL-X {name} must have shape [{frames}, {joints * 3}] in axis-angle radians')
    return array.reshape(frames, joints, 3)


def _coefficients(value, frames, limit, name):
    array = _numeric(value, name)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[0] not in (1, frames) or not 1 <= array.shape[1] <= limit:
        raise ValueError(f'SMPL-X {name} must have 1 or {frames} rows and 1 to {limit} coefficients')
    return array


class _JointModel:
    """Project vertex shape bases onto the 55 joints once, then release the mesh."""

    def __init__(self, path):
        if path.suffix.lower() == '.npz':
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key] for key in
                        ('v_template', 'J_regressor', 'shapedirs', 'kintree_table',
                         'hands_meanl', 'hands_meanr') if key in archive}
        elif path.suffix.lower() == '.pkl':
            # Model pickles must come from a trusted SMPL-X model distribution.
            with path.open('rb') as stream:
                data = pickle.load(stream, encoding='latin1')
        else:
            raise ValueError('SMPL-X body model must be an .npz or .pkl file')
        vertices = _numeric(_field(data, 'v_template'), 'model template')
        directions = _numeric(_field(data, 'shapedirs'), 'model shape directions')
        regressor = csr_matrix(_field(data, 'J_regressor'), dtype=float)
        tree = np.asarray(_field(data, 'kintree_table'))
        if (vertices.ndim != 2 or vertices.shape[1] != 3
                or regressor.shape != (55, len(vertices)) or not np.isfinite(regressor.data).all()
                or directions.shape[:2] != vertices.shape or directions.ndim != 3
                or directions.shape[2] not in (20, 400)
                or tree.shape != (2, 55) or not np.array_equal(tree[1], np.arange(55))):
            raise ValueError('Expected a standard SMPL-X model with 55 joints and 20 or 400 shape/expression directions')
        self.parents = tree[0].astype(np.int64)
        self.parents[0] = -1
        if np.any(self.parents[1:] < 0) or np.any(self.parents[1:] >= np.arange(1, 55)):
            raise ValueError('Invalid SMPL-X joint hierarchy')
        self.joints = regressor @ vertices
        projected = (regressor @ directions.reshape(len(vertices), -1)).reshape(55, 3, -1)
        split = 300 if directions.shape[2] == 400 else 10
        self.shape = projected[:, :, :split]
        self.expression = projected[:, :, split:]
        self.hand_mean = None
        if 'hands_meanl' in data and 'hands_meanr' in data:
            self.hand_mean = _numeric(np.r_[data['hands_meanl'], data['hands_meanr']], 'hand mean')
            if self.hand_mean.shape != (90,):
                raise ValueError('SMPL-X model hand mean must contain 90 axis-angle values')
            self.hand_mean = self.hand_mean.reshape(30, 3)
        for array in (self.parents, self.joints, self.shape, self.expression, self.hand_mean):
            if array is not None:
                array.setflags(write=False)

    def forward(self, pose, betas, expression=None):
        bind = self.joints + np.einsum('jcb,fb->fjc', self.shape[:, :, :betas.shape[1]], betas)
        if expression is not None:
            bind = bind + np.einsum('jce,fe->fjc', self.expression[:, :, :expression.shape[1]], expression)
        rotations = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(len(pose), 55, 3, 3)
        positions = np.empty((len(pose), 55, 3))
        positions[:, 0] = bind[:, 0]
        for joint, parent in enumerate(self.parents[1:], 1):
            offset = bind[:, joint] - bind[:, parent]
            positions[:, joint] = positions[:, parent] + np.einsum('fij,fj->fi', rotations[:, parent], offset)
            rotations[:, joint] = rotations[:, parent] @ rotations[:, joint]
        return positions, rotations[:, 0].copy()


class SMPLXSource:
    """Read SMPL-X NPZ motions using explicit, user-supplied body models.

    AMASS world coordinates are Z-up by default; use up_axis='y' for Y-up files.
    Hand axis angles are absolute by default (flat_hand_mean=True). Set False
    for parameters expressed relative to the model's learned hand mean.
    Model geometry is cached per instance, independent of robot workspaces.
    """

    # Same standing joint rotations as the bundled BVH normalization pose.
    normalization_pose = {
        'left_ankle': (radians(-20), 0., 0.),
        'right_ankle': (radians(-20), 0., 0.),
        'left_shoulder': (0., 0., radians(-90)),
        'right_shoulder': (0., 0., radians(90)),
    }

    def __init__(self, model_path, *, up_axis='z', gender=None, flat_hand_mean=True):
        self.model_path = Path(model_path)
        if up_axis not in ('y', 'z'):
            raise ValueError('SMPL-X up_axis must be y or z')
        if gender is not None and gender not in ('neutral', 'male', 'female'):
            raise ValueError('SMPL-X gender must be neutral, male, or female')
        self.up_axis, self.gender, self.flat_hand_mean = up_axis, gender, flat_hand_mean
        self.normalization_pose = {name: tuple(values) for name, values in self.normalization_pose.items()}
        for name, values in self.normalization_pose.items():
            if name not in JOINT_NAMES or _numeric(values, 'normalization pose').shape != (3,):
                raise ValueError('SMPL-X normalization pose requires joint names and axis-angle triples')
        self._models = {}

    def _model(self, gender):
        if gender not in self._models:
            path = self.model_path
            if path.is_dir():
                candidates = [folder/f'SMPLX_{gender.upper()}{suffix}'
                              for suffix in ('.npz', '.pkl') for folder in (path, path/'smplx')]
                path = next((candidate for candidate in candidates if candidate.is_file()), None)
                if path is None:
                    raise FileNotFoundError(f'No SMPLX_{gender.upper()}.npz or .pkl model in {self.model_path}')
            elif path.stem.upper().startswith('SMPLX_') and path.stem.upper() != f'SMPLX_{gender.upper()}':
                raise ValueError(f'Model {path.name} does not match gender {gender}; select a matching model or set gender explicitly')
            self._models[gender] = _JointModel(path)
        return self._models[gender]

    def read(self, path):
        """Return a full motion and a standing reference with its first-frame shape."""
        with np.load(path, allow_pickle=False) as data:
            for key in ('surface_model_type', 'model_type'):
                if key in data and _text(data[key], key) != 'smplx':
                    raise ValueError('Expected SMPL-X motion parameters, not SMPL or SMPL-H')
            translation = _numeric(_field(data, 'trans', 'transl'), 'translation')
            if translation.ndim != 2 or translation.shape[1] != 3 or not len(translation):
                raise ValueError('SMPL-X translation must have shape [frames, 3] with at least one frame')
            frames = len(translation)
            rate = _numeric(_field(data, 'mocap_frame_rate', 'mocap_framerate', 'fps'), 'frame rate')
            if rate.size != 1 or rate.item() <= 0:
                raise ValueError('SMPL-X frame rate must be a positive scalar')
            fps = float(rate.item())
            gender = self.gender or (_text(data['gender'], 'gender') if 'gender' in data else 'neutral')
            if gender not in ('neutral', 'male', 'female'):
                raise ValueError(f'Unsupported SMPL-X gender: {gender}')
            if 'poses' in data:
                pose = _pose(data['poses'], frames, 55, 'poses').copy()
            else:
                pose = np.zeros((frames, 55, 3))
                pose[:, :1] = _pose(_field(data, 'root_orient', 'global_orient'), frames, 1, 'root orientation')
                pose[:, 1:22] = _pose(_field(data, 'pose_body', 'body_pose'), frames, 21, 'body pose')
                for names, start, count in ((('pose_jaw', 'jaw_pose'), 22, 1),
                                            (('leye_pose',), 23, 1), (('reye_pose',), 24, 1),
                                            (('left_hand_pose',), 25, 15), (('right_hand_pose',), 40, 15)):
                    if any(name in data for name in names):
                        pose[:, start:start+count] = _pose(_field(data, *names), frames, count, names[0])
                if 'pose_eye' in data:
                    pose[:, 23:25] = _pose(data['pose_eye'], frames, 2, 'pose_eye')
                if 'pose_hand' in data:
                    pose[:, 25:] = _pose(data['pose_hand'], frames, 30, 'pose_hand')
            model = self._model(gender)
            betas = _coefficients(_field(data, 'betas'), frames, model.shape.shape[2], 'betas')
            expression = (_coefficients(data['expression'], frames, model.expression.shape[2], 'expression')
                          if 'expression' in data else None)
        reference_pose = np.zeros((1, 55, 3))
        for name, values in self.normalization_pose.items():
            reference_pose[0, JOINT_NAMES.index(name)] = values
        if not self.flat_hand_mean:
            if model.hand_mean is None:
                raise ValueError('SMPL-X body model does not contain hand means')
            pose[:, 25:] += model.hand_mean
            reference_pose[:, 25:] += model.hand_mean
        positions, rotations = model.forward(pose, betas, expression)
        positions += translation[:, None]
        world = np.eye(3) if self.up_axis == 'z' else CANONICAL_TO_MAHR
        positions = positions @ world.T
        # Express the root frame in robot-compatible axes without rotating an upright
        # AMASS subject onto its side (SMPL-X's local root frame remains Y-up).
        rotations = world @ rotations @ CANONICAL_TO_MAHR.T
        rest_positions, _ = model.forward(reference_pose, betas[:1])
        rest_positions = rest_positions @ CANONICAL_TO_MAHR.T
        return (Motion(JOINT_NAMES, positions, rotations, fps, 'meters'),
                Motion(JOINT_NAMES, rest_positions, np.eye(3)[None], fps, 'meters'))
