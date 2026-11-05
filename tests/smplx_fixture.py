"""Small synthetic model data; no licensed SMPL-X weights are distributed."""
from pathlib import Path
import pickle

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation


PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
                    13, 14, 16, 17, 18, 19, 15, 15, 15]
                   + [p for root, start in ((20, 25), (21, 40))
                      for finger in range(start, start + 15, 3) for p in (root, finger, finger + 1)])
UPRIGHT = np.eye(3)[[2, 0, 1]]


def model_data():
    joints = np.array([
        [0, 1, 0], [.1, .95, 0], [-.1, .95, 0], [0, 1.1, 0],
        [.1, .55, 0], [-.1, .55, 0], [0, 1.25, 0], [.1, .12, 0],
        [-.1, .12, 0], [0, 1.4, 0], [.1, .06, .15], [-.1, .06, .15],
        [0, 1.55, 0], [.08, 1.5, 0], [-.08, 1.5, 0], [0, 1.7, 0],
        [.22, 1.5, 0], [-.22, 1.5, 0], [.5, 1.5, 0], [-.5, 1.5, 0],
        [.75, 1.5, 0], [-.75, 1.5, 0], [0, 1.65, .04], [.03, 1.7, .05], [-.03, 1.7, .05],
    ] + [[sign * (.75 + .03 * step), 1.5, .02 * (finger - 2)]
         for sign in (1, -1) for finger in range(5) for step in (1, 2, 3)], dtype=float)
    vertices = np.repeat(joints, 2, axis=0)
    regressor = np.zeros((55, len(vertices)))
    regressor[np.arange(55), 2*np.arange(55)] = .25
    regressor[np.arange(55), 2*np.arange(55)+1] = .75
    directions = np.random.default_rng(9).normal(0, .0004, (len(vertices), 3, 20))
    return dict(v_template=vertices, J_regressor=regressor, shapedirs=directions,
                kintree_table=np.stack((PARENTS, np.arange(55))),
                hands_meanl=np.full(45, .02), hands_meanr=np.full(45, -.02))


def write_model(path):
    path = Path(path)
    data = model_data()
    if path.suffix == '.pkl':
        data['J_regressor'] = csr_matrix(data['J_regressor'])
        with path.open('wb') as stream:
            pickle.dump(data, stream)
    else:
        np.savez(path, **data)
    return data


def motion_data(frames=8):
    pose = np.zeros((frames, 55, 3))
    pose[:, 0] = Rotation.from_matrix(UPRIGHT).as_rotvec()
    pose[:, 7, 0] = pose[:, 8, 0] = np.deg2rad(-20)
    pose[:, 16, 2], pose[:, 17, 2] = -np.pi/2, np.pi/2
    pose[:, 18, 1] = np.linspace(0, .2, frames)
    translation = np.broadcast_to([0., -1., 1.], (frames, 3)).copy()
    translation[:, 0] += np.arange(frames)*.002
    return dict(poses=pose.reshape(frames, 165), trans=translation,
                betas=np.zeros(10), gender=np.array('neutral'),
                mocap_frame_rate=np.array(120.), surface_model_type=np.array('smplx'))
