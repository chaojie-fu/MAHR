"""Mesh-free URDF loading and landmark-only FK/Jacobians."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


class Robot:
    """One mutable MuJoCo workspace per retargeting stream; no dynamics evaluation.

    Fixed bodies are retained to preserve URDF marker origins. Only independent
    scalar joints are supported; reject unsupported joints instead of silently
    changing their meaning. Coordinates follow MuJoCo's joint order, exported by name.
    """

    def __init__(self, urdf: str | Path, landmarks: list[str]):
        root = ET.parse(urdf).getroot()
        if root.tag != 'robot':
            raise ValueError('Expected a URDF robot')
        for joint in root.findall('joint'):
            if joint.get('type') not in ('fixed', 'revolute', 'continuous', 'prismatic'):
                raise ValueError(f'Unsupported joint: {joint.attrib}')
            if joint.find('mimic') is not None:
                raise ValueError('Mimic joints require an explicit coordinate mapping')
        for link in root.findall('link'):
            for tag in ('visual', 'collision'):
                for child in link.findall(tag):
                    link.remove(child)
        ext = root.find('mujoco')
        if ext is None:
            ext = ET.SubElement(root, 'mujoco')
        compiler = ext.find('compiler')
        if compiler is None:
            compiler = ET.SubElement(ext, 'compiler')
        compiler.set('fusestatic', 'false')
        compiler.set('balanceinertia', 'true')
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        if self.model.nq != self.model.nv or self.model.nq != self.model.njnt:
            raise ValueError('Only scalar joint coordinates are supported')
        self.names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                      for j in range(self.model.njnt)]
        self.landmarks = tuple(landmarks)
        if len(set(landmarks)) != len(landmarks) or 'ground' in landmarks:
            raise ValueError('Landmarks must be unique body names, excluding ground')
        self.body_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                                  for n in landmarks])
        if np.any(self.body_ids < 0):
            raise ValueError(f'Missing URDF landmarks: {[n for n, i in zip(landmarks, self.body_ids) if i < 0]}')
        limited = self.model.jnt_limited.astype(bool)
        self.lower = np.where(limited, self.model.jnt_range[:, 0], -np.pi)
        self.upper = np.where(limited, self.model.jnt_range[:, 1], np.pi)
        self._jacp = np.empty((3, self.model.nv))
        self._jacr = np.empty_like(self._jacp)

    @property
    def dofs(self):
        return self.model.nq

    def coordinates(self, values: dict[str, float]):
        unknown = set(values) - set(self.names)
        if unknown:
            raise ValueError(f'Unknown joints: {unknown}')
        return np.array([values.get(n, 0.) for n in self.names])

    def body_positions(self, q, names=None):
        """Evaluate body origins explicitly; omit names for all non-world bodies."""
        ids = slice(1, None)
        if names is not None:
            ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names])
            if np.any(ids < 0):
                raise ValueError(f'Missing URDF bodies: {[n for n, i in zip(names, ids) if i < 0]}')
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.xpos[ids].copy()

    def evaluate(self, q, jacobian=True):
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        positions = self.data.xpos[self.body_ids].copy()
        if not jacobian:
            return positions
        mujoco.mj_comPos(self.model, self.data)
        jac = np.empty((len(self.body_ids), 3, self.dofs))
        for k, body in enumerate(self.body_ids):
            mujoco.mj_jacBody(self.model, self.data, self._jacp, self._jacr, int(body))
            jac[k] = self._jacp
        return positions, jac
