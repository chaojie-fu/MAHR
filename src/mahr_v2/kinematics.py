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
        ground_spheres = []
        for link in root.findall('link'):
            for collision in link.findall('collision'):
                sphere = collision.find('geometry/sphere')
                if sphere is not None:
                    origin = collision.find('origin')
                    center = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
                    radius = float(sphere.get('radius'))
                    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 0:
                        raise ValueError('Invalid collision sphere')
                    ground_spheres.append((link.get('name'), center, radius))
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
        self.rest_coordinates = np.zeros(self.model.nq)
        self.foot_bodies = np.empty(0, dtype=int)
        self.foot_up_axes = np.empty((0,3))
        self.ground_meshes = []
        self.ground_spheres = [(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n), p, r)
                               for n, p, r in ground_spheres]
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


    def ground_clearances(self, rotation, height, jacobian=False):
        """Collision-sphere and cached foot surface clearances using evaluated FK.

        Foot hull vertices exactly bound linear height over the original STL mesh.
        """
        count = len(self.ground_spheres) + sum(len(v) for _,v in self.ground_meshes)
        clearances = np.empty(count)
        jac = np.zeros((count, self.dofs+1)) if jacobian else None
        for i, (body, center, radius) in enumerate(self.ground_spheres):
            point = self.data.xpos[body] + self.data.xmat[body].reshape(3, 3) @ center
            clearances[i] = rotation[2] @ point + height - radius
            if jacobian:
                mujoco.mj_jac(self.model, self.data, self._jacp, self._jacr, point, body)
                jac[i, :-1] = rotation[2] @ self._jacp
                jac[i, -1] = 1.
        index = len(self.ground_spheres)
        normal = rotation[2]
        for body, vertices in self.ground_meshes:
            offsets = vertices @ self.data.xmat[body].reshape(3,3).T
            points = offsets + self.data.xpos[body]
            count = len(vertices)
            clearances[index:index+count] = points @ normal + height
            if jacobian:
                mujoco.mj_jacBody(self.model, self.data, self._jacp, self._jacr, body)
                jac[index:index+count, :-1] = normal @ self._jacp + np.cross(offsets,normal) @ self._jacr
                jac[index:index+count, -1] = 1.
            index += count
        return (clearances, jac) if jacobian else clearances


    def ground_penalty(self, rotation, height, jacobian=False):
        """Fused hinge-square value/gradient; average per foot, not per mesh vertex.

        Aggregate vertex forces before applying body Jacobians. This avoids a
        [vertices, dofs] array and keeps mesh density from increasing its weight.
        """
        value = 0.
        gradient = np.zeros(self.dofs+1)
        normal = rotation[2]
        for body, center, radius in self.ground_spheres:
            point = self.data.xpos[body]+self.data.xmat[body].reshape(3,3)@center
            depth = max(0., radius-normal@point-height)
            value += depth**2
            if jacobian and depth > 0:
                mujoco.mj_jac(self.model,self.data,self._jacp,self._jacr,point,body)
                gradient[:-1] -= 2*depth*(normal@self._jacp)
                gradient[-1] -= 2*depth
        for body, vertices in self.ground_meshes:
            body_rotation = self.data.xmat[body].reshape(3,3)
            clearance = vertices@(body_rotation.T@normal)+normal@self.data.xpos[body]+height
            depth = np.maximum(-clearance,0.)
            value += np.dot(depth,depth)/len(vertices)
            if jacobian and np.any(depth > 0):
                mujoco.mj_jacBody(self.model,self.data,self._jacp,self._jacr,body)
                total = depth.sum()/len(vertices)
                offset = body_rotation@(depth@vertices/len(vertices))
                gradient[:-1] -= 2*(total*(normal@self._jacp)+np.cross(offset,normal)@self._jacr)
                gradient[-1] -= 2*total
        return float(value), gradient
