"""Check world coordinates after mesh compilation and animation, without OpenGL."""
import unittest

import mujoco
import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from mahr.rendering import set_mesh_vertices


class AnimatedMeshTests(unittest.TestCase):
    def test_compiled_transform_preserves_animated_world_vertices_and_normals(self):
        # Asymmetry, rotation, and an offset force a nontrivial compiler transform.
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        points = points @ Rotation.from_euler('xyz', [25, 10, 65], degrees=True).as_matrix().T
        points += [.3, -.2, .4]
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        normals = points - points.mean(axis=0)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        spec = mujoco.MjSpec()
        spec.add_mesh(name='animated', uservert=points.ravel(), userface=faces.ravel(),
                      usernormal=normals.ravel(), userfacenormal=faces.ravel())
        body = spec.worldbody.add_body(name='subject', mocap=True)
        body.add_geom(name='surface', type=mujoco.mjtGeom.mjGEOM_MESH, meshname='animated')
        model = spec.compile()
        data = mujoco.MjData(model)
        geom = model.geom('surface').id
        self.assertGreater(np.linalg.norm(model.geom_pos[geom]), .1)
        # The historical failure left this optimization tied to the inertial frame
        # while resetting the geom transform and uploading unconverted vertices.
        self.assertNotEqual(int(model.geom_sameframe[geom]), 0)
        original_pos, original_quat = model.geom_pos.copy(), model.geom_quat.copy()
        for step in range(3):
            rotation = Rotation.from_euler('xyz', [15 * step, -8, 35 * step], degrees=True)
            data.mocap_pos[0] = [.4 * step, -.6, 1.]
            data.mocap_quat[0] = rotation.as_quat()[[3, 0, 1, 2]]
            animated = points * (1 + .05 * step) + [.1 * step, 0, -.02 * step]
            set_mesh_vertices(model, geom, animated, normals)
            mujoco.mj_forward(model, data)
            geom_rotation = data.geom_xmat[geom].reshape(3, 3)
            actual = model.mesh_vert @ geom_rotation.T + data.geom_xpos[geom]
            expected = rotation.apply(animated) + data.mocap_pos[0]
            assert_allclose(actual, expected, atol=2e-7, rtol=0)
            assert_allclose(model.mesh_normal @ geom_rotation.T, rotation.apply(normals),
                            atol=1e-7, rtol=0)
        assert_allclose(model.geom_pos, original_pos, atol=0, rtol=0)
        assert_allclose(model.geom_quat, original_quat, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
