import json
from pathlib import Path
import shutil
import tempfile
import unittest

import mujoco
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from PIL import Image
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial.transform import Rotation

from mahr import HeightMap, prepare, run, save_result
from mahr.objective import FrameObjective
from mahr.rendering import add_ground
from mahr.solver import Retargeter, Settings
from test_core import ASSET_ROOT, MOTION


class TerrainTests(unittest.TestCase):
    def test_png_coordinates_clamping_and_validation(self):
        pixels = np.array([[0, 51, 102], [153, 204, 255]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'height.png'
            Image.fromarray(pixels).save(path)
            terrain = HeightMap.from_png(path, half_size=(2, 3), elevation=2, origin=(10, 20, -1))
            assert_allclose(terrain.sample([[8, 23], [12, 23], [8, 17], [12, 17]]), [-1, -.2, .2, 1])
            height, slopes = terrain.sample([[100, -100]], gradient=True)
            assert_allclose(height, [1])
            assert_array_equal(slopes, [[0, 0]])
            with self.assertRaises(FileNotFoundError):
                HeightMap.from_png(Path(folder)/'missing.png')
        for values in (np.zeros((1, 2)), np.zeros((2, 2, 2)), [[0, 1], [0, np.nan]]):
            with self.assertRaises(ValueError):
                HeightMap(values)
        for kwargs in (dict(half_size=(0, 1)), dict(origin=(0, np.nan, 0))):
            with self.assertRaises(ValueError):
                HeightMap(np.zeros((2, 2)), **kwargs)

    def test_slopes_and_mujoco_surface_agree(self):
        # Asymmetric, non-square grid catches transposes, Y flips, and wrong diagonals.
        samples = np.array([[.2, .6, .4, .7], [.3, .1, .9, .2], [.8, .5, .3, .6]])
        terrain = HeightMap(samples, half_size=(2, 3), origin=(1, -2, -.4))
        points = np.random.default_rng(8).uniform([-2, -3], [2, 3], size=(30, 2)) + terrain.origin[:2]
        heights, slopes = terrain.sample(points, gradient=True)
        for xy, slope in zip(points, slopes):
            assert_allclose(slope, approx_derivative(terrain.sample, xy).ravel(), atol=1e-9)
        for field in (terrain, HeightMap(np.full((3, 4), .7), origin=(1, -2, -.4))):
            spec = mujoco.MjSpec()
            add_ground(spec, field)
            model = spec.compile()
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            for xy in points:
                distance = mujoco.mj_ray(model, data, np.r_[xy, 5.], np.array([0., 0., -1.]),
                                         None, True, -1, None)
                self.assertGreater(distance, 0)
                assert_allclose(5-distance, field.sample(xy), atol=1e-7)

    def test_objective_gradient_with_terrain_and_root_translation(self):
        d = prepare(ASSET_ROOT, MOTION, frame_slice=slice(0, 1))
        terrain = HeightMap([[.1, .3, .2], [.5, .4, .7], [.2, .6, .3]], half_size=(2, 3))
        rotation = Rotation.from_euler('xyz', [.21, -.32, .71]).as_matrix()
        x = d.initial + .03
        objective = FrameObjective(d.robot, d.edges, d.scales, d.source_vectors[0], rotation, Settings(),
                                   posture_reference=d.posture_reference, previous=x-.02, older=x-.03,
                                   terrain=terrain, root_xy=np.array([.37, -.28]))
        value, gradient = objective.value_gradient(x)
        assert_allclose(value, objective.value(x), atol=1e-12)
        assert_allclose(gradient, approx_derivative(objective.value, x).ravel(), atol=2e-7, rtol=1e-5)

    def test_rendered_submillimeter_terrain_keeps_its_height(self):
        for span in (0., 1e-6, 1e-4, 5e-4, .001):
            with self.subTest(span=span):
                terrain = HeightMap([[0., span], [0., span]], half_size=(1, 1), origin=(0, 0, -.2))
                spec = mujoco.MjSpec()
                add_ground(spec, terrain)
                model = spec.compile()
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                for x in (-.5, 0., .5):
                    distance = mujoco.mj_ray(model, data, np.array([x, 0., 5.]), np.array([0., 0., -1.]),
                                             None, True, -1, None)
                    self.assertGreater(distance, 0.)
                    assert_allclose(5-distance, terrain.sample([x, 0.]), atol=1e-10, rtol=0.)

    def test_zero_map_preserves_flat_solve_and_normalization(self):
        flat = run(ASSET_ROOT, MOTION, robot='x1', frame_slice=slice(0, 5))
        mapped = run(ASSET_ROOT, MOTION, robot='x1', frame_slice=slice(0, 5),
                     terrain=HeightMap(np.zeros((2, 3))))
        assert_array_equal(flat.prepared.scales, mapped.prepared.scales)
        assert_array_equal(flat.prepared.source_vectors, mapped.prepared.source_vectors)
        assert_array_equal(flat.trajectory.coordinates, mapped.trajectory.coordinates)

    def test_vertical_translation_preserves_loss_and_gradient(self):
        d = prepare(ASSET_ROOT, MOTION, robot='icub', frame_slice=slice(0, 1))
        x = d.initial + .02
        values = []
        for offset in (0., -2., 5.):
            state = x.copy()
            state[-1] += offset
            objective = FrameObjective(d.robot, d.edges, d.scales, d.source_vectors[0],
                                       d.motion.root_rotations[0], Settings(),
                                       posture_reference=d.posture_reference, previous=state-.02,
                                       older=state-.03, terrain=HeightMap([[0, 0], [0, 0]], origin=(0, 0, offset)),
                                       root_xy=np.array([.5, -.3]))
            values.append(objective.value_gradient(state))
        for value, gradient in values[1:]:
            assert_allclose(value, values[0][0], atol=1e-12)
            assert_allclose(gradient, values[0][1], atol=1e-12)

    def test_infeasible_terrain_velocity_bounds_report_the_frame(self):
        d = prepare(ASSET_ROOT, MOTION, frame_slice=slice(0, 2))
        terrain = HeightMap([[0, 10], [0, 10]], half_size=(1, 1))
        solver = Retargeter(d.robot, d.edges, d.scales, posture_reference=d.posture_reference, terrain=terrain)
        with self.assertRaisesRegex(ValueError, 'world root XY'):
            solver.solve(d.source_vectors, d.motion.root_rotations, d.initial)
        with self.assertRaisesRegex(RuntimeError, 'frame 1'):
            solver.solve(d.source_vectors, d.motion.root_rotations, d.initial, root_xy=[[-1, 0], [1, 0]])

    def test_elevated_map_export_remains_portable(self):
        terrain = HeightMap(np.zeros((2, 3)), origin=(0, 0, 5))
        result = run(ASSET_ROOT, MOTION, frame_slice=slice(0, 4), terrain=terrain)
        self.assertEqual(result.report['unsuccessful_frames'], 0)
        self.assertTrue(np.all(result.trajectory.root_positions[:, 2] > 5))
        self.assertTrue(np.all(result.trajectory.root_positions[:, 2] <= 9))
        with tempfile.TemporaryDirectory() as folder:
            original, relocated = Path(folder)/'original', Path(folder)/'relocated'
            save_result(result, original/'clip.json')
            shutil.move(original, relocated)
            payload = json.loads((relocated/'clip.json').read_text())
            restored = HeightMap.load(relocated/payload['terrain'])
            assert_array_equal(restored.heights, terrain.heights)
            assert_array_equal(restored.origin, terrain.origin)
            assert_array_equal(restored.half_size, terrain.half_size)


if __name__ == '__main__':
    unittest.main()
