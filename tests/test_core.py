import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial.transform import Rotation

from mahr_v2.graph import EdgeSpace, proximity_weights
from mahr_v2.pipeline import prepare
from mahr_v2.solver import FrameObjective, Settings, Retargeter

PROJECT = Path(os.environ.get('MAHR_LEGACY_PROJECT', Path(__file__).resolve().parents[2] / 'MAHR'))
MOTION = PROJECT / 'mahr/rsc/amass_bvh/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh'


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot, cls.edges, cls.scales, cls.source, cls.motion, _, cls.x = prepare(
            PROJECT, MOTION, frame_slice=slice(0, 4))

    def test_landmark_jacobian(self):
        q = self.x[:-1] + .12
        _, jac = self.robot.evaluate(q)
        numerical = approx_derivative(lambda z: self.robot.evaluate(z, jacobian=False).ravel(), q)
        assert_allclose(jac.reshape(-1, self.robot.dofs), numerical, atol=2e-8)

    def test_objective_gradient(self):
        x = self.x.copy() + .03
        # Both weighted/unweighted, pruned/full, rotated ground edges.
        for weighted in (True, False):
            for pairs in (None, [[0, 1], [1, 16], [3, 16]]):
                edges = EdgeSpace(16, pairs=pairs)
                p = self.robot.evaluate(self.x[:-1], jacobian=False)
                scale = edges.scales(p, self.x[-1])
                target = edges.vectors(p, np.eye(3), self.x[-1]) / scale[:, None] + .04
                rotation = Rotation.from_euler('xyz', [.2, -.3, .7]).as_matrix()
                obj = FrameObjective(self.robot, edges, scale, target, rotation, Settings(weighted=weighted))
                value, grad = obj.value_gradient(x)
                assert_allclose(value, obj.value(x), atol=1e-14)
                assert_allclose(grad, approx_derivative(obj.value, x).ravel(), atol=2e-7, rtol=1e-5)

    def test_legacy_loss(self):
        import networkx as nx
        spec = importlib.util.spec_from_file_location('legacy_distance', PROJECT / 'mahr/utils/graph_distance/compute_graph_distance.py')
        legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
        x = self.x + .02
        p = self.robot.evaluate(x[:-1], jacobian=False)
        vectors = self.edges.vectors(p, self.motion.root_rotations[0], x[-1]) / self.scales[:, None]
        names = list(range(17))
        source, target = nx.Graph(), nx.Graph()
        for (a, b), sv, tv in zip(self.edges.pairs, self.source[0], vectors):
            source.add_edge(a, b, position_vector=sv)
            target.add_edge(a, b, position_vector=tv)
        for weighted in (True, False):
            old = legacy.compute_vector_graph_distance(source, names, target, names, weighted)
            obj = FrameObjective(self.robot, self.edges, self.scales, self.source[0],
                                 self.motion.root_rotations[0], Settings(weighted=weighted, regularization=0))
            assert_allclose(obj.value(x), old, atol=1e-13)

    def test_warm_start_bounds_and_finite_results(self):
        result, stats = Retargeter(self.robot, self.edges, self.scales).solve(
            self.source, self.motion.root_rotations, self.x)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(all(s['success'] for s in stats))
        self.assertTrue(np.all(np.abs(np.diff(result, axis=0)) <= 50/120 + 1e-9))
        self.assertTrue(np.all(result[:, :-1] >= self.robot.lower - 1e-9))
        self.assertTrue(np.all(result[:, :-1] <= self.robot.upper + 1e-9))

    def test_bad_edges_and_stable_weights(self):
        for pairs in ([[1, 1]], [[0, 17]], [[2, 1]], [[0, 1], [0, 1]]):
            with self.assertRaises(ValueError):
                EdgeSpace(16, pairs=pairs)
        assert_allclose(proximity_weights(np.full((5, 3), 1e6)), np.full(5, .2))


if __name__ == '__main__':
    unittest.main()
