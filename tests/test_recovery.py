"""Single-output detection, shared objective linearization and conditional recovery."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize._numdiff import approx_derivative
from scipy.optimize import OptimizeResult

from mahr import prepare, solve, Settings, RecoverySettings, HeightMap
from mahr.objective import FrameObjective
from mahr.pose import PoseDirections
from mahr.quality import MotionQuality
from mahr.robots import PRESETS
from mahr.window import optimize_window
from mahr.recovery import Recovery
from mahr.objective import LossBreakdown
from test_core import ASSET_ROOT, MOTION


class RecoveryTests(unittest.TestCase):
    def quality(self, angles, dt=.01):
        robot = SimpleNamespace(dofs=1, body_ids=np.array([1, 2]),
            model=SimpleNamespace(jnt_bodyid=np.array([2]), body_parentid=np.array([0, 0, 1]), jnt_type=np.array([3])))
        pose = PoseDirections(['a', 'b'], [('a', 'b')], body_height=1.)
        targets = np.array([[[np.cos(a), np.sin(a), 0.]] for a in angles])
        return MotionQuality(robot, pose, targets, np.tile(np.eye(3), (len(angles), 1, 1)),dt,1.,RecoverySettings())

    def test_one_result_can_trigger_without_an_alternative(self):
        q = self.quality([0., 0., 0.])
        check = q.assess(2, np.array([1.]), np.array([0.]), [np.array([0.]), np.array([0.])], [np.array([0.])]*2)
        self.assertIn('velocity', check['reasons'])
        self.assertIn('acceleration', check['reasons'])
        self.assertNotIn('pose', check['reasons'])

    def test_fast_source_motion_is_not_a_fixed_velocity_violation(self):
        q = self.quality([0., .5, 1.])
        check = q.assess(2,np.array([1.]),np.array([0.]),[np.array([0.]),np.array([.5])],[np.array([0.])]*2)
        self.assertFalse(check['reasons'])
        self.assertGreater(q.speed[2,0],50.)

    def test_persistent_pose_error_and_numerical_failure_are_independent(self):
        q = self.quality(np.zeros(30))
        states = np.zeros((30,1));errors=np.full((30,1),np.deg2rad(100))
        checks=q.scan(states,errors)
        self.assertNotIn('pose',checks[0]['reasons'])
        self.assertIn('pose',checks[10]['reasons'])
        check=q.assess(0,states[0],np.zeros(1),[],[],success=False,feasible=False)
        self.assertEqual(set(check['reasons']),{'optimizer','infeasible'})

    def test_linearization_matches_objective_and_finite_differences(self):
        for robot in PRESETS:
            for terrain in (None,HeightMap([[.1,.2],[.2,.4]],half_size=(2,3))):
                d=prepare(ASSET_ROOT,MOTION,robot=robot,frame_slice=slice(0,2),terrain=terrain)
                x=np.r_[d.initial+.034,.017,-.023]
                obj=FrameObjective(d.robot,d.edges,d.scales,d.source_vectors[1],d.motion.root_rotations[1],Settings(),
                    posture_reference=d.posture_reference,body_height=d.normalization['target_height'],
                    ground_targets=d.ground_targets[1],root_xy=d.motion.positions[1,0,:2]*d.horizontal_ratio,
                    pose=d.pose,pose_target=d.pose_targets[1],previous=x+.01,older=x-.02,terrain=terrain,frame_dt=1/60)
                residual,jacobian=obj.linearize(x)
                value,gradient=obj.value_gradient(x)
                assert_allclose(residual@residual,value,atol=1e-11,err_msg=robot)
                assert_allclose(2*jacobian.T@residual,gradient,atol=1e-10,err_msg=robot)
                numeric=approx_derivative(lambda state:obj.linearize(state)[0],x)
                assert_allclose(jacobian,numeric,atol=3e-7,rtol=1e-5,err_msg=robot)

    def test_clean_motion_does_not_launch_extra_optimizations(self):
        d=prepare(ASSET_ROOT,MOTION,robot='g1',frame_slice=slice(0,5))
        with patch('mahr.recovery.optimize_window',side_effect=AssertionError('Unexpected window solve')):
            automatic=solve(d)
        disabled=solve(d,replace(Settings(),recovery=RecoverySettings(enabled=False)))
        assert_allclose(automatic.trajectory.coordinates,disabled.trajectory.coordinates,atol=0,rtol=0)
        self.assertEqual(automatic.report['recovery']['restart_attempts'],0)
        self.assertEqual(automatic.report['recovery']['unresolved_frames'],0)

    def test_window_removes_an_injected_jump_and_preserves_endpoints(self):
        # A small quadratic objective isolates temporal constraints from robot IK.
        class Quadratic:
            terrain=None
            def __init__(self,target):self.target=target
            def value(self,x):return float(np.sum((x-self.target)**2))
            def linearize(self,x):return x-self.target,np.eye(len(x))
        seed=np.zeros((21,4));seed[:,1]=.5;seed[9:12,0]=1.
        objectives=[Quadratic(np.array([0.,.5,0.,0.])) for _ in seed]
        fixed=np.zeros(21,dtype=bool);fixed[:2]=fixed[-2:]=True
        settings=replace(Settings(),recovery=RecoverySettings(window_iterations=5))
        candidate,info=optimize_window(objectives,seed,np.array([-2.,0.,-.2,-.2]),np.array([2.,4.,.2,.2]),
            np.full((21,1),8.),np.full((21,1),1000.),1/120,fixed,settings)
        self.assertIsNotNone(candidate)
        assert_allclose(candidate[fixed],seed[fixed],atol=1e-9)
        self.assertLessEqual(abs(np.diff(candidate[:,0])).max(),8/120+1e-7)
        self.assertLess(info['candidate_loss'],info['original_loss'])

    def test_configuration_validation(self):
        for args in ({'joint_speed':0.},{'pose_error':4.},{'window_iterations':1.5},
                     {'window_seconds':3.},{'source_motion_gain':np.inf},{'enabled':1},
                     {'window_pose_regression':np.nan}):
            with self.assertRaises(ValueError):RecoverySettings(**args)
        with self.assertRaises(ValueError):Settings(recovery=None)

    def test_reports_use_repaired_trajectory_history(self):
        d=prepare(ASSET_ROOT,MOTION,robot='g1',frame_slice=slice(0,5))
        def injected_repair(recovery,states,errors,stats,make_objective,settings):
            states[1,0]+=.01
            errors[1]=recovery.quality.errors(d.robot.evaluate(states[1,:d.robot.dofs],jacobian=False),1)
            stats[1]['window_repaired']=True
            return states,errors
        with patch.object(Recovery,'windows',injected_repair):
            result=solve(d)
        root_xy=d.motion.positions[:,0,:2]*d.horizontal_ratio
        states=np.c_[result.trajectory.coordinates,result.trajectory.root_positions[:,:2]-root_xy]
        for i,x in enumerate(states):
            objective=FrameObjective(d.robot,d.edges,d.scales,d.source_vectors[i],d.motion.root_rotations[i],Settings(),
                posture_reference=d.posture_reference,body_height=d.normalization['target_height'],
                pose=d.pose,pose_target=d.pose_targets[i],ground_targets=d.ground_targets[i],root_xy=root_xy[i],
                previous=states[i-1] if i else None,older=states[i-2] if i>1 else None,frame_dt=1/d.motion.fps)
            losses=objective.losses(x)
            self.assertAlmostEqual(result.report['frames_detail'][i]['loss'],losses.total,places=11)
            self.assertAlmostEqual(result.report['frames_detail'][i]['temporal_loss'],losses.temporal,places=11)
        self.assertEqual(result.report['recovery']['window_repaired_frames'],1)

    def test_triggered_restarts_explore_a_different_joint_configuration(self):
        quality=self.quality(np.zeros(21))
        class Objective:
            def losses(self,x):
                self.positions=np.array([[0.,0.,0.],[np.cos(x[0]),np.sin(x[0]),0.]])
                return LossBreakdown(float(x[0]**2),0.,0.,0.,0.,0.,0.)
        objective=Objective()
        low=np.array([-2.,0.,-.2,-.2]);high=np.array([2.,4.,.2,.2])
        recovery=Recovery(quality,low,high,np.array([-.2]),RecoverySettings())
        seen=[]
        def fit(x):
            seen.append(x.copy())
            return OptimizeResult(x=x.copy(),success=True,fun=float(x[0]**2))
        result=OptimizeResult(x=np.array([1.,.5]),success=True,fun=1.)
        selected,errors,info=recovery.restart(10,result,objective,[np.array([0.,.5])]*2,[np.zeros(1)]*2,
            fit,lambda x:True)
        self.assertTrue(info['restart_accepted'])
        self.assertEqual(info['restart_attempts'],3)
        self.assertTrue(any(abs(x[0]+1)<1e-12 for x in seen))
        self.assertAlmostEqual(selected.x[0],0.)
        self.assertEqual(recovery.reliable_frame,10)

    def test_frame_restart_accepts_pose_tradeoff_within_twenty_degrees(self):
        quality=self.quality(np.zeros(21))
        class Objective:
            def losses(self,x):
                angle=(1-x[0])*np.deg2rad(15)
                self.positions=np.array([[0.,0.,0.],[np.cos(angle),np.sin(angle),0.]])
                return LossBreakdown(.1+.1*abs(float(x[0])),0.,0.,0.,0.,0.,0.)
        def fit(seed):
            return OptimizeResult(x=np.array([0.,.5]),success=True,fun=.1)
        for settings,expected in [(RecoverySettings(),True),
                                  (RecoverySettings(pose_regression=np.deg2rad(10)),False)]:
            recovery=Recovery(quality,np.array([-2.,0.,-.2,-.2]),np.array([2.,4.,.2,.2]),
                              np.zeros(1),settings)
            result=OptimizeResult(x=np.array([1.,.5]),success=True,fun=.2)
            selected,_,info=recovery.restart(10,result,Objective(),[np.array([0.,.5])]*2,
                                             [np.zeros(1)]*2,fit,lambda x:True)
            self.assertEqual(info['restart_accepted'],expected)
            self.assertAlmostEqual(selected.x[0],0. if expected else 1.)

    def test_lower_window_loss_cannot_override_a_pose_regression(self):
        quality=self.quality(np.zeros(21))
        quality.robot.evaluate=lambda x,jacobian=False:np.array([[0.,0.,0.],[np.cos(x[0]),np.sin(x[0]),0.]])
        states=np.zeros((21,4));states[:,1]=.5;states[9:12,0]=1.
        errors=states[:,:1].copy()
        class Objective:
            def losses(self,x):return LossBreakdown(.1,0.,0.,0.,0.,0.,0.)
        recovery=Recovery(quality,np.array([-2.,0.,-.2,-.2]),np.array([2.,4.,.2,.2]),np.zeros(1),RecoverySettings())
        stats=[dict(success=True,pose_success=True,recovery_seconds=0.,seconds=0.,window_repaired=False) for _ in states]
        def bad_candidate(objectives,seed,*args,**kwargs):
            result=seed.copy();result[9:12,0]=1.+np.deg2rad(15)
            return result,dict(original_loss=10.,candidate_loss=1.)
        with patch('mahr.recovery.optimize_window',side_effect=bad_candidate):
            result,_=recovery.windows(states.copy(),errors.copy(),stats,lambda i:Objective(),Settings())
        assert_allclose(result,states,atol=0,rtol=0)
        self.assertTrue(recovery.events)
        self.assertTrue(all(e['reason']=='pose_regression' for e in recovery.events))
        self.assertFalse(any(s['window_repaired'] for s in stats))

    def test_window_preserves_nonlinear_terrain_clearance(self):
        terrain=HeightMap([[.1,.3],[.2,.5]],half_size=(1,1))
        class Objective:
            root_xy=np.zeros(2)
            def __init__(self):self.terrain=terrain
            def value(self,x):return float((x[1]+1)**2+np.sum(x[[0,2,3]]**2))
            def linearize(self,x):return x+np.array([0.,1.,0.,0.]),np.eye(4)
        seed=np.zeros((9,4));seed[:,1]=.6
        fixed=np.zeros(9,dtype=bool);fixed[:2]=fixed[-2:]=True
        result,_=optimize_window([Objective() for _ in seed],seed,np.array([-2.,.1,-.2,-.2]),
            np.array([2.,4.5,.2,.2]),np.full((9,1),16.),np.full((9,1),2000.),1/120,fixed,Settings())
        self.assertIsNotNone(result)
        self.assertTrue(np.all(result[:,1]>=terrain.sample(result[:,2:])-1e-8))
        assert_allclose(result[fixed],seed[fixed],atol=1e-9)


if __name__=='__main__':unittest.main()
