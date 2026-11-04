import unittest
import numpy as np
from numpy.testing import assert_allclose
from mahr_v2.motion import Motion
from mahr_v2.source_ground import calibrate_source_ground
from mahr_v2.pipeline import prepare
from test_core import PROJECT,MOTION


class SourceGroundTests(unittest.TestCase):
    def test_static_plane_no_inplace_changes_and_unit_invariance(self):
        positions=np.zeros((5,3,3))
        positions[:,0,2]=[1,2,3,2,1]
        positions[:,1,2]=[-2,-1,0,-1,-2]
        positions[:,2,2]=[-1,0,1,0,-1]
        motion=Motion(('root','left','right'),positions,np.tile(np.eye(3),(5,1,1)),120)
        calibrated,meta=calibrate_source_ground(motion,['left','right'])
        expected=float(np.quantile(positions[:,1:,2].min(axis=1),.05))
        self.assertEqual(meta['height_raw_units'],expected)
        assert_allclose(calibrated.positions[:,:,2],positions[:,:,2]-expected)
        assert_allclose(calibrated.positions[:,:,:2],positions[:,:,:2])
        assert_allclose(np.diff(calibrated.positions[:,:,2],axis=0),np.diff(positions[:,:,2],axis=0))
        self.assertFalse(np.shares_memory(calibrated.positions,positions))
        scaled=Motion(motion.names,positions*100,motion.root_rotations,120)
        result,_=calibrate_source_ground(scaled,['left','right'])
        assert_allclose(result.positions,calibrated.positions*100)
        manual,manual_meta=calibrate_source_ground(motion,['left','right'],height=-2)
        self.assertIsNone(manual_meta['estimated_from_frames'])
        assert_allclose(manual.positions,calibrated.positions)
        disabled,meta=calibrate_source_ground(motion,[],enabled=False)
        assert_allclose(disabled.positions,positions)
        self.assertEqual(meta['mode'],'disabled')
        for kwargs in (dict(height=float('nan')),dict(quantile=-1),dict(quantile=float('nan')),dict(enabled=False,height=0)):
            with self.assertRaises(ValueError):calibrate_source_ground(motion,['left','right'],**kwargs)

    def test_pipeline_calibrates_full_clip_before_slicing(self):
        full=prepare(PROJECT,MOTION)
        short=prepare(PROJECT,MOTION,frame_slice=slice(0,4))
        stride=prepare(PROJECT,MOTION,frame_slice=slice(0,4,2))
        self.assertEqual(full[1].normalization['source_ground'],short[1].normalization['source_ground'])
        assert_allclose(short[3],full[3][:4])
        assert_allclose(stride[3],full[3][:4:2])
        self.assertAlmostEqual(stride[4].fps,full[4].fps/2)
        self.assertAlmostEqual(full[1].normalization['source_ground']['height_raw_units'],-10.189564252651557)
        manual=prepare(PROJECT,MOTION,source_ground_height=0,frame_slice=slice(0,4))
        off=prepare(PROJECT,MOTION,source_ground_calibration=False,frame_slice=slice(0,4))
        assert_allclose(manual[3],off[3])


if __name__=='__main__':unittest.main()
