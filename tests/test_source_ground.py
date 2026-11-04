import unittest
import numpy as np
from numpy.testing import assert_allclose
from mahr_v2.motion import Motion
from mahr_v2.source_ground import calibrate_source_ground
from mahr_v2.pipeline import prepare
from test_core import ASSET_ROOT,MOTION


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
        for quantile in (-1,float('nan')):
            with self.assertRaises(ValueError):calibrate_source_ground(motion,['left','right'],quantile=quantile)

    def test_pipeline_calibrates_full_clip_before_slicing(self):
        full=prepare(ASSET_ROOT,MOTION)
        short=prepare(ASSET_ROOT,MOTION,frame_slice=slice(0,4))
        stride=prepare(ASSET_ROOT,MOTION,frame_slice=slice(0,4,2))
        self.assertEqual(full.normalization['source_ground'],short.normalization['source_ground'])
        assert_allclose(short.source_vectors,full.source_vectors[:4])
        assert_allclose(stride.source_vectors,full.source_vectors[:4:2])
        self.assertAlmostEqual(stride.motion.fps,full.motion.fps/2)
        self.assertAlmostEqual(full.normalization['source_ground']['height_raw_units'],-10.189564252651557)



if __name__=='__main__':unittest.main()
