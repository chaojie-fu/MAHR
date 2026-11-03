"""Optional integration regression against unmodified legacy source/robot graphs.

Requires legacy dependencies on PYTHONPATH. Writes only into a temporary folder.
"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from mahr_v2.pipeline import prepare
from mahr_v2.config import legacy_preset


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=Path('../MAHR'))
    args=parser.parse_args()
    np.float=float  # legacy urdfpy compatibility with NumPy 2, reference process only
    from urdfpy import URDF
    from mahr.utils.graph_distance.robot_graph_generator import generate_vector_symbolic_graph_for_robot
    from mahr.utils.graph_distance.mocap_graph_generator import generate_vector_graphs_for_motion
    from config.MocapConfig.AmassSmplHGConfig import AMASS_DanceDB_SMPL_HG_WholeBody_Config
    project=args.project
    path=project/'mahr/rsc/amass_bvh/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh'
    report={}
    for pose in ('stand','t_pose'):
        r,e,s,v,m,ratio,x=prepare(project,path,pose=pose,frame_slice=slice(0,5))
        preset=legacy_preset(project,pose=pose)
        old_robot=URDF.load(preset['urdf'])
        config=SimpleNamespace(robot=old_robot,key_joints=preset['key_joints'],toes=preset['toes'],norm_pose_in_rad=preset['norm_pose_in_rad'])
        _,_,_,fn,_=generate_vector_symbolic_graph_for_robot(config,normalize_edge_weights=True)
        order=[r.names.index(n) for n in old_robot.actuated_joint_names]
        max_target=0
        for q in (x,x+0.2):
            rotation=Rotation.from_euler('xyz',[.2,-.3,.7]).as_matrix()
            old=fn(q[:-1][order],rotation,0.,0.,q[-1])
            expected=np.array([old.get_edge_data(preset['key_joints'][a],preset['key_joints'][b])['position_vector'] for a,b in e.pairs])
            actual=e.vectors(r.evaluate(q[:-1],False),rotation,q[-1])/s[:,None]
            assert_allclose(actual,expected,atol=2e-12)
            max_target=max(max_target,float(np.max(np.abs(actual-expected))))
        mocap=AMASS_DanceDB_SMPL_HG_WholeBody_Config(pose).load_motion(str(path),str(project/'mahr/rsc/amass_bvh/norm_pose'),motion_range=(0,5))
        with tempfile.TemporaryDirectory() as tmp:
            _, graphs, global_dict,_=generate_vector_graphs_for_motion(mocap,Path(tmp)/'motions',Path(tmp)/'graphs',normalize_edge_weights=True)
        expected=np.array([[g.get_edge_data(mocap.key_bodies[a],mocap.key_bodies[b])['position_vector'] for a,b in e.pairs] for g in graphs])
        # Legacy bvhio/poselib uses float32 rotations; new decode is float64.
        assert_allclose(v,expected,atol=3e-5,rtol=3e-5)
        report[pose]=dict(target_vector_max_error=max_target,
                         source_vector_max_error=float(np.max(np.abs(v-expected))))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
