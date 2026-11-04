"""Compare X1 Dance foot geometry/angles before and after default source calibration."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from mahr_v2.pipeline import prepare


def main():
    project=Path(__file__).resolve().parents[1];legacy=project.parent/'MAHR'
    path=legacy/'mahr/rsc/amass_bvh/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh'
    robot,edges,scale,source,motion,_,initial=prepare(legacy,path,robot_file='AgibotX1Config.py')
    robot.evaluate(initial[:-1],False)
    rest_rotations={body:robot.data.xmat[body].reshape(3,3).copy() for body,_ in robot.ground_meshes}
    report={}
    for label,folder in [('before','ground_floor_all_dance'),('calibrated','calibrated_all_dance')]:
        payload=json.loads((project/'results'/folder/'x1.json').read_text())
        q=np.array([[f['JointPos'][n] for n in robot.names] for f in payload['trajectory']])
        angles={}
        for j,n in enumerate(robot.names):
            if any(t in n for t in ('ankle','hip_yaw')):
                angles[n]=dict(min_deg=float(np.rad2deg(q[:,j].min())),max_deg=float(np.rad2deg(q[:,j].max())),
                    limit_frames=int(np.sum((q[:,j]<robot.lower[j]+.001)|(q[:,j]>robot.upper[j]-.001))))
        minimums=[];tilts=[];yaws=[]
        for f,x in zip(payload['trajectory'],q):
            robot.evaluate(x,False);root=Rotation.from_quat(f['RootOrient']).as_matrix()
            minimums.append(float(robot.ground_clearances(root,f['RootPos'][2]).min()))
            frame_tilt=[];frame_yaw=[]
            for body,_ in robot.ground_meshes:
                relative=robot.data.xmat[body].reshape(3,3)@rest_rotations[body].T
                normal=root@relative[:,2]
                frame_tilt.append(float(np.rad2deg(np.arccos(np.clip(normal[2],-1,1)))))
                frame_yaw.append(float(np.rad2deg(np.arctan2(relative[1,0],relative[0,0]))))
            tilts.append(frame_tilt);yaws.append(frame_yaw)
        tilt=np.asarray(tilts);yaw=np.asarray(yaws)
        report[label]=dict(angles=angles,foot_tilt_max_deg=float(tilt.max()),
            foot_tilt_mean_deg=float(tilt.mean()),foot_frames_over_45deg=int(np.sum(np.max(tilt,axis=1)>45)),
            root_relative_foot_yaw_max_abs_deg=float(np.max(np.abs(yaw))),
            minimum_surface_clearance_m=float(np.min(minimums)),
            frames_below_minus_1mm=int(np.sum(np.array(minimums)<-.001)))
        print(label,json.dumps(report[label]),flush=True)
    out=project/'results/source_ground_calibration';out.mkdir(parents=True,exist_ok=True)
    (out/'x1_comparison.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
