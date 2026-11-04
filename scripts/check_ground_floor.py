"""Audit support-surface clearances before/after floor and unilateral-ground changes."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from mahr_v2.pipeline import prepare
from retarget_all import PRESETS


def main():
    project=Path(__file__).resolve().parents[1]
    legacy=project.parent/'MAHR'
    output=project/'results/ground_floor'
    output.mkdir(parents=True,exist_ok=True)
    report={}
    for name,filename,old,new in [
        ('dance','DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh','temporal_validation_dance','ground_floor_all_dance'),
        ('smpl_h_small','smpl_h_small.bvh','temporal_validation_smpl','ground_floor_all_smpl')]:
        report[name]={}
        for robot_name,preset in PRESETS.items():
            r,e,s,v,clip,_,_=prepare(legacy,legacy/'mahr/rsc/amass_bvh'/filename,robot_file=preset)
            report[name][robot_name]={}
            for variant,folder in [('before',old),('after',new)]:
                path=project/'results'/folder/f'{robot_name}.json'
                payload=json.loads(path.read_text())
                marker,surface=[],[]
                for frame in payload['trajectory']:
                    p=r.evaluate(r.coordinates(frame['JointPos']),False)
                    rot=Rotation.from_quat(frame['RootOrient']).as_matrix()
                    height=frame['RootPos'][2]
                    marker.append(float((-e.vectors(p,rot,height)[e.ground,2]).min()))
                    clear=r.ground_clearances(rot,height)
                    # STL hull vertices start after the collision spheres.
                    surface.append(float(clear[len(r.ground_spheres):].min()))
                report[name][robot_name][variant]={
                    key:dict(min_clearance_m=float(np.min(values)),
                             frames_below_minus_1mm=int(np.sum(np.array(values)<-.001)))
                    for key,values in [('ground_markers',marker),('visual_foot_surface',surface)]}
            print(name,robot_name,report[name][robot_name],flush=True)
    (output/'clearance_comparison.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
