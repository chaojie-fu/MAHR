"""Focused ground-component and weight ablations; patch only this analysis process."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from mahr_v2.pipeline import run
from mahr_v2.solver import FrameObjective,Settings
from ablate_x1_regression import metrics
from mahr_v2.pipeline import prepare


def ground_component(mode):
    def terms(self,x,raw,derivatives=None):
        weight=self.settings.ground_weight/getattr(self.edges,'normalization',{}).get('target_height',1.)**2
        gradient=np.zeros_like(x);value=0.
        if mode=='markers':
            depth=np.maximum(raw[self.edges.ground,2],0.)
            value=weight*np.dot(depth,depth)
            if derivatives is not None:gradient=2*weight*depth@derivatives[self.edges.ground,2,:]
        else:
            value,gradient=self.robot.ground_penalty(self.rotation,x[-1],jacobian=derivatives is not None)
            value*=weight;gradient*=weight
        return float(value),gradient
    return terms


def main():
    project=Path(__file__).resolve().parents[1];legacy=project.parent/'MAHR'
    out=project/'results/x1_factorial';motion=legacy/'mahr/rsc/amass_bvh/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh'
    robot,_,_,_,_,_,initial=prepare(legacy,motion,robot_file='AgibotX1Config.py')
    cases=[]
    for n,c,f in ((0,0,0),(1,1,0),(1,1,1)):
        for component in ('markers','surfaces'):
            cases.append((f'N{n}_C{c}_F{f}_{component}',n,c,f,component,10000.,True))
    for f in (0,1):cases.append((f'N1_C1_F{f}_all_T0',1,1,f,'all',10000.,False))
    for weight in (1.,10.,100.,1000.):cases.append((f'N1_C1_F1_all_W{int(weight)}',1,1,1,'all',weight,True))
    report={}
    for name,n,c,f,component,weight,temporal in cases:
        settings=Settings(posture_weight=0.,foot_support_weight=0.,ground_weight=weight,temporal_weight=.02 if temporal else 0.,
            acceleration_weight=.1 if temporal else 0.,max_joint_velocity=8 if temporal else None,
            max_root_velocity=2 if temporal else None)
        kwargs=dict(robot_file='AgibotX1Config.py',min_edge_ratio=.05 if n else 0,
                    clamp_source_ground=bool(c),source_ground_calibration=bool(f),settings=settings)
        path=out/f'{name}.json'
        if component=='all':stats=run(legacy,motion,path,**kwargs)
        else:
            with patch.object(FrameObjective,'ground_terms',ground_component(component)):
                stats=run(legacy,motion,path,**kwargs)
        measured=metrics(path,robot,initial)
        measured.update(name=name,component=component,weight=weight,temporal=temporal,
                        failed_frames=stats['unsuccessful_frames'],solve_seconds=stats['solve_seconds'])
        report[name]=measured
        (out/f'{name}.metrics.json').write_text(json.dumps(measured,indent=2))
        (out/'components_summary.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(measured),flush=True)


if __name__=='__main__':main()
