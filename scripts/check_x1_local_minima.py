"""Frozen-frame multi-start experiments with explicit feasible-set controls."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from mahr_v2.pipeline import prepare
from mahr_v2.solver import FrameObjective, Settings


def main():
    p=Path(__file__).resolve().parents[1];out=p/'results/x1_local_minima';out.mkdir(parents=True,exist_ok=True)
    robot,edges,scale,targets,motion,ratio,initial=prepare(p.parent/'MAHR',p.parent/'MAHR/mahr/rsc/amass_bvh/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh',robot_file='AgibotX1Config.py')
    def load(path):
        return np.array([[f['JointPos'][n] for n in robot.names]+[f['RootPos'][2]] for f in json.loads(path.read_text())['trajectory']])
    normal=load(p/'results/x1_factorial/N1_C1_F1_markers.json')
    original=load(p/'results/all_robots/x1.json')
    robot.evaluate(initial[:-1],False);rests={b:robot.data.xmat[b].reshape(3,3).copy() for b,_ in robot.ground_meshes}
    settings=Settings(ground_weight=10000.,posture_weight=0.,foot_support_weight=0.);rng=np.random.default_rng(20261104);rows=[]
    for config in ('baseline','pitch45_roll45'):
        path=p/f'results/x1_ankle_limits/{config}.json';seq=load(path)
        lower=np.r_[robot.lower,0.];upper=np.r_[robot.upper,4.]
        if config!='baseline':
            for j,n in enumerate(robot.names):
                if 'ankle' in n:lower[j]=max(lower[j],-np.pi/4);upper[j]=min(upper[j],np.pi/4)
        for frame in (85,120,250,350,450,490):
            prev=seq[frame-1];older=seq[frame-2];rotation=motion.root_rotations[frame]
            obj=FrameObjective(robot,edges,scale,targets[frame],rotation,settings,previous=prev,older=older,frame_dt=1/motion.fps)
            step=np.r_[np.full(robot.dofs,min(settings.step_limit,8/motion.fps)),min(settings.step_limit,2/motion.fps)]
            def describe(x,lo,hi):
                value,grad=obj.value_gradient(x);pos=robot.evaluate(x[:-1],False);raw=edges.vectors(pos,rotation,x[-1]);ground=obj.ground_terms(x,raw)[0];temporal=obj.temporal_terms(x)[0];prior=settings.regularization*np.dot(np.clip(x,-.78,.78),np.clip(x,-.78,.78))
                tilt=max(float(np.rad2deg(np.arccos(np.clip((rotation@robot.data.xmat[b].reshape(3,3)@r.T)[2,2],-1,1)))) for b,r in rests.items())
                clearance=float(robot.ground_clearances(rotation,x[-1]).min())
                return dict(loss=float(value),graph=float(value-ground-temporal-prior),ground=float(ground),temporal=float(temporal),tilt=tilt,min_clearance=clearance,projected_gradient_inf=float(np.max(np.abs(x-np.clip(x-grad,lo,hi)))))
            for domain in ('production_step_bounds','joint_bounds_only','geometry_only'):
                obj=FrameObjective(robot,edges,scale,targets[frame],rotation,settings,previous=None if domain=='geometry_only' else prev,older=None if domain=='geometry_only' else older,frame_dt=1/motion.fps)
                lo=np.maximum(lower,prev-step) if domain=='production_step_bounds' else lower.copy()
                hi=np.minimum(upper,prev+step) if domain=='production_step_bounds' else upper.copy()
                starts={'current':seq[frame],'previous':prev,'marker_reference':normal[frame],'historical_reference':original[frame]}
                for k in range(6):starts[f'perturb{k}']=seq[frame]+rng.normal(0,.2 if k<3 else .7,len(initial))
                trials=[]
                for name,start in starts.items():
                    x0=np.clip(start,lo,hi)
                    r=minimize(obj.value_gradient,x0,jac=True,bounds=list(zip(lo,hi)),method='L-BFGS-B',options=dict(maxiter=2000,maxls=40,ftol=settings.ftol,gtol=settings.gtol))
                    trials.append(dict(start=name,success=bool(r.success),message=str(r.message),iterations=int(r.nit),x=r.x.tolist(),**describe(r.x,lo,hi)))
                best=min(trials,key=lambda r:r['loss']);natural=[t for t in trials if t['tilt']<90]
                current=describe(seq[frame],lo,hi)
                # Central differences check full analytic gradient at the best solution.
                x=np.array(best['x']);value,g=obj.value_gradient(x);eps=1e-6
                numeric=np.array([(obj.value(x+np.eye(len(x))[j]*eps)-obj.value(x-np.eye(len(x))[j]*eps))/(2*eps) for j in range(len(x))])
                row=dict(config=config,frame=frame,domain=domain,current=current,trials=trials,best_start=best['start'],best_loss=best['loss'],best_tilt=best['tilt'],best_natural_loss=min(t['loss'] for t in natural) if natural else None,gradient_max_error=float(np.max(np.abs(g-numeric))))
                rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2));print(config,frame,domain,round(current['loss'],6),round(best['loss'],6),round(current['tilt'],1),round(best['tilt'],1),row['best_natural_loss'],flush=True)


if __name__=='__main__':main()
