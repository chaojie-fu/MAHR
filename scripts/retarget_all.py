"""Retarget and validate every fixed robot preset on the same full BVH clip."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize._numdiff import approx_derivative
from mahr_v2.config import legacy_preset
from mahr_v2.pipeline import run, prepare
from mahr_v2.solver import FrameObjective, Settings

PRESETS = {'g1':'UnitreeG1.py', 'h1':'UnitreeH1.py', 'gr1':'FourierGR1Config.py',
           'n1':'FourierN1Config.py', 'x1':'AgibotX1Config.py',
           'berkeley':'BerkeleyHumanoidConfig.py', 'atlas':'BostonDynamicsAtlasHDConfig.py',
           'icub':'iCubConfig.py'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=Path('../MAHR'))
    parser.add_argument('--motion',type=Path,help='BVH clip; defaults to bundled dance example')
    parser.add_argument('--output',type=Path,default=Path('results/all_robots'))
    parser.add_argument('--check-legacy-fk',action='store_true',help='Requires old urdfpy on PYTHONPATH')
    args=parser.parse_args()
    motion=args.motion or args.project/'mahr/rsc/amass_bvh/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh'
    args.output.mkdir(parents=True,exist_ok=True)
    report={}
    for name,preset in PRESETS.items():
        output=args.output/f'{name}.json'
        stats=run(args.project,motion,output,robot_file=preset)
        robot,edges,scales,source,clip,ratio,initial=prepare(args.project,motion,robot_file=preset)
        payload=json.loads(output.read_text())
        q=np.array([[f['JointPos'][n] for n in robot.names]+[f['RootPos'][2]] for f in payload['trajectory']])
        low,high=np.r_[robot.lower,0],np.r_[robot.upper,4]
        assert len(q)==len(clip.positions) and len(q)>0
        check_indices=sorted(set((0,len(q)//2,len(q)-1)))
        assert np.isfinite(q).all() and np.all(q>=low-1e-8) and np.all(q<=high+1e-8)
        assert len(q)==1 or np.max(np.abs(np.diff(q,axis=0)))<=50/120+1e-8
        max_error=0.
        # Real optimized states, avoid regularizer's non-differentiable clip boundary.
        for index in check_indices:
            x=q[index].copy()
            x[np.abs(np.abs(x)-.78)<1e-4]+=.001
            obj=FrameObjective(robot,edges,scales,source[index],clip.root_rotations[index],Settings())
            _,g=obj.value_gradient(x)
            numerical=approx_derivative(obj.value,x).ravel()
            error=float(np.max(np.abs(g-numerical)))
            assert np.allclose(g,numerical,atol=2e-6,rtol=1e-4),(name,index,error)
            max_error=max(max_error,error)
        fk_report={}
        if args.check_legacy_fk:
            np.float=float  # Compatibility only for the old reference module.
            from urdfpy import URDF
            old=URDF.load(legacy_preset(args.project,preset)['urdf'])
            error=0.
            for index in check_indices:
                actual=robot.evaluate(q[index,:-1],jacobian=False)
                expected=old.link_fk(payload['trajectory'][index]['JointPos'],use_names=True)
                error=max(error,float(np.max(np.abs(actual-np.array([expected[n][:3,3] for n in robot.landmarks])))))
            assert error<1e-7,(name,error)
            fk_report['legacy_fk_max_error']=error
        report[name]={k:v for k,v in stats.items() if k!='frames_detail'}
        report[name].update(fk_report)
        report[name].update(dofs=robot.dofs,gradient_max_error=max_error,
            finite_and_within_bounds=True,source_motion=str(motion.resolve()),source_fps=clip.fps,
            checked_frames=check_indices,urdf=legacy_preset(args.project,preset)['urdf'],
            trajectory=str(output))
        (args.output/'summary.json').write_text(json.dumps(report,indent=2))
        print(name,json.dumps(report[name]),flush=True)
    if any(v['unsuccessful_frames'] for v in report.values()):
        raise RuntimeError('Some frames did not converge; inspect per-frame diagnostics')


if __name__=='__main__':
    main()
