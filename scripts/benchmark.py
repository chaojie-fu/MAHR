"""Controlled comparison: same frames, objective, bounds, initialization and tolerances.

Run with PYTHONPATH=src and optional legacy dependencies on PYTHONPATH.
The legacy-graph arm uses actual old graph/loss functions and urdfpy FK; its
first frame uses local optimization as well to isolate runtime implementation.
"""
import argparse
import platform
import sys
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize

from mahr_v2.pipeline import prepare
from mahr_v2.solver import Retargeter, Settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path('../MAHR'))
    parser.add_argument('--frames', type=int, default=50)
    parser.add_argument('--legacy-graph', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('results/benchmark.json'))
    args = parser.parse_args()
    p = args.project / 'mahr/rsc/amass_bvh/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh'
    robot, edges, scales, source, motion, _, initial = prepare(args.project, p, frame_slice=slice(0,args.frames), min_edge_ratio=0, clamp_source_ground=False, foot_surfaces=False, source_ground_calibration=False)
    reports = {}
    settings = Settings(posture_weight=0.,foot_support_weight=0.,ground_weight=0, temporal_weight=0, acceleration_weight=0, max_joint_velocity=None, max_root_velocity=None)
    for label, analytic in [('analytic',True), ('array_numerical',False)]:
        start = perf_counter()
        trajectory, stats = Retargeter(robot,edges,scales,settings).solve(source,motion.root_rotations,initial,analytic)
        reports[label] = dict(seconds=perf_counter()-start, evaluations=sum(s['evaluations'] for s in stats),
                              mean_loss=float(np.mean([s['loss'] for s in stats])),
                              unsuccessful_frames=sum(not s['success'] for s in stats),
                              frame_losses=[s['loss'] for s in stats])
    if args.legacy_graph:
        import networkx as nx
        # Compatibility alias restricted to the legacy benchmark process.
        np.float = float
        from urdfpy import URDF
        from mahr_v2.config import legacy_preset
        from mahr.utils.graph_distance.robot_graph_generator import generate_vector_symbolic_graph_for_robot
        from mahr.utils.graph_distance.compute_graph_distance import compute_vector_graph_distance
        preset = legacy_preset(args.project)
        old_robot = URDF.load(preset['urdf'])
        config = SimpleNamespace(robot=old_robot,key_joints=preset['key_joints'],
                                 toes=preset['toes'],norm_pose_in_rad=preset['norm_pose_in_rad'])
        _, _, _, graph_fn, _ = generate_vector_symbolic_graph_for_robot(config, normalize_edge_weights=True)
        # URDF and MuJoCo expose different joint ordering; map explicitly.
        order = np.array([robot.names.index(n) for n in old_robot.actuated_joint_names])
        keys = preset['key_joints']
        lower, upper = np.r_[robot.lower,0],np.r_[robot.upper,4]
        x = initial.copy(); stats=[]
        start = perf_counter()
        for frame, (sv, rot) in enumerate(zip(source,motion.root_rotations)):
            src=nx.Graph()
            for (a,b), v in zip(edges.pairs,sv):
                src.add_edge(keys[a],keys[b],position_vector=v)
            def objective(q):
                graph=graph_fn(q[:-1][order],rot,0.,0.,q[-1])
                return (compute_vector_graph_distance(src,keys,graph,keys,True)
                        +1e-3*np.sum(np.clip(q,-.78,.78)**2))
            lo,hi=lower,upper
            if frame:
                lo=np.maximum(lo,x-50/120); hi=np.minimum(hi,x+50/120)
            res=minimize(objective,x,method='L-BFGS-B',bounds=list(zip(lo,hi)),
                         options=dict(maxiter=2000,maxls=40))
            x=res.x
            stats.append(dict(loss=float(res.fun),evaluations=int(res.nfev),success=bool(res.success)))
        reports['legacy_graph_local_init']=dict(seconds=perf_counter()-start,
            evaluations=sum(s['evaluations'] for s in stats),mean_loss=float(np.mean([s['loss'] for s in stats])),
            unsuccessful_frames=sum(not s['success'] for s in stats),frame_losses=[s['loss'] for s in stats])
    import scipy, mujoco
    reports['metadata']=dict(environment=dict(python=sys.version.split()[0], numpy=np.__version__,
        scipy=scipy.__version__, mujoco=mujoco.__version__, platform=platform.platform()), frames=len(source),robot='Unitree G1',motion=p.name,
        first_frame='local optimization from rest pose in all arms; legacy global initialization excluded',
        timing='optimization only; decoding/model loading excluded',
        settings=settings.__dict__)
    for key in ('array_numerical','legacy_graph_local_init'):
        if key in reports:
            reports[key]['speedup_vs_analytic']=reports[key]['seconds']/reports['analytic']['seconds']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2))


if __name__ == '__main__':
    main()
