"""No source rotation tracking: sweep a non-saturating robot rest-pose prior."""
import json
from pathlib import Path
from mahr_v2.pipeline import prepare,run
from mahr_v2.solver import Settings
from audit_support_prior import audit

p=Path(__file__).resolve().parents[1];legacy=p.parent/'MAHR';motion=legacy/'mahr/rsc/amass_bvh/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh';out=p/'results/posture_sweep';out.mkdir(parents=True,exist_ok=True);summary={}
for name,preset in [('x1','AgibotX1Config.py'),('h1','UnitreeH1.py'),('atlas','BostonDynamicsAtlasHDConfig.py')]:
 r,e,_,_,_,_,initial=prepare(legacy,motion,robot_file=preset)
 for weight in (.001,.003,.01,.03):
  key=f'{name}_{weight}';path=out/f'{key}.json';stats=run(legacy,motion,path,robot_file=preset,settings=Settings(ground_weight=10000.,foot_support_weight=0.,posture_weight=weight))
  m=audit(path,r,e,initial);m.update(mean_graph_loss=stats['mean_graph_loss'],failed=stats['unsuccessful_frames'],solve_seconds=stats['solve_seconds']);summary[key]=m
  (out/'summary.json').write_text(json.dumps(summary,indent=2));print(key,json.dumps(m),flush=True)
