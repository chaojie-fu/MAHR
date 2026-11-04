"""Audit source landmarks, target markers, exact visual foot vertices and collision spheres.

Measures height relative to z=0 in the exported trajectories; does not change poses.
Run with trimesh available in the analysis environment.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh

from mahr_v2.pipeline import prepare


def stats(values, tolerance=1e-4):
    values=np.asarray(values)
    return dict(min_m=float(values.min()),max_m=float(values.max()),
                below_ground_frames=int(np.sum(np.min(values,axis=-1)<-tolerance)),
                worst_frame=int(np.argmin(np.min(values,axis=-1))))


def main():
    project=Path(__file__).resolve().parents[1]
    legacy=project.parent/'MAHR'
    urdf=legacy/'mahr/rsc/assets/urdf/unitree_g1/g1_29dof_stl.urdf'
    xml=ET.parse(urdf).getroot()
    geometry=[]
    for side in ('left','right'):
        link=xml.find(f"link[@name='{side}_ankle_roll_link']")
        mesh=link.find('visual/geometry/mesh')
        vertices=np.asarray(trimesh.load(urdf.parent/mesh.get('filename'),process=False).vertices)
        vertices=vertices*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
        origin=link.find('visual/origin')
        if origin is not None:
            vertices=vertices@Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix().T
            vertices+=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
        vertices=vertices[ConvexHull(vertices).vertices]
        spheres=[]
        for collision in link.findall('collision'):
            sphere=collision.find('geometry/sphere')
            if sphere is not None:
                spheres.append((np.fromstring(collision.find('origin').get('xyz','0 0 0'),sep=' '),float(sphere.get('radius'))))
        geometry.append((side+'_ankle_roll_link',vertices,spheres))
    report={}
    for name,old,motion in [('dance','all_robots','DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh'),
                            ('smpl_h_small','all_robots_smpl_h_small','smpl_h_small.bvh')]:
        robot,edges,scales,vectors,clip,ratio,initial=prepare(legacy,legacy/'mahr/rsc/amass_bvh'/motion)
        source=clip.positions[:,[clip.names.index('left_foot'),clip.names.index('right_foot')],2]
        source_ankle=clip.positions[:,[clip.names.index('left_ankle'),clip.names.index('right_ankle')],2]
        # Source BVH units are kept raw. No source surface meshes are available.
        report[name]=dict(source_foot_raw_units=dict(min=float(source.min()),max=float(source.max()),
                         below_zero_frames=int(np.sum(source.min(axis=1)<0))),
                         source_ankle_min_raw_units=float(source_ankle.min()))
        ground_indices=[np.flatnonzero((edges.a==i)&edges.ground)[0] for i in (3,6)]
        desired=-vectors[:,ground_indices,2]*scales[ground_indices]
        report[name]['target_toe_heights_if_ground_edges_match_exactly']=stats(desired)
        for variant,path in [('before',project/'results'/old/'g1.json'),('after_temporal_fix',project/'results/g1_continuous'/f'{name}.json')]:
            payload=json.loads(path.read_text())
            toe_heights=[];mesh_heights=[];collision_heights=[]
            for frame in payload['trajectory']:
                q=robot.coordinates(frame['JointPos'])
                p=robot.evaluate(q,jacobian=False)
                rz=Rotation.from_quat(frame['RootOrient']).as_matrix()[2]
                h=frame['RootPos'][2]
                toe_heights.append(p[[3,6]]@rz+h)
                min_mesh=[];min_collision=[]
                for link,vertices,spheres in geometry:
                    body=mujoco.mj_name2id(robot.model,mujoco.mjtObj.mjOBJ_BODY,link)
                    rotation=robot.data.xmat[body].reshape(3,3)
                    translation=robot.data.xpos[body]
                    zrow=rz@rotation
                    zoffset=rz@translation+h
                    min_mesh.append(float(np.min(vertices@zrow+zoffset)))
                    min_collision.append(min(float(center@zrow+zoffset-radius) for center,radius in spheres))
                mesh_heights.append(min_mesh);collision_heights.append(min_collision)
            toes=np.asarray(toe_heights);meshes=np.asarray(mesh_heights)
            report[name][variant]=dict(toe_markers=stats(toes),visual_foot_surface=stats(meshes),
                collision_spheres=stats(collision_heights),
                frames_surface_below_with_both_toes_above=int(np.sum((meshes.min(axis=1)<-1e-4)&(toes.min(axis=1)>=0))),
                max_marker_to_surface_gap_m=float(np.max(toes-meshes)))
    output=project/'results/g1_continuous/foot_ground_audit.json'
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
