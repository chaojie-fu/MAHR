"""Cache STL foot surface support vertices; no mesh work inside optimization."""
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def stl_vertices(path):
    data=Path(path).read_bytes()
    count=struct.unpack_from('<I',data,80)[0] if len(data)>=84 else 0
    if len(data)==84+50*count:
        dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
        vertices=np.frombuffer(data,dtype=dtype,count=count,offset=84)['vertices'].reshape(-1,3)
    else:
        number=r'([-+\d.eE]+)'
        vertices=np.array(re.findall(r'\bvertex\s+'+number+r'\s+'+number+r'\s+'+number,data.decode()),dtype=float)
    vertices=np.unique(np.asarray(vertices,dtype=float),axis=0)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError(f'Invalid STL vertices: {path}')
    if len(vertices)>4 and np.linalg.matrix_rank(vertices-vertices[0])==3:
        vertices=vertices[ConvexHull(vertices).vertices]
    return vertices


def foot_support_meshes(robot, urdf, toes):
    """Find the nearest mesh-bearing ancestor of each toe. Fixed markers are skipped."""
    urdf=Path(urdf).resolve()
    root=ET.parse(urdf).getroot()
    links={node.get('name'):node for node in root.findall('link')}
    parents={joint.find('child').get('link'):joint.find('parent').get('link') for joint in root.findall('joint')}
    selected=set()
    for toe in toes:
        name=toe
        while name in links:
            if links[name].find('visual/geometry/mesh') is not None:
                selected.add(name);break
            name=parents.get(name)
        else:
            raise ValueError(f'No visual foot mesh ancestor for {toe}; disable foot surfaces explicitly for non-STL models')
    result=[]
    for name in sorted(selected):
        body=mujoco.mj_name2id(robot.model,mujoco.mjtObj.mjOBJ_BODY,name)
        for visual in links[name].findall('visual'):
            mesh=visual.find('geometry/mesh')
            if mesh is None:continue
            filename=urdf.parent/mesh.get('filename')
            if filename.suffix.lower()!='.stl':
                raise ValueError(f'Foot surface loader currently requires STL: {filename}')
            vertices=stl_vertices(filename)*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            origin=visual.find('origin')
            if origin is not None:
                vertices=vertices@Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix().T
                vertices+=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            result.append((body,vertices))
    return result
