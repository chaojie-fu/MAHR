"""Headless mesh playback; requires ffmpeg and EGL or OSMesa, no dynamics stepping."""
import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trajectory', type=Path)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--enhance-dark-materials', action='store_true', help='Lift very dark display colors for visibility')
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('stride must be positive')
    payload = json.loads(args.trajectory.read_text())
    root = ET.parse(args.urdf).getroot()
    for mesh in root.iter('mesh'):
        mesh.set('filename', str((args.urdf.resolve().parent / mesh.get('filename')).resolve()))
    for compiler in root.iter('compiler'):
        compiler.set('meshdir', '')
        compiler.set('discardvisual', 'false')
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding='unicode'))
    if args.enhance_dark_materials:
        for material in spec.materials:
            if np.mean(material.rgba[:3]) < .2:
                material.rgba[:3] = np.maximum(material.rgba[:3], .35)
        for geom in spec.geoms:
            if np.mean(geom.rgba[:3]) < .2:
                geom.rgba[:3] = np.maximum(geom.rgba[:3], .35)
    body = next(iter(spec.worldbody.bodies))
    body.add_joint(name='playback_root', type=mujoco.mjtJoint.mjJNT_FREE)
    spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[20, 20, .01], rgba=[.22, .27, .33, 1], contype=0, conaffinity=0)
    spec.worldbody.add_light(pos=[1, -2, 5], dir=[-1, 2, -5], diffuse=[.8, .8, .8])
    spec.visual.headlight.ambient = [.4, .4, .4]
    spec.visual.headlight.diffuse = [.7, .7, .7]
    spec.visual.global_.offwidth = args.width
    spec.visual.global_.offheight = args.height
    model = spec.compile()
    data = mujoco.MjData(model)
    free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'playback_root')
    base = int(model.jnt_qposadr[free])
    addresses = {model.joint(i).name: int(model.jnt_qposadr[i]) for i in range(model.njnt) if i != free}
    camera = mujoco.MjvCamera()
    camera.distance = 3.0
    camera.azimuth = 140
    camera.elevation = -16
    frames = payload['trajectory'][::args.stride]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{args.width}x{args.height}', '-r', str(payload['fps']/args.stride), '-i', '-', '-an',
           '-threads', '2', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output)]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
            for frame in frames:
                pos, quat = np.asarray(frame['RootPos']), np.asarray(frame['RootOrient'])
                data.qpos[base:base+3] = pos
                data.qpos[base+3:base+7] = quat[[3, 0, 1, 2]]
                for name, value in frame['JointPos'].items():
                    data.qpos[addresses[name]] = value
                mujoco.mj_forward(model, data)
                camera.lookat[:] = pos + [0, 0, .12]
                renderer.update_scene(data, camera)
                process.stdin.write(renderer.render().tobytes())
    finally:
        process.stdin.close()
        code = process.wait()
    if code:
        raise RuntimeError(f'ffmpeg exited with {code}')
    print(f'{args.output}: {len(frames)} frames, {len(payload["trajectory"])/payload["fps"]:.2f}s')


if __name__ == '__main__':
    main()
