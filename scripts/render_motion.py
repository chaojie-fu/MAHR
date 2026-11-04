"""Headless mesh playback; requires ffmpeg and EGL or OSMesa, no dynamics stepping."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from mahr.rendering import add_ground
from mahr.terrain import HeightMap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trajectory', type=Path)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('-o', '--output', type=Path, required=True)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument('--stride', type=int, default=4)
    sampling.add_argument('--fps', type=float, help='Fixed output FPS; sample source timestamps at original speed')
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--bitrate', type=int, help='Target and maximum video bitrate in kbit/s')
    parser.add_argument('--ground-grid', action='store_true', help='Checker ground texture to show terrain shape')
    parser.add_argument('--camera-distance', type=float, default=3., help='Camera distance in meters (default: 3)')
    parser.add_argument('--enhance-dark-materials', action='store_true', help='Lift very dark display colors for visibility')
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('stride must be positive')
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error('fps must be positive and finite')
    if args.width < 2 or args.height < 2 or args.width % 2 or args.height % 2:
        parser.error('width and height must be positive even integers')
    if args.bitrate is not None and args.bitrate <= 0:
        parser.error('bitrate must be positive')
    if not math.isfinite(args.camera_distance) or args.camera_distance <= 0:
        parser.error('camera-distance must be positive and finite')
    payload = json.loads(args.trajectory.read_text())
    terrain = HeightMap.load(args.trajectory.parent / payload['terrain']) if 'terrain' in payload else None
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
    add_ground(spec, terrain, grid=args.ground_grid)
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
    camera.distance = args.camera_distance
    camera.azimuth = 140
    camera.elevation = -16
    fps = args.fps if args.fps is not None else payload['fps']/args.stride
    if args.fps is None:
        frames = payload['trajectory'][::args.stride]
    else:
        count = math.ceil(len(payload['trajectory']) * fps / payload['fps'])
        indices = np.minimum(np.rint(np.arange(count) * payload['fps'] / fps).astype(int),
                             len(payload['trajectory']) - 1)
        frames = [payload['trajectory'][index] for index in indices]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{args.width}x{args.height}', '-r', str(fps), '-i', '-', '-an',
           '-threads', '2', '-c:v', 'libx264']
    if args.bitrate is None:
        cmd += ['-crf', '20']
    else:
        cmd += ['-preset', 'slow', '-b:v', f'{args.bitrate}k', '-maxrate', f'{args.bitrate}k',
                '-bufsize', f'{2*args.bitrate}k', '-g', str(max(1, round(fps*2)))]
    cmd += ['-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output)]
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
