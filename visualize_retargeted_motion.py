#!/usr/bin/env python3
"""Visualize retargeted motion trajectories on a MuJoCo robot (modular refactor).

Key features:
- Loads *_gd_v2.json trajectory (named joints) and plays it on a URDF robot.
- Optional follow camera with offset, looping, and playback speed scaling.
- Optional offscreen recording (FFmpeg) with either a dedicated tracking camera (default)
  or mirroring the interactive viewer camera (use --mirror-camera to enable).
- Headless mode support (skip interactive viewer).

New CLI flags (see parse_args):
  --record <PATH>          Save an MP4 recording (offscreen renderer)
  --record-width/height    Resolution (default 1920x1080)
  --mirror-camera          Mirror interactive viewer camera for recording (default: use dedicated tracker)
  --fixed-camera NAME      Use a named model camera for recording (overrides mirror/tracker)
  --no-viewer              Run headless (still records if --record)
  --max-frames N           Limit total frames (debug / quick preview)

"""
from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, NamedTuple, Optional, Sequence, Tuple

import mujoco
import numpy as np
from mujoco import mjtBuiltin, mjtGeom, mjtJoint, mjtTexture, viewer

LOGGER = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------------
class FreeJointInfo(NamedTuple):
    name: str
    qpos_address: int
    body_id: int


@dataclass
class Recorder:
    """Encapsulates offscreen rendering + FFmpeg pipe."""
    renderer: mujoco.Renderer
    process: subprocess.Popen
    fps_out: float
    mirror: bool
    fixed_camera: Optional[str]
    camera_obj: mujoco.MjvCamera
    active: bool = True

    def write_frame(self, data: mujoco.MjData, source_cam: Optional[mujoco.MjvCamera] = None):
        if not self.active:
            return
        try:
            if self.fixed_camera is not None:
                # Use named / id camera directly
                self.renderer.update_scene(data, camera=self.fixed_camera)
            elif self.mirror and source_cam is not None:
                # Copy fields from interactive viewer camera
                cam = self.camera_obj
                cam.type = source_cam.type
                cam.fixedcamid = source_cam.fixedcamid
                cam.trackbodyid = source_cam.trackbodyid
                cam.distance = source_cam.distance
                cam.azimuth = source_cam.azimuth
                cam.elevation = source_cam.elevation
                cam.lookat[:] = source_cam.lookat
                self.renderer.update_scene(data, camera=cam)
            else:
                # Free camera fallback (keep current camera_obj settings)
                self.renderer.update_scene(data, camera=self.camera_obj)
            frame = self.renderer.render()  # (H,W,3) uint8
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            LOGGER.warning("FFmpeg pipe closed unexpectedly; stopping recording.")
            self.active = False
        except Exception as exc:
            LOGGER.error("Recording error: %s", exc)
            self.active = False

    def finalize(self):
        if self.process and self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except Exception:
                pass
        if self.process:
            try:
                self.process.wait(timeout=5)
            except Exception:
                pass
        self.active = False


# ----------------------------------------------------------------------------------
# CLI & configuration
# ----------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play back a retargeted motion trajectory using MuJoCo (modular).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-t", "--trajectory", type=Path, required=True, help="*_gd_v2.json trajectory path")
    parser.add_argument("-u", "--urdf", type=Path, required=True, help="Robot URDF path")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True, help="Loop motion")
    parser.add_argument("--playback-speed", type=float, default=1.0, help="Playback speed scale")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index")
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive end frame index")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit processed frames (debug)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["CRITICAL","ERROR","WARNING","INFO","DEBUG"], help="Logging level")
    parser.add_argument("--height-map", type=str, default=None, help="Height map PNG file path")
    # can I set None as default for --camera-offset?
    parser.add_argument("--camera-offset", type=float, nargs=3, metavar=("X","Y","Z"), default=None, help="Follow camera offset")
    parser.add_argument("--record", type=Path, default=None, help="MP4 output (enable recording)")
    parser.add_argument("--record-width", type=int, default=1920, help="Recording width")
    parser.add_argument("--record-height", type=int, default=1080, help="Recording height")
    parser.add_argument("--mirror-camera", action="store_true", help="Mirror interactive viewer camera for recording (default: off; use dedicated tracking camera)")
    parser.add_argument("--fixed-camera", type=str, default=None, help="Named model camera for recording (overrides mirror)")
    parser.add_argument("--no-viewer", action="store_true", help="Run headless (still records if --record)")
    parser.add_argument("--ffmpeg-crf", type=int, default=20, help="FFmpeg CRF quality (lower=better)")
    parser.add_argument("--ffmpeg-preset", type=str, default="veryfast", help="FFmpeg preset")
    return parser.parse_args()


# ----------------------------------------------------------------------------------
# Trajectory utilities
# ----------------------------------------------------------------------------------

def load_trajectory(path: Path) -> Tuple[float, Sequence[Mapping[str, object]]]:
    with path.open("r") as fh:
        payload = json.load(fh)
    if "trajectory" not in payload:
        raise KeyError(f"Trajectory file '{path}' missing 'trajectory'.")
    frames = payload["trajectory"]
    if not isinstance(frames, Sequence):
        raise TypeError("'trajectory' must be a sequence of frames.")
    fps = float(payload.get("fps", 30.0))
    return fps, frames


def slice_trajectory(frames: Sequence[Mapping[str, object]], start: int, end: Optional[int]) -> Sequence[Mapping[str, object]]:
    total = len(frames)
    start = max(0, min(start, total - 1))
    end = total if end is None else max(start + 1, min(end, total))
    return frames[start:end]


# ----------------------------------------------------------------------------------
# Quaternion & joint utilities
# ----------------------------------------------------------------------------------

def quaternion_xyzw_to_wxyz(quaternion: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(quaternion), dtype=float)
    if values.shape != (4,):
        raise ValueError(f"Expected 4D quaternion, got {values.shape}.")
    return np.array([values[3], values[0], values[1], values[2]], dtype=float)


def joint_sizes(model: mujoco.MjModel) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not name:
            continue
        jt = model.jnt_type[jid]
        sizes[name] = 7 if jt == mjtJoint.mjJNT_FREE else 4 if jt == mjtJoint.mjJNT_BALL else 1
    return sizes


def joint_qpos_addresses(model: mujoco.MjModel) -> Dict[str, int]:
    adr: Dict[str, int] = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name:
            adr[name] = int(model.jnt_qposadr[jid])
    return adr


def find_free_joint(model: mujoco.MjModel) -> Optional[FreeJointInfo]:
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mjtJoint.mjJNT_FREE:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            return FreeJointInfo(name=name, qpos_address=int(model.jnt_qposadr[jid]), body_id=int(model.jnt_bodyid[jid]))
    return None


# ----------------------------------------------------------------------------------
# Scene building & assets (reused from original, condensed)
# ----------------------------------------------------------------------------------

def _resolve_asset_path(base_dir: Path, path_value: str) -> str:
    if not path_value:
        return path_value
    p = Path(path_value)
    return str(p if p.is_absolute() else (base_dir / p).resolve())


def _ensure_free_root_joint(spec: mujoco.MjSpec) -> str:
    if not spec.worldbody.bodies:
        raise ValueError("URDF spec has no bodies.")
    root = spec.worldbody.bodies[0]
    for j in root.joints:
        if j.type == mjtJoint.mjJNT_FREE:
            return j.name or ""
    base = "root_free_joint"
    existing = {j.name for b in spec.worldbody.bodies for j in b.joints if j.name}
    name = base
    i = 1
    while name in existing:
        name = f"{base}_{i}"; i += 1
    root.add_joint(type=mjtJoint.mjJNT_FREE, name=name)
    return name


def _ensure_texture(spec: mujoco.MjSpec, *, name: str, texture_type: mjtTexture, builtin: mjtBuiltin, **kwargs):
    if any(t.name == name for t in spec.textures):
        return None
    return spec.add_texture(type=texture_type, builtin=builtin, name=name, **kwargs)


def _ensure_material(spec: mujoco.MjSpec, *, name: str, **kwargs):
    if any(m.name == name for m in spec.materials):
        return None
    return spec.add_material(name=name, **kwargs)

def convert_to_height_field_userdata(height_array: np.ndarray):
    new_height_array = np.zeros(height_array.shape)
    for i in range(height_array.shape[0]):
        for j in range(height_array.shape[1]):
            new_height_array[i][height_array.shape[1] - 1 - j] = height_array[j][i] # scale factor can be adjusted
    return new_height_array.ravel('F').tolist()

def _configure_scene_overlays(spec: mujoco.MjSpec, width: int, height: int, height_map_path: Optional[str] = None) -> None:
    ground_half_extent = 30.0
    ground_thickness = 0.02
    spec.stat.center = [1.0, 0.7, 1.0]
    spec.stat.extent = 0.8
    vis = spec.visual
    vis.headlight.diffuse = [0.6, 0.6, 0.6]
    vis.headlight.ambient = [0.1, 0.1, 0.1]
    vis.headlight.specular = [0.9, 0.9, 0.9]
    vis.rgba.haze = [0.15, 0.25, 0.35, 1.0]
    g = getattr(vis, "global")
    g.azimuth = -140; g.elevation = -20; g.offwidth = width; g.offheight = height

    _ensure_texture(spec, name="skybox_white", texture_type=mjtTexture.mjTEXTURE_SKYBOX, builtin=mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[1,1,1], rgb2=[1,1,1], width=800, height=800)
    _ensure_texture(spec, name="check_tex", texture_type=mjtTexture.mjTEXTURE_2D, builtin=mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[1.0, 1.0, 1.0], rgb2=[0.3,0.3,0.3], width=512, height=512)
    mat = _ensure_material(spec, name="floor_mat", emission=0.30, roughness=1.0, texuniform=False, texrepeat=[50,50], reflectance=0.0)
    if mat is not None:
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'check_tex'

    size = [ground_half_extent, ground_half_extent, ground_thickness]
    floor = next((gm for gm in spec.worldbody.geoms if gm.name == "floor"), None)
    if floor is None:
        if height_map_path is None:
            spec.worldbody.add_geom(name="floor", type=mjtGeom.mjGEOM_PLANE, size=size, material="floor_mat", contype=0, conaffinity=0, friction=[0.6,0.6,0.6], condim=3, priority=1)
        else:
            from PIL import Image
            img = Image.open(height_map_path).convert('L')  # convert to grey scale image
            height_arr = np.array(img, dtype=np.float32)
            height_arr = height_arr / 255.0  # normalize the height map to [0, 1]
            height_flat = convert_to_height_field_userdata(height_arr)

            spec.add_hfield(
                name="heightfield",
                userdata=height_flat,
                # size=[1.0, 1.0, 1.0, 0.1],
                size=[20.0, 20.0, 1.0, 0.1],
                nrow=height_arr.shape[0],
                ncol=height_arr.shape[1],
            )
            spec.worldbody.add_geom(
                name="floor",
                type=mjtGeom.mjGEOM_HFIELD,
                hfieldname="heightfield",
                material="floor_mat",
                contype=0,
                conaffinity=0,
                friction=[0.6, 0.6, 0.6],
                condim=3,
                priority=1
            )
    else:
        floor.size = size; floor.material = "floor_mat"; floor.contype = 0; floor.conaffinity = 0; floor.friction = [0.6]*3; floor.condim = 3

    if not any(l.name == "sun_light" for l in spec.worldbody.lights):
        directional_enum = None
        if hasattr(mujoco, 'mjtLightType') and hasattr(mujoco.mjtLightType, 'mjLIGHT_DIRECTIONAL'):
            directional_enum = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        elif hasattr(mujoco, 'mjtLight') and hasattr(mujoco.mjtLight, 'mjLIGHT_DIRECTIONAL'):
            directional_enum = mujoco.mjtLight.mjLIGHT_DIRECTIONAL
        kwargs = dict(name="sun_light", pos=[0,0,3], diffuse=[0.1]*3, ambient=[0.1]*3, specular=[0.05]*3, dir=[0,0,-1], castshadow=True)
        if directional_enum is not None:
            kwargs['type'] = directional_enum
        spec.worldbody.add_light(**kwargs)


def build_scene_model(urdf_path: Path, width: int, height: int, height_map_path: Optional[str] = None) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(urdf_path))
    base = urdf_path.parent
    spec.meshdir = _resolve_asset_path(base, spec.meshdir)
    spec.texturedir = _resolve_asset_path(base, spec.texturedir)
    spec.modelfiledir = _resolve_asset_path(base, spec.modelfiledir)
    if spec.compiler is not None:
        spec.compiler.meshdir = _resolve_asset_path(base, spec.compiler.meshdir)
    _configure_scene_overlays(spec, width, height, height_map_path=height_map_path)
    root_name = _ensure_free_root_joint(spec)
    LOGGER.info("Using free root joint '%s' for root pose playback.", root_name or '<unnamed>')
    return spec.compile()


# ----------------------------------------------------------------------------------
# Frame application
# ----------------------------------------------------------------------------------

def apply_frame_to_state(frame: Mapping[str, object], base_qpos: np.ndarray, data: mujoco.MjData, qpos_lookup: Mapping[str, int], size_lookup: Mapping[str, int], free_joint: Optional[FreeJointInfo]) -> None:
    qpos = np.array(base_qpos, copy=True)
    if free_joint is not None:
        addr = free_joint.qpos_address
        root_pos = frame.get("RootPos")
        root_orient = frame.get("RootOrient")
        if root_pos is not None and root_orient is not None:
            try:
                qpos[addr:addr+3] = np.asarray(root_pos, dtype=float)
                qpos[addr+3:addr+7] = quaternion_xyzw_to_wxyz(root_orient)
            except Exception as exc:
                LOGGER.debug("Root pose skipped: %s", exc)
    jp = frame.get("JointPos", {})
    iterable = jp.items() if isinstance(jp, MutableMapping) else []
    for name, value in iterable:
        adr = qpos_lookup.get(name)
        if adr is None:
            continue
        size = size_lookup.get(name, 1)
        try:
            qpos[adr:adr+size] = np.asarray(value, dtype=float)
        except Exception as exc:
            LOGGER.debug("Joint '%s' skipped: %s", name, exc)
    data.qpos[:] = qpos
    data.qvel[:] = 0
    mujoco.mj_forward(data.model, data)


# ----------------------------------------------------------------------------------
# Camera utilities
# ----------------------------------------------------------------------------------

def _camera_offset_to_params(offset_vec: np.ndarray) -> Tuple[float, float, float]:
    dist = float(np.linalg.norm(offset_vec))
    if dist < 1e-8:
        raise ValueError("Camera offset magnitude too small.")
    direction = -offset_vec / dist
    horiz = float(np.hypot(direction[0], direction[1]))
    azimuth = float(np.degrees(np.arctan2(direction[1], direction[0])))
    elevation = float(np.degrees(np.arctan2(direction[2], max(horiz, 1e-9))))
    return dist, azimuth, elevation


def _configure_follow_camera(viewer_ctx: viewer.Viewer, model: mujoco.MjModel, data: mujoco.MjData, free_joint: Optional[FreeJointInfo], offset: Sequence[float]) -> Optional[Callable[[], None]]:
    if free_joint is None:
        LOGGER.warning("No free joint: follow camera disabled.")
        return None
    vec = np.asarray(offset, dtype=float)
    dist, azimuth, elevation = _camera_offset_to_params(vec)
    cam = viewer_ctx.cam
    tracking_type = getattr(mujoco.mjtCamera, 'mjCAMERA_TRACKING', None)
    cam.fixedcamid = -1; cam.distance = dist; cam.azimuth = azimuth; cam.elevation = elevation
    if tracking_type is not None:
        cam.type = tracking_type; cam.trackbodyid = free_joint.body_id
        return None
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    def _manual():
        cam.lookat[:] = data.xpos[free_joint.body_id]
    _manual(); return _manual


def _configure_tracking_mjv_camera(cam: mujoco.MjvCamera, data: mujoco.MjData, free_joint: Optional[FreeJointInfo], offset: Sequence[float]) -> Optional[Callable[[], None]]:
    """Configure a standalone MjvCamera to track the robot root (free joint) or stay fixed.

    Returns a callable that updates camera lookat each frame if manual tracking is needed,
    otherwise None (for native tracking camera types)."""
    vec = np.asarray(offset, dtype=float)
    try:
        dist, azimuth, elevation = _camera_offset_to_params(vec)
    except ValueError:
        dist, azimuth, elevation = 2.0, 90.0, -45.0  # fallback defaults
    cam.fixedcamid = -1
    cam.distance = dist
    cam.azimuth = azimuth
    cam.elevation = elevation
    tracking_type = getattr(mujoco.mjtCamera, 'mjCAMERA_TRACKING', None)
    if free_joint is not None and tracking_type is not None:
        cam.type = tracking_type
        cam.trackbodyid = free_joint.body_id
        return None
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.trackbodyid = -1
    def _manual():
        if free_joint is not None:
            cam.lookat[:] = data.xpos[free_joint.body_id]
    _manual()
    return _manual


# ----------------------------------------------------------------------------------
# Light tracking utilities
# ----------------------------------------------------------------------------------

def _configure_light_tracking_xy(model: mujoco.MjModel, data: mujoco.MjData, free_joint: Optional[FreeJointInfo], *, light_name: str = "sun_light") -> Optional[Callable[[], None]]:
    """Create an updater that keeps the given light's XY position aligned with the robot root.

    If the light or free joint is unavailable, returns None. Z is preserved from the initial
    light position so only XY tracks the body each frame.
    """
    if free_joint is None:
        LOGGER.warning("No free joint: light tracking disabled.")
        return None
    try:
        light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, light_name)
    except Exception:
        light_id = -1
    if light_id < 0:
        LOGGER.warning("Light '%s' not found: light tracking disabled.", light_name)
        return None
    # Preserve initial Z height of the light
    try:
        initial_z = float(model.light_pos[light_id][2])
    except Exception:
        # Fallback in case array layout differs
        initial_z = 3.0
    def _update():
        # Track the root body's current XY
        root_xy = data.xpos[free_joint.body_id][:2]
        model.light_pos[light_id][0] = float(root_xy[0])
        model.light_pos[light_id][1] = float(root_xy[1])
        model.light_pos[light_id][2] = initial_z
    # Initialize once so the first frame is already aligned
    _update()
    return _update


# ----------------------------------------------------------------------------------
# Recorder & playback loop
# ----------------------------------------------------------------------------------

def init_recorder(model: mujoco.MjModel, args: argparse.Namespace, fps: float) -> Optional[Recorder]:
    if not args.record:
        return None
    width, height = args.record_width, args.record_height
    try:
        renderer = mujoco.Renderer(model, height, width)
    except Exception as exc:
        LOGGER.error("Failed to create offscreen renderer: %s", exc)
        return None
    fps_out = fps * max(args.playback_speed, 1e-6)
    cmd = (
        f"ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size {width}x{height} "
        f"-r {fps_out} -i - -pix_fmt yuv420p -crf {args.ffmpeg_crf} -preset {args.ffmpeg_preset} {args.record}" )
    try:
        proc = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE)
    except FileNotFoundError:
        LOGGER.error("ffmpeg not found; recording disabled.")
        return None
    LOGGER.info("Recording to %s at %.2f FPS (%dx%d)", args.record, fps_out, width, height)
    return Recorder(renderer=renderer, process=proc, fps_out=fps_out, mirror=args.mirror_camera, fixed_camera=args.fixed_camera, camera_obj=mujoco.MjvCamera())


def default_camera_offset_based_on_initial_frame(quat):
    # use the quat multiple the vector (sqrt(2) / 2, sqrt(2) / 2, 0.0) to get a default offset
    q = quaternion_xyzw_to_wxyz(quat)
    # rotate vector (sqrt(2)/2, sqrt(2)/2, 0.0)
    v = np.array([2.0, 1.0, 0.0], dtype=float)
    # quaternion rotation: v' = q * v * q_conjugate
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)
    v_as_quat = np.array([0.0, v[0], v[1], v[2]], dtype=float)

    def quat_mult(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dtype=float)

    rotated_v = quat_mult(quat_mult(q, v_as_quat), q_conj)

    offset_vec = rotated_v[1: ]
    offset_vec[0:2] = offset_vec[0:2] / np.linalg.norm(offset_vec[0:2]) * 2.5  # set horizontal distance to 2.0
    offset_vec[2] = 0.5
    return offset_vec  # return the vector part


def run_playback(model: mujoco.MjModel, data: mujoco.MjData, frames: Sequence[Mapping[str, object]], fps: float, args: argparse.Namespace) -> None:
    if args.camera_offset is None:
        args.camera_offset = default_camera_offset_based_on_initial_frame(frames[0]["RootOrient"])
    free_joint = find_free_joint(model)
    qpos_lookup = joint_qpos_addresses(model)
    size_lookup = joint_sizes(model)
    base_qpos = np.array(data.qpos, copy=True)
    frame_dt = 1.0 / max(fps * max(args.playback_speed, 1e-6), 1e-6)
    recorder = init_recorder(model, args, fps)
    # Prepare tracking camera for headless or non-mirror recording
    rec_cam_update: Optional[Callable[[], None]] = None
    if recorder and (args.no_viewer or (not recorder.mirror and recorder.fixed_camera is None)):
        free_joint = find_free_joint(model)
        rec_cam_update = _configure_tracking_mjv_camera(recorder.camera_obj, data, free_joint, args.camera_offset)
    # Headless mode
    if args.no_viewer:
        LOGGER.info("Headless mode: viewer disabled.")
        free_joint = find_free_joint(model)
        qpos_lookup = joint_qpos_addresses(model)
        size_lookup = joint_sizes(model)
        base_qpos = np.array(data.qpos, copy=True)
        frame_dt = 1.0 / max(fps * max(args.playback_speed, 1e-6), 1e-6)
        # Light tracking updater
        light_update = _configure_light_tracking_xy(model, data, free_joint, light_name="sun_light")
        max_frames = args.max_frames if args.max_frames is not None else len(frames)
        frame_index = 0
        start_time = time.time()
        while frame_index < max_frames:
            frame = frames[frame_index]
            apply_frame_to_state(frame, base_qpos, data, qpos_lookup, size_lookup, free_joint)
            if light_update:
                light_update()
            if rec_cam_update:
                rec_cam_update()
            if recorder:
                recorder.write_frame(data)
            frame_index += 1
        if recorder:
            recorder.finalize()
        LOGGER.info("Headless playback completed in %.2fs", time.time() - start_time)
        return
    # Interactive viewer path
    free_joint = find_free_joint(model)
    qpos_lookup = joint_qpos_addresses(model)
    size_lookup = joint_sizes(model)
    base_qpos = np.array(data.qpos, copy=True)
    frame_dt = 1.0 / max(fps * max(args.playback_speed, 1e-6), 1e-6)
    max_frames = args.max_frames if args.max_frames is not None else len(frames)
    try:
        with viewer.launch_passive(model, data) as vctx:
            cam_updater = _configure_follow_camera(vctx, model, data, free_joint, args.camera_offset)
            # Light tracking updater
            light_update = _configure_light_tracking_xy(model, data, free_joint, light_name="sun_light")
            frame_index = 0
            next_frame_time = time.time()
            while vctx.is_running() and frame_index < max_frames:
                frame = frames[frame_index]
                apply_frame_to_state(frame, base_qpos, data, qpos_lookup, size_lookup, free_joint)
                if cam_updater:
                    cam_updater()
                if light_update:
                    light_update()
                if rec_cam_update:
                    rec_cam_update()  # update recording camera separately
                vctx.sync()
                if recorder:
                    # Mirror or use dedicated tracking camera
                    recorder.write_frame(data, source_cam=(vctx.cam if recorder.mirror else None))
                frame_index += 1
                if frame_index >= len(frames):
                    if args.loop:
                        frame_index = 0
                    else:
                        LOGGER.info("Reached end of trajectory; stopping viewer loop.")
                        break
                next_frame_time += frame_dt
                delay = next_frame_time - time.time()
                if delay > 0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    finally:
        if recorder:
            recorder.finalize()


# ----------------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    fps, frames = load_trajectory(args.trajectory)
    frames = slice_trajectory(frames, args.start_frame, args.end_frame)
    LOGGER.info("Loaded %d frames at %.2f FPS from '%s'", len(frames), fps, args.trajectory)

    # Validate format (must be dict-like JointPos)
    first_jp = frames[0].get("JointPos", {}) if frames else {}
    if isinstance(first_jp, Sequence) and not isinstance(first_jp, MutableMapping):
        raise ValueError("Positional JointPos format detected; require *_gd_v2.json named joint format.")

    model = build_scene_model(args.urdf, args.record_width, args.record_height, args.height_map)
    data = mujoco.MjData(model)
    run_playback(model, data, frames, fps, args)


if __name__ == "__main__":
    main()
