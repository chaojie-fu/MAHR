"""Terrain scene geometry and mesh animation in explicit coordinate frames."""
import mujoco
import numpy as np

from .terrain import HeightMap


def set_mesh_vertices(model, geom_id, vertices, normals):
    """Update a dedicated mesh from vertices/normals in its attached body's frame.

    MuJoCo compiles mesh vertices into a centered, rotated frame and compensates
    with the geom transform. Preserve that transform (and its compiled same-frame
    flags); convert animated data into the compiled frame instead. Call
    ``mjr_uploadMesh`` afterwards when rendering with an existing GPU context.
    """
    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
        raise ValueError('Expected a mesh geom')
    mesh_id = int(model.geom_dataid[geom_id])
    if np.shape(vertices) != (model.mesh_vertnum[mesh_id], 3):
        raise ValueError('Vertex count must match the compiled mesh')
    if np.shape(normals) != (model.mesh_normalnum[mesh_id], 3):
        raise ValueError('Normal count must match the compiled mesh')
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
    rotation = rotation.reshape(3, 3)
    start = int(model.mesh_vertadr[mesh_id])
    model.mesh_vert[start:start + len(vertices)] = (vertices - model.geom_pos[geom_id]) @ rotation
    start = int(model.mesh_normaladr[mesh_id])
    model.mesh_normal[start:start + len(normals)] = normals @ rotation


def add_ground(spec: mujoco.MjSpec, terrain: HeightMap | None = None, *, grid=False):
    style = dict(name='floor', rgba=[.8, .8, .8, 1], contype=0, conaffinity=0)
    if grid:
        spec.add_texture(name='terrain_grid', type=mujoco.mjtTexture.mjTEXTURE_2D,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=128, height=128,
                         rgb1=[.65, .65, .65], rgb2=[.95, .95, .95])
        material = spec.add_material(name='terrain_grid', texuniform=True, texrepeat=[2., 2.],
                                     reflectance=0., specular=0.)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'terrain_grid'
        style.update(material='terrain_grid', rgba=[1., 1., 1., 1.])
    if terrain is None:
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, .01], **style)
        return
    minimum = float(terrain.heights.min())
    span = float(np.ptp(terrain.heights))
    # MuJoCo normalizes nonconstant samples to [0, 1]; retain their true span.
    # Constant maps still need a positive size, but their samples remain zero.
    if span == 0.:
        span = .001
    # MuJoCo rows increase along world Y, opposite to image rows.
    samples = np.flipud((terrain.heights - minimum) / span)
    spec.add_hfield(name='terrain', nrow=samples.shape[0], ncol=samples.shape[1],
                    userdata=samples.ravel().tolist(), size=[*terrain.half_size, span, .1])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname='terrain',
                            pos=terrain.origin + [0., 0., minimum], **style)


def add_environment(spec: mujoco.MjSpec, terrain: HeightMap | None = None, *, grid=True):
    """Shared playback environment: white sky, gray/white ground, and lighting."""
    spec.add_texture(name='sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     rgb1=[1., 1., 1.], rgb2=[1., 1., 1.], width=64, height=384)
    add_ground(spec, terrain, grid=grid)
    spec.worldbody.add_light(pos=[1, -2, 5], dir=[-1, 2, -5], diffuse=[.8, .8, .8])
    spec.visual.headlight.ambient = [.4, .4, .4]
    spec.visual.headlight.diffuse = [.7, .7, .7]
