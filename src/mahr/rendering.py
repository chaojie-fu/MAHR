"""Scene geometry for the same terrain used by the retargeting objective."""
import mujoco
import numpy as np

from .terrain import HeightMap


def add_ground(spec: mujoco.MjSpec, terrain: HeightMap | None = None, *, grid=False):
    style = dict(name='floor', rgba=[.22, .27, .33, 1], contype=0, conaffinity=0)
    if grid:
        spec.add_texture(name='terrain_grid', type=mujoco.mjtTexture.mjTEXTURE_2D,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=128, height=128,
                         rgb1=[.25, .30, .35], rgb2=[.38, .43, .48])
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
