"""Height fields in world meters, shared by optimization and playback."""
from pathlib import Path

import numpy as np


class HeightMap:
    """Image rows run from +Y to -Y; columns run from -X to +X.

    Samples are heights relative to origin.z. Each cell uses the same triangle
    diagonal as MuJoCo's height fields. Queries outside the map clamp to its edge.
    """

    def __init__(self, heights, *, half_size=(20., 20.), origin=(0., 0., 0.)):
        self.heights = np.array(heights, dtype=float, copy=True)
        self.half_size = np.array(half_size, dtype=float, copy=True)
        self.origin = np.array(origin, dtype=float, copy=True)
        if (self.heights.ndim != 2 or min(self.heights.shape) < 2
                or not np.isfinite(self.heights).all()):
            raise ValueError('Height map must be a finite 2D array with at least two rows and columns')
        if (self.half_size.shape != (2,) or not np.isfinite(self.half_size).all()
                or np.any(self.half_size <= 0)):
            raise ValueError('Terrain half sizes must be two positive finite values')
        if self.origin.shape != (3,) or not np.isfinite(self.origin).all():
            raise ValueError('Terrain origin must contain three finite coordinates')
        for array in (self.heights, self.half_size, self.origin):
            array.setflags(write=False)
        self._pixel_scale = (np.array(self.heights.shape[::-1]) - 1) / (2 * self.half_size)

    @classmethod
    def from_png(cls, path, *, half_size=(20., 20.), elevation=1., origin=(0., 0., 0.)):
        """Load an 8-bit grayscale PNG: black=0, white=elevation meters."""
        from PIL import Image

        if not np.isfinite(elevation) or elevation <= 0:
            raise ValueError('Terrain elevation must be positive and finite')
        with Image.open(path) as image:
            if image.format != 'PNG' or image.mode != 'L':
                raise ValueError('Expected an 8-bit grayscale PNG (mode L)')
            heights = np.asarray(image, dtype=float) / 255. * elevation
        return cls(heights, half_size=half_size, origin=origin)

    def sample(self, xy, *, gradient=False):
        """Return world height and, optionally, its derivatives with respect to XY."""
        xy = np.asarray(xy, dtype=float)
        if xy.shape[-1:] != (2,) or not np.isfinite(xy).all():
            raise ValueError('Expected finite XY coordinates with shape [..., 2]')
        local = xy - self.origin[:2]
        pixel = (local * [1., -1.] + self.half_size) * self._pixel_scale
        last = np.array(self.heights.shape[::-1]) - 1
        pixel = np.clip(pixel, 0, last)
        cell = np.minimum(pixel.astype(int), last - 1)
        u, v = (pixel - cell)[..., 0], (pixel - cell)[..., 1]
        col, row = cell[..., 0], cell[..., 1]
        a, b = self.heights[row, col], self.heights[row, col + 1]
        c, d = self.heights[row + 1, col], self.heights[row + 1, col + 1]
        upper = u + v <= 1.
        heights = np.where(upper, a + (b-a)*u + (c-a)*v,
                           d + (c-d)*(1-u) + (b-d)*(1-v)) + self.origin[2]
        if not gradient:
            return heights
        slopes = np.stack((np.where(upper, b-a, d-c), np.where(upper, c-a, d-b)), axis=-1)
        slopes *= self._pixel_scale * [1., -1.]
        slopes *= np.abs(local) < self.half_size
        return heights, slopes

    def save(self, path):
        """Store samples and coordinates without an external image dependency."""
        with Path(path).open('wb') as stream:
            np.savez_compressed(stream, heights=self.heights, half_size=self.half_size, origin=self.origin)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data['heights'], half_size=data['half_size'], origin=data['origin'])

    def describe(self):
        return dict(shape=list(self.heights.shape), half_size=self.half_size.tolist(),
                    origin=self.origin.tolist(), interpolation='triangular', boundary='clamp',
                    min_height=float(self.heights.min() + self.origin[2]),
                    max_height=float(self.heights.max() + self.origin[2]))
