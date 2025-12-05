import numpy
from perlin_noise import PerlinNoise

# a function to generate a normalized height map image using perlin noise
def generate_height_map(
    width: int,
    height: int,
    scale: float,
    octaves: int,
    png_path: str = None
) -> numpy.ndarray:

    height_map = numpy.zeros((height, width), dtype=numpy.float32)

    noise = PerlinNoise(octaves=octaves)

    for y in range(height):
        for x in range(width):
            height_value = noise([x / scale, y / scale])
            height_map[y, x] = height_value

    # Normalize to [0, 1]
    min_val = numpy.min(height_map)
    max_val = numpy.max(height_map)
    normalized_height_map = (height_map - min_val) / (max_val - min_val)

    if png_path is not None:
        from PIL import Image
        img = Image.fromarray((normalized_height_map * 255).astype(numpy.uint8))
        img.save(png_path)

    return normalized_height_map