import os
from mahr.utils.height_map_utils import generate_height_map

if __name__ == '__main__':
    png_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "height_map", 'height_map.png')
    os.makedirs(os.path.dirname(png_file), exist_ok=True)
    generate_height_map(1000, 1000, scale=100.0, octaves=2, png_path=png_file)
