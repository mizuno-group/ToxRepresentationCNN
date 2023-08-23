import multiprocessing
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import openslide
import torch
from PIL import Image
from skimage import io


def read_tiles(image_path: str, tile_dir: str, n_tiles: int, selected_tiles: Optional[List[str]] = None) -> Tuple[np.ndarray, int]:
    """
    This function returns randomly one tile image generated from the given WSI.
    Args:
        image_path (str):                   the path to the original WSI
        tile_dir (str):                     the path to the tile directory
        n_tiles (int):                      the number of tiles generated in the tiling process
        selected_tiles (List[str]|None):    If you set a list, then this function chooses a tile randomly from the list.
    Returns:
        tile (np.ndarray):                  a tile image in numpy array format
        idx (int):                          the index of the selected tile
    """
    if selected_tiles:
        idx = int(torch.randint(len(selected_tiles), (1,)).item())
        tile = io.imread(os.path.join(tile_dir, image_path.split(
            "/")[-1], f'{selected_tiles[idx]}.tiff'))
    else:
        idx = int(torch.randint(n_tiles, (1,)).item())
        tile = io.imread(os.path.join(
            tile_dir, image_path.split("/")[-1], f'{idx}.tiff'))
    return tile, idx


def _read_region(image_path: str, tile_size: int, x_padding1: int, y_padding1: int, i: int, j: int) -> Image.Image:
    """
    This function reads a region from the selected WSI.
    Args:
        image_path (str):       the path to the original WSI
        tile_size (int):        the length of a square
        x_padding1 (int):       padding size
        y_padding1 (int):       padding size
        i (int):                column
        j (int):                row
    Returns:
        tile (Image.Image):     a tile image in PIL Image format
    """
    ops = openslide.OpenSlide(image_path)
    return ops.read_region((x_padding1+i*(tile_size//2), y_padding1+j * (tile_size//2)), 0, (tile_size, tile_size)).convert("RGB")


def make_tiles(image_path: str, thumb_scale: int = 64, tile_size: int = 1000, n_tiles: int = 100) -> List[Image.Image]:
    """
    This function make tiles from the selected WSI.
    Args:
        image_path (str):       the path to the original WSI
        thumb_scale (int):      
        tile_size (int):        the length of a square
        n_tiles (int):              
    Returns:
        tile_list (List[Image.Image]):     a tile image in PIL Image format
    """
    ops = openslide.OpenSlide(image_path)
    x_orig, y_orig = ops.level_dimensions[0]

    thumb = ops.get_thumbnail((x_orig//thumb_scale, y_orig//thumb_scale))

    x_padding1 = (x_orig % (tile_size//2))//2
    y_padding1 = (y_orig % (tile_size//2))//2
    thumb_array = np.transpose(np.array(thumb), (1, 0, 2))

    x_num = x_orig//(tile_size//2) - 1
    y_num = y_orig//(tile_size//2) - 1
    conv_array = np.array([[np.mean(
        thumb_array[(x_padding1+i*(tile_size//2))//thumb_scale:(x_padding1+(i+1)*(tile_size//2))//thumb_scale,
                    (y_padding1+j*(tile_size//2))//thumb_scale:(y_padding1+(j+1)*(tile_size//2))//thumb_scale] < 200
    ) for j in range(y_num)] for i in range(x_num)])

    # return conv_array
    t_array = conv_array > 0.6
    t_array = t_array.reshape((conv_array.shape[0], conv_array.shape[1]))
    extraceted_tiles = [(i, j) for j in range(y_num)
                        for i in range(x_num) if t_array[i, j]]
    extraceted_tiles = random.sample(extraceted_tiles, n_tiles*3)
    cpu_count = min(multiprocessing.cpu_count(), 9)
    with multiprocessing.Pool(cpu_count-1) as p:
        async_list = [p.apply_async(_read_region,
                                    (image_path, tile_size, x_padding1, y_padding1,
                                     i, j)) for i, j in extraceted_tiles]
        tile_list = [f.get() for f in async_list]
    tile_list = [tile for tile in tile_list if (
        np.array(tile) < 200).mean() > 0.6][:n_tiles]
    return tile_list


def save_tiles(image_path: str, tile_size: int, n_tiles: int, tile_dir: str) -> None:
    tile_list = make_tiles(image_path=image_path,
                           tile_size=tile_size, n_tiles=n_tiles)
    for i, tile in enumerate(tile_list):
        tile.save(os.path.join(tile_dir, f'{i}.tiff'))
