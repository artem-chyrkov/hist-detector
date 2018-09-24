from math import sqrt
import numpy as np
from cv2 import imshow, waitKey


def draw_patches(patches_rgb, window_name='patches'):
    n = len(patches_rgb)
    if n == 0: return
    patch_height, patch_width, patch_depth = patches_rgb[0].shape

    rows = int(sqrt(n) + 0.5)
    max_rows = 750 // patch_height - 1
    if rows > max_rows: rows = max_rows
    cols = n // rows
    if n % rows != 0: cols += 1

    # image_to_show = np.zeros(((patch_height + 1) * rows, (patch_width + 1) * cols, patch_depth),
    #                          patches_rgb[0].dtype)
    image_to_show = np.full(((patch_height + 1) * rows, (patch_width + 1) * cols, patch_depth),
                            255, patches_rgb[0].dtype)
    pos_x, pos_y, i = 0, 0, 0

    for patch in patches_rgb:
        for y in range(patch_height):
            for x in range(patch_width):
                image_x = pos_x * (patch_width + 1) + x
                image_y = pos_y * (patch_height + 1) + y
                image_to_show[image_y, image_x] = patch[y, x]
        pos_x += 1
        if pos_x == cols:
            pos_x = 0
            pos_y += 1

    imshow(window_name, image_to_show)
    # waitKey(0)
