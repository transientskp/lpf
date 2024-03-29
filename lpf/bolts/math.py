#type: ignore
import numpy as np


def odd_root(x, k):
    return np.sign(x) * np.abs(x) ** (1 / k)


def create_circular_mask(h, w, center=None, radius=None):

    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = ((X - center[0]) ** 2 + (Y - center[1]) ** 2) ** 0.5

    mask = dist_from_center < radius
    return mask