"""
Generates a simulated intensity map. It includes surrogates for galactic emission & 
Cygnus-A and Cassiopeia-A calibration renmants.

Author: David Ruhe
Date: 2020-12-04
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.functional_models import Gaussian2D
from definitions import DATA_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default=3072, help="Size of image (edge)")
    args = parser.parse_args()

    ny, nx = args.size, args.size
    y, x = np.mgrid[:ny, :nx]

    # Simulates the galactic emission.
    gradient_corner = x * y
    gradient_corner = gradient_corner / np.abs(gradient_corner).max()

    gradient_diagonal = -1 * (np.abs(x - y) ** 0.7)
    gradient_diagonal = gradient_diagonal / np.abs(gradient_diagonal).max()

    gradient = gradient_diagonal + 1.5 * gradient_corner

    # Simulates renmants of calibration pipeline.
    x_mean = nx // 3
    y_mean = ny // 3

    amplitude = 0.9
    g_1 = Gaussian2D(amplitude, 2 * x_mean, y_mean, x_stddev=32, y_stddev=32)(x, y)
    g_2 = Gaussian2D(amplitude, x_mean, 2 * y_mean, x_stddev=32, y_stddev=32)(x, y)

    data = np.zeros((nx, ny)) + gradient / np.abs(gradient).max() + g_1 + g_2

    os.makedirs(os.path.join(DATA_DIR, "intensity_maps"), exist_ok=True)
    save_dir = os.path.join(DATA_DIR, "intensity_maps", f"{nx}x{ny}.npy")
    np.save(save_dir, data)
    
    print(f"Saved simulated intensity map to {save_dir}.")
