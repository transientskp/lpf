"""
Generates a 50/50 RFI classification dataset.
Author: David Ruhe
Date: 2020-12-04
"""
from lpf.simulation import SkySimulator
import numpy as np
from lpf.bolts.filters import circular_lowpass_filter
import numpy as np
import os
from astropy.table import Table  # type: ignore
from tqdm import trange  # type: ignore

IM_SIZE = 1024
N_CHANNELS = 16
BEAM_WIDTH = np.pi / 6
INTENSITY_MAP = "data/intensity_maps/1024x1024.npy"
INTENSITY_MULTIPLIER = 8e-2
N_IMAGES = 256
MU = 1.4
SIGMA = 1.1
N_SOURCES = IM_SIZE ** 2 // 1024
OUTPUT_DIR = f"data/rfi_detection/{IM_SIZE}x{IM_SIZE}"


def flux_sample_fn(n_sources: int):
    print(
        f"Simulating sources from log-normal at mean SNR (assuming a standard noise distribution) {np.exp(2 * MU + 2 * SIGMA ** 2):.4f}"
    )
    return np.exp(np.random.randn(n_sources) * SIGMA + MU)


if __name__ == "__main__":
    psf = circular_lowpass_filter(IM_SIZE, BEAM_WIDTH, order=1)
    skysim = SkySimulator(
        IM_SIZE,
        psf=psf,
        n_channels=N_CHANNELS,
        intensity_map=INTENSITY_MAP,
        intensity_map_multiplier=INTENSITY_MULTIPLIER,
    )
    catalog: Table = skysim.sample_catalog(N_SOURCES, flux_sample_fn)

    labels = []

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i in trange(N_IMAGES):  # type: ignore
        rfi = np.random.randn() < 0.5
        labels.append(rfi)
        image = skysim.simulate_sky(catalog, rfi)
        np.save(os.path.join(OUTPUT_DIR, f"{i:03}.npy"), image)

    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), np.array(labels))
