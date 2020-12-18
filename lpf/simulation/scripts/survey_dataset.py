"""
Generates a simulated observation.
Author: David Ruhe
Date: 2020-12-06
"""
from lpf.simulation import SkySimulator, SurveySimulator
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
# INTENSITY_MULTIPLIER = 1e-1
INTENSITY_MULTIPLIER = 5e-1
N_TIMESTEPS = 1
# MU = -0.6
# SIGMA = 1.6
MU = 1.4
SIGMA = 1.1
N_SOURCES = IM_SIZE ** 2 // 1024
OUTPUT_DIR = f"data/surveys/sim_{IM_SIZE}x{IM_SIZE}_background/"
TEMPLATE_FILE = '/project/druhe/ASf_201909011855/bands/295/2019-09-01T18:57:00-S295-B1.fits'
# TEMPLATE_FILE = '/project/druhe/2020-05-10T05:39:00-05:42:10_all/2020-05-10T05:38:59.2-SB308.fits'


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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    catalog: Table = skysim.sample_catalog(N_SOURCES, flux_sample_fn)

    survey = SurveySimulator(skysim, catalog, OUTPUT_DIR, TEMPLATE_FILE)

    survey(N_TIMESTEPS)