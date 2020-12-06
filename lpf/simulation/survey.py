"""
Author: David Ruhe
Date: 2020-12-05
"""
import os
from tqdm import trange  # type: ignore
from lpf.simulation.sky import SkySimulator
from astropy.table import Table  # type: ignore
import numpy as np
from astropy.io import ascii  # type: ignore


class SurveySimulator:
    def __init__(
        self,
        skysim: SkySimulator,
        catalog: Table,
        output_dir: str,
    ):
        self.catalog = catalog
        ascii.write(catalog, os.path.join(output_dir, "catalog.csv"), format="csv", fast_writer=False)  # type: ignore
        self.skysim = skysim

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def __call__(self, n_timesteps: int):

        for t in trange(n_timesteps):  # type: ignore
            f = os.path.join(self.output_dir, f"{t:03}.npy")

            sky = np.float32(self.skysim(self.catalog))
            np.save(f, sky)