"""
Author: David Ruhe
Date: 2020-12-05
"""
import os
from tqdm import trange  # type: ignore
from lpf.simulation.sky import SkySimulator
from astropy.table import Table  # type: ignore
import numpy as np
from astropy.io import ascii, fits  # type: ignore
import datetime


class SurveySimulator:
    def __init__(
        self,
        skysim: SkySimulator,
        catalog: Table,
        output_dir: str,
        fits_template_file: str,
    ):
        self.catalog = catalog
        ascii.write(catalog, os.path.join(output_dir, "catalog.csv"), format="csv", fast_writer=False, overwrite=True)  # type: ignore
        self.skysim = skysim
        self.fits_template_file = fits_template_file
        self.output_dir = output_dir
        self.timestamp = datetime.datetime.now()

        self.hdu = fits_template_file


    def __call__(self, n_timesteps: int):

        t: int
        for t in trange(n_timesteps):  # type: ignore
            # f = os.path.join(self.output_dir, f"{t:03}.npy")

            sky = np.float32(self.skysim(self.catalog))
            for band, im in enumerate(sky):  # type: ignore
                im: np.ndarray
                band_str = f'S{band:03}'
                output_dir = os.path.join(self.output_dir, band_str)
                os.makedirs(output_dir, exist_ok=True)
                delta_t = self.timestamp + datetime.timedelta(seconds=t)
                t_str = delta_t.strftime("%Y-%m-%dT%H:%M:%S")
                filename = f'{t_str}-{band_str}.fits'
                filename = os.path.join(output_dir, filename)

                with fits.open(self.fits_template_file) as hdu:  # type: ignore
                    hdu[0].data = hdu[0].data * 0 + im[None, None]  # type: ignore
                    hdu.writeto(filename)  # type: ignore