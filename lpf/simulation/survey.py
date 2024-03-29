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
from skimage.transform import resize
import matplotlib.pyplot as plt


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
        print(f"Saving to {os.path.abspath(output_dir)}")
        self.timestamp = datetime.datetime.now()
        self.template: np.ndarray
        self.header: fits.Header
        self.template, self.header = fits.getdata(self.fits_template_file, header=True)  # type: ignore


    def __call__(self, n_timesteps: int):

        t: int
        for t in trange(n_timesteps):  # type: ignore
            im: np.ndarray
            sky = np.float32(self.skysim(self.catalog))

            delta_t = self.timestamp + datetime.timedelta(seconds=t)
            t_str = delta_t.strftime("%Y-%m-%dT%H:%M:%S")
            for band, im in enumerate(sky):  # type: ignore
                band_str = f'S{band:03}'
                output_dir = os.path.join(self.output_dir, band_str)
                os.makedirs(output_dir, exist_ok=True)
                filename = f'{t_str}-{band_str}.fits'
                filename = os.path.join(output_dir, filename)

                if t == 0:
                    mean = im.mean()
                    std = im.std()
                    plt.imsave(os.path.join(output_dir, 'example.png'), im, vmin=mean - 3 * std, vmax = mean + 3 * std)
                data = im[None, None]


                hdu = fits.PrimaryHDU(data, header=self.header)
                hdu.writeto(filename)  # type: ignore