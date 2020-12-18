"""
Author: David Ruhe
Date: 2020-12-04
"""
import numpy as np
from definitions import opt_str, opt_int
from astropy import table  # type: ignore
from scipy.ndimage import gaussian_filter  # type: ignore
from scipy.signal import fftconvolve  # type: ignore
from typing import Callable
from astropy.wcs import WCS  # type: ignore
from typing import Union
from astropy.wcs.utils import pixel_to_skycoord  # type: ignore
from lpf.source_finder import filter_nan


class SkySimulator:
    def __init__(
        self,
        size: int,
        psf: np.ndarray,
        n_channels: int,
        intensity_map: opt_str = None,
        source_radius: opt_int = None,
        intensity_map_multiplier: float = 1e-2,
        rfi_multiplier: float = 4,
    ):
        self.size = size
        if source_radius is None:
            source_radius = size // 2
        self.source_radius = source_radius
        self.intensity_map = intensity_map
        self.intensity_map_multiplier = intensity_map_multiplier
        self.psf = psf
        self.n_channels = n_channels
        self.rfi_multiplier: float = rfi_multiplier

        self.sky: np.ndarray= None  # type: ignore

    def sample_source_locations(self, n_sources: int) -> table.Table:
        image_center = self.size // 2
        a = np.random.rand(n_sources) * 2 * np.pi
        r = np.sqrt(np.random.rand(n_sources) * self.source_radius ** 2)
        x_l = np.int64(r * np.cos(a) + image_center)
        y_l = np.int64(r * np.sin(a) + image_center)
        return table.Table([x_l, y_l], names=("x_peak", "y_peak"))

    def sample_catalog(
        self, n_sources: int, flux_sample_fn: Callable[[int], np.ndarray], wcs: Union[WCS, None] = None, 
    ):
        catalog = self.sample_source_locations(n_sources)
        fluxes: np.ndarray = flux_sample_fn(n_sources)
        catalog["int_flux"] = fluxes
        if wcs is not None:
            catalog['coordinate'] = pixel_to_skycoord(catalog["y_peak"], catalog["x_peak"], wcs=wcs)
            catalog = filter_nan(catalog)

        return catalog

    def load_intensity_map(self) -> np.ndarray:
        if self.intensity_map is None:
            intensity_map = np.zeros((self.size, self.size))
        else:
            intensity_map = np.load(self.intensity_map)
            if intensity_map.shape != (self.size, self.size):
                raise ValueError(
                    "Provided intensity map has different shape than the specified image size."
                )

        return intensity_map

    def simulate_sky(
        self, catalog: table.Table, rfi: bool = False
    ) -> np.ndarray:

        intensity_map = self.load_intensity_map()

        sky = (
            np.random.randn(self.n_channels, *intensity_map.shape)
            + self.intensity_map_multiplier * intensity_map[None, ...]
        )

        if rfi:
            rfi_index = np.random.randint(0, self.n_channels)
        else:
            rfi_index = None

        for i, im in enumerate(sky):
            im[tuple(catalog[["x_peak", "y_peak"]].values())] = catalog["int_flux"]  # type: ignore

            if i == rfi_index:
                loc = self.sample_source_locations(1)
                im[tuple(loc[["x_peak", "y_peak"]].values())] = np.random.uniform(1, 4) * np.max(catalog["int_flux"])  # type: ignore

        sky: np.ndarray = fftconvolve(sky, self.psf[None, ...], mode="same", axes=(1, 2))  # type: ignore
        return sky # type: ignore

    def __call__(self, *args, **kwargs):  # type: ignore
        return self.simulate_sky(*args, **kwargs)  # type: ignore
