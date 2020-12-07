"""
Author: David Ruhe
Date: 2020-12-06
Implements functionalities for local statistics estimation.
"""
from typing import Union
import torch
import numpy as np
from lpf.bolts.torch.filters import GaussianFilter
from scipy.ndimage import gaussian_filter  # type: ignore

# from typing import Callable


class StatisticsEstimator:
    # TODO: Need better name.
    def __init__(
        self,
        intensity_sigma: float,
        variability_sigma: float,
        stride: int = 4,
        truncate: float = 4,
        fillna: bool = False,
    ):
        self.intensity_filter_torch = GaussianFilter(
            intensity_sigma, stride, truncate, fillna
        )
        self.variability_filter_torch = GaussianFilter(
            variability_sigma, stride, truncate, fillna
        )

        self.intensity_sigma = intensity_sigma
        self.variability_sigma = variability_sigma
        self.truncate = truncate

    def batch_gaussian_filter_np(self, images: np.ndarray):
        # TODO, This can be sped up by using FFT convolutions.
        return np.stack([gaussian_filter(image, self.intensity_sigma, truncate=self.truncate) for image in images])  # type: ignore

    def get_intensity_variability_maps(self, images: Union[torch.Tensor, np.ndarray]):

        if isinstance(images, torch.Tensor):
            images = images[:, None]
            intensity_map = self.intensity_filter_torch(images)
            intensity_subtracted = images - intensity_map
            variability_map = torch.sqrt(
                self.variability_filter_torch((images - intensity_map) ** 2)
            )
        else:
            intensity_map = self.batch_gaussian_filter_np(images)
            intensity_subtracted = images - intensity_map
            variability_map = np.sqrt(self.batch_gaussian_filter_np(images))

        return (
            intensity_map.squeeze(),
            variability_map.squeeze(),
            intensity_subtracted.squeeze(),
        )

    def __call__(self, images: Union[torch.Tensor, np.ndarray]):
        return self.get_intensity_variability_maps(images)
