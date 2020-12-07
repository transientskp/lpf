from typing import Union

import numpy as np
import torch
from definitions import opt_int
from lpf.bolts.math import create_circular_mask
from typing import Tuple
from lpf.bolts.torch import nanmean


def sigma_clip_torch(
    to_clip: torch.Tensor, kappa: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    peaks = torch.zeros_like(to_clip)
    is_nan = torch.isnan(to_clip)
    mean = nanmean(to_clip, is_nan=is_nan, axis=(-1, -2), keepdims=True)
    diff = to_clip - mean
    std = torch.sqrt(nanmean(diff ** 2, is_nan=is_nan, axis=(-1, -2), keepdims=True))
    islands: torch.Tensor = to_clip > mean + kappa * std
    while islands.any():
        islands = islands.float()
        peaks = peaks + islands * to_clip
        to_clip = to_clip - islands * to_clip
        is_nan = torch.isnan(to_clip)
        mean = nanmean(to_clip, is_nan=is_nan, axis=(-1, -2), keepdims=True)
        diff = to_clip - mean
        std = torch.sqrt(
            nanmean(diff ** 2, is_nan=is_nan, axis=(-1, -2), keepdims=True)
        )
        islands = to_clip > mean + kappa * std

    return peaks.squeeze(1), to_clip.squeeze(1)


def sigma_clip_np(to_clip: np.ndarray, kappa: float) -> Tuple[np.ndarray, np.ndarray]:
    peaks = np.float32(np.zeros_like(to_clip))
    mean = np.nanmean(to_clip, axis=(-1, -2), keepdims=True)
    std = np.nanstd(to_clip, axis=(-1, -2), keepdims=True)

    islands: np.ndarray = to_clip > mean + kappa * std

    while islands.any():

        islands = islands.astype(np.float32)

        peaks = peaks + islands * to_clip
        to_clip = to_clip - islands * to_clip

        mean = np.nanmean(to_clip, axis=(-1, -2), keepdims=True)
        std = np.nanstd(to_clip, axis=(-1, -2), keepdims=True)
        islands = to_clip > mean + kappa * std

    return peaks, to_clip  # type: ignore


class SigmaClipper:
    def __init__(self, image_shape: Tuple[int, int], kappa: int, radius: opt_int = None):
        self.radius = radius
        self.kappa = kappa
        self.mask: np.ndarray = create_circular_mask(*image_shape, radius=self.radius)
        pass

    def __call__(
        self, images: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:

        if isinstance(images, torch.Tensor):
            to_clip = images.clone()[:, None]
            to_clip[:, :, ~self.mask] = float("nan")
            return sigma_clip_torch(to_clip, self.kappa)
        else:
            to_clip = images.copy()[:, None]
            to_clip[:, :, ~self.mask] = float("nan")
            return sigma_clip_np(to_clip, self.kappa)
