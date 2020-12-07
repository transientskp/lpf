import numpy as np
from scipy.ndimage.filters import _gaussian_kernel1d  # type: ignore

import torch
from torch.nn import functional as F

from .utils import same_padding


class GaussianFilter:
    # TODO: could be sped up by using FFT convolutions.
    def __init__(
        self,
        sigma: float,
        stride: int = 1,
        truncate: float = 4,
        fillna: bool = False,
        verbose: bool = True,
    ):
        radius = int(truncate * sigma + 0.5)
        self.k = _gaussian_kernel1d(sigma, 0, radius).astype(np.float32)
        f: np.ndarray = self.k[:, None] * self.k[None:]
        if verbose:
            print(f"Using kernel shape {f.shape}")

        assert np.isclose(f.sum(), 1), f.sum()  # type: ignore

        self.f: torch.Tensor = torch.from_numpy(f.astype(np.float32))[None, None]  # type: ignore

        self.fillna = fillna
        self.stride = stride

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        assert len(input.shape) == 4, "I require batch of shape [n, c, h, w]."
        assert input.shape[-1] == input.shape[-2], "I need a rectangular image."
        image_size = input.shape[-1]
        p = same_padding(input.shape[-1], len(self.k))
        if self.fillna:
            input[torch.isnan(input)] = 0
        mean = F.conv2d(
            input,
            self.f.to(input.device),  # Perhaps pre-load to device in __init__.
            padding=p,
            stride=self.stride,
        )

        if self.stride > 1:
            mean: torch.Tensor = F.interpolate(mean, image_size)  # type: ignore

        return mean



class MaxFilter:
    def __init__(self, image_size: int, k: int = 3):
        self.p: int = same_padding(image_size, k)
        self.k = k
    def __call__(self, input: torch.Tensor):
        return torch.max_pool2d(F.pad(input, (self.p, self.p, self.p, self.p), "reflect"), self.k, stride=1)