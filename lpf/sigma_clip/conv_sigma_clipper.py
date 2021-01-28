import torch
import matplotlib.pyplot as plt
from torch.fft import fftn, ifftn
from numpy.compat import integer_types
from numpy.core import integer
import numpy as np
from lpf.bolts.math import create_circular_mask

integer_types = integer_types + (integer,)


def ifftshift(x, axes=None):
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, integer_types):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[ax] // 2) for ax in axes]

    return torch.roll(x, shift, axes)


def convolve_fft(array, kernel, axes=None):
    arrayfft = fftn(array, dim=axes)
    kernelfft = fftn(ifftshift(kernel, axes=axes), dim=axes)
    fftmult = kernelfft * arrayfft
    return torch.real(ifftn(fftmult, dim=axes))


class ConvSigmaClipper:
    def __init__(self, 
                 image_size: int, 
                 kappa: float, 
                 center_sigma: int, 
                 scale_sigma: int, 
                 radius: int,
                 maxiter: int,
                 device: str):

        kernel_boundary = (image_size - 1) / 2
        kernel_domain = torch.linspace(-kernel_boundary,
                                       kernel_boundary, image_size)

        center_gaussian_kernel1d = torch.exp(
            -(kernel_domain / center_sigma) ** 2 / 2).to(device)
        scale_gaussian_kernel1d = torch.exp(
            -(kernel_domain / scale_sigma) ** 2 / 2).to(device)

        self.center_filter = center_gaussian_kernel1d[:,
                                                      None] * center_gaussian_kernel1d[None, :]
        self.scale_filter = scale_gaussian_kernel1d[:,
                                                    None] * scale_gaussian_kernel1d[None, :]

        self.center_filter /= self.center_filter.sum()
        self.scale_filter /= self.scale_filter.sum()

        self.radius = radius
        self.mask: np.ndarray = create_circular_mask(image_size, image_size, radius=self.radius)
        self.mask = torch.from_numpy(self.mask).to(device)

        self.maxiter = maxiter
        self.kappa = kappa

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        x = images.clone()
        all_peaks = torch.zeros_like(x, dtype=bool)
        i = 1
        while True:
            center = convolve_fft(x, self.center_filter, axes=(-1, -2))
            scale = convolve_fft(
                (x - center) ** 2, self.scale_filter, axes=(-1, -2)).sqrt()

            peaks = x > center + self.kappa * scale

            if not peaks.any():
                if i == 1:
                    print("WARNING - Did not find any peaks, consider lowering kappa.")
                break

            all_peaks = all_peaks | peaks
            if i == self.maxiter:
                break

            i += 1

            x[peaks] = center[peaks]

        all_peaks = all_peaks & self.mask

        return all_peaks, center, scale
