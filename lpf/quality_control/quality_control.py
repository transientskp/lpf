from typing import Union
import numpy as np
import torch
from lpf.bolts.torch import nanmax, nanmean


class QualityControl:
    def __init__(
        self, rfi_threshold: float = 5, corruption_threshold: float = 1e5,
    ) -> None:
        self.rfi_threshold = rfi_threshold
        self.corruption_threshold = corruption_threshold

    def filter_corruption_torch(self, images: torch.Tensor):
        assert len(images.shape) == 3
        abs_images: torch.Tensor = abs(images)
        max_values: torch.Tensor = nanmax(abs_images, axis=(-1, -2))
        for i, v in enumerate(max_values):  # type:ignore
            if v > self.corruption_threshold:
                images[i] = torch.zeros_like(images[i])

        return images

    def filter_rfi_torch(self, images: torch.Tensor):
        median_image: torch.Tensor = torch.median(images, dim=0, keepdims=True)[0]  # type: ignore
        scores: np.ndarray = nanmean((images - median_image) ** 2, axis=(-1, -2)).to("cpu").numpy()  # type: ignore
        median = np.median(scores)
        mad_std = 1.482602218505602 * np.median(abs(scores - median))

        z = (scores - median) / mad_std
        for i, z in enumerate(z):
            if z > self.rfi_threshold:
                images[i] = torch.zeros_like(images[i])

        return images

    def filter_corruption_cpu(self, images: np.ndarray):
        max_values = np.nanmax(abs(images), axis=(-1, -2))

        for i, v in enumerate(max_values):
            if v > self.corruption_threshold:
                images[i] = np.zeros_like(images[i])

        return images

    def filter_rfi_cpu(self, images: np.ndarray):
        median_image = np.median(images, axis=0, keepdims=True)
        scores = np.nanmean((images - median_image) ** 2, axis=(-1, -2))
        median = np.median(scores)
        mad_std = 1.482602218505602 * np.median(abs(scores - median))

        z = (scores - median) / mad_std

        for i, z in enumerate(z):
            if z > self.rfi_threshold:
                images[i] = np.zeros_like(images[i])

        return images

    def filter_bad_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, torch.Tensor):
            images = self.filter_corruption_torch(images)
            images = self.filter_rfi_torch(images)

        else:
            images = self.filter_corruption_cpu(images)
            images = self.filter_rfi_cpu(images)

        return images

    def __call__(self, images: Union[torch.Tensor, np.ndarray]):
        return self.filter_bad_images(images)
