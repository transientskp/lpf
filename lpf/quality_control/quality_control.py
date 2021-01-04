import numpy as np
import torch
from lpf.bolts.torch import nanmax, nanmean


class QualityControl:
    def __init__(
        self, rfi_threshold: float = 5, corruption_threshold: float = 1e5,
    ) -> None:
        self.rfi_threshold = rfi_threshold
        self.corruption_threshold = corruption_threshold

    def filter_corruption(self, images: torch.Tensor):
        assert len(images.shape) == 3
        abs_images: torch.Tensor = abs(images)
        max_values: torch.Tensor = nanmax(abs_images, axis=(-1, -2))
        for i, v in enumerate(max_values):  # type:ignore
            if v > self.corruption_threshold:
                images[i] = torch.zeros_like(images[i])

        return images

    def filter_rfi(self, images: torch.Tensor):
        median_image: torch.Tensor = torch.median(images, dim=0, keepdims=True)[0]  # type: ignore
        scores: np.ndarray = nanmean((images - median_image) ** 2, axis=(-1, -2)).to("cpu").numpy()  # type: ignore
        median = np.median(scores)
        mad_std = 1.482602218505602 * np.median(abs(scores - median))

        z = (scores - median) / mad_std
        for i, z in enumerate(z):
            if z > self.rfi_threshold:
                images[i] = torch.zeros_like(images[i])

        return images

    def filter_corruption_np(self, images: np.ndarray):
        max_values = np.nanmax(abs(images), axis=(-1, -2))

        for i, v in enumerate(max_values):
            if v > self.corruption_threshold:
                images[i] = np.zeros_like(images[i])

        return images

    def filter_rfi_np(self, images: np.ndarray):
        median_image = np.median(images, axis=0, keepdims=True)
        scores = np.nanmean((images - median_image) ** 2, axis=(-1, -2))
        median = np.median(scores)
        mad_std = 1.482602218505602 * np.median(abs(scores - median))

        z = (scores - median) / mad_std

        for i, z in enumerate(z):
            if z > self.rfi_threshold:
                images[i] = np.zeros_like(images[i])

        return images

    def filter_bad_images(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore

        if images.device == torch.device('cpu'):
            print("WARNING: Running on CPU, switching to NumPy for quality control due to performance issues.")
            images: np.ndarray = self.filter_corruption_np(images.numpy())  # type: ignore
            images: np.ndarray = self.filter_rfi_np(images)  # type: ignore
            images: torch.Tensor = torch.from_numpy(images).float()  # type: ignore

        else:
            images: torch.Tensor = self.filter_corruption(images)
            images: torch.Tensor = self.filter_rfi(images)

        return images

    def __call__(self, images: torch.Tensor):
        return self.filter_bad_images(images)
