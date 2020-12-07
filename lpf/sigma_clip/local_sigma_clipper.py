import torch
from torch import nn
from typing import Union, List, Tuple
import numpy as np
from lpf.bolts.math import create_circular_mask


def local_sigma_clip(to_clip: torch.Tensor, kappa: float) -> List[torch.Tensor]:

    islands = torch.zeros_like(to_clip).float()

    while True:
        mean: torch.Tensor = to_clip.mean(1, keepdims=True)  # type: ignore
        std: torch.Tensor = to_clip.std(1, keepdims=True)  # type: ignore
        thresholds = mean + kappa * std
        # There might an approach more elegant than this for cases where the local peak
        # is lesser than zero.
        thresholds[thresholds < 0] = 0
        peaks = to_clip > thresholds

        if not peaks.any():
            break
            
        peaks = peaks.float()
        
        islands = islands + peaks * to_clip
        to_clip = to_clip - peaks * to_clip

    return [islands, to_clip]


class LocalSigmaClipper:
    def __init__(self, image_shape: Tuple[int, int], kappa: float, radius: int, kernel_size: int, stride: int):
        self.unfolder = nn.Unfold(kernel_size, stride=stride)
        self.radius = radius
        self.folder = nn.Fold(image_shape[0], kernel_size, stride=stride)
        self.mask: np.ndarray = create_circular_mask(*image_shape, radius=self.radius)
        self.kappa = kappa
        assert image_shape[0] % kernel_size == 0

    def __call__(self, images: Union[torch.Tensor, np.ndarray]) -> List[Union[np.ndarray, torch.Tensor]]:
        if isinstance(images, torch.Tensor):
            to_clip = images.clone()

        else:
            to_clip: torch.Tensor = torch.from_numpy(images).float()  # type: ignore

        to_clip[:, ~self.mask] = float("nan")

        to_clip = self.unfolder(to_clip[:, None])

        output = local_sigma_clip(to_clip, self.kappa)

        for i in range(len(output)):
            if isinstance(images, np.ndarray):
                output[i] = self.folder(output[i].float()).numpy().squeeze()
            else:
                output[i] = self.folder(output[i].float()).squeeze()

        return output  # type: ignore




