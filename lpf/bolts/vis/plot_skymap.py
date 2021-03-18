# type: ignore
import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs.utils import skycoord_to_pixel  # type: ignore
import torch
from astropy.visualization import ZScaleInterval

def plot_skymap(
    image, skycoord=None, wcs=None, n_std=None, reverse=False, c="red", fname=None
):

    if skycoord is not None:
        pixels = list(skycoord_to_pixel(skycoord, wcs))

        if reverse:
            y, x = pixels[0], pixels[1]
            pixels[0] = x
            pixels[1] = y

    if isinstance(image, torch.Tensor):
        image = image.to("cpu").numpy()

    image = image.squeeze()

    fig = plt.figure(figsize=(16, 16))
    if n_std:
        vmin, vmax = ZScaleInterval().get_limits(image)
        plt.imshow(
            image, vmin=vmin, vmax=vmax, cmap="viridis", origin="lower"
        )
    else:
        vmin, vmax = ZScaleInterval().get_limits(image)
        plt.imshow(image, vmin=vmin, vmax=vmax,  cmap="viridis", origin="lower")
    plt.axis("off")
    plt.colorbar(orientation="horizontal", fraction=0.046, pad=0.04)

    if skycoord is not None:
        plt.scatter(
            pixels[0],
            pixels[1],
            s=128,
            edgecolor=c,
            facecolor="none",
            lw=1,
        )
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
    plt.close()
