# type: ignore
import torch
from astropy.table import Table
import numpy as np


def take_measurements(detected_sources, images, box_size) -> Table:
    # Take measurements
    measurements = []

    for i in range(len(detected_sources)):
        x = detected_sources["x_peak"][i]
        y = detected_sources["y_peak"][i]
        data = images[
            :,
            x - box_size // 2 : x + box_size // 2,
            y - box_size // 2 : y + box_size // 2,
        ]

        if isinstance(data, torch.Tensor):
            assert not torch.any(torch.isnan(data))
        else:
            assert not np.any(np.isnan(data))

        if isinstance(images, torch.Tensor):
            peak_flux = torch.amax(data, axis=(-1, -2))
            # integrated_flux = torch.sum(data, axis=(-1, -2))
            measurements.append(peak_flux.to("cpu").numpy())
        else:
            peak_flux = np.max(data, axis=(-1, -2))
            # integrated_flux = np.sum(data, axis=(-1, -2))
            measurements.append(peak_flux)

    detected_sources["peak_flux"] = measurements

    return detected_sources