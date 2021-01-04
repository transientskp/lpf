from lpf.bolts.torch.filters import MaxFilter
from scipy import ndimage  # type: ignore
import numpy as np
from typing import Union, List
import torch
from astropy.table import Table  # type: ignore
from astropy.wcs import WCS  # type: ignore
from astropy.wcs.utils import pixel_to_skycoord  # type: ignore
from lpf.bolts.math import create_circular_mask

class SourceFinderMaxFilter:
    def __init__(self, image_size: int):
        self.maxfilter_torch = MaxFilter(image_size)

    def __call__(
        self, images, peaks: Union[np.ndarray, torch.Tensor], wcs: Union[None, WCS] = None
    ) -> Table:
        
        assert len(peaks.shape) == 3

        maxima: Union[np.ndarray, torch.Tensor]
        peaks_mask: Union[np.ndarray, torch.Tensor]

        model = peaks * images

        maxima = self.maxfilter_torch(model[:, None]).squeeze(1)
        peaks_mask = (maxima == model) & peaks
        peak_locs = peaks_mask.nonzero(as_tuple=True)
        peak_values = maxima[peak_locs].to("cpu").numpy()  # type: ignore
        channel, x_locs, y_locs = (
            peak_locs[0].to("cpu").numpy(),
            peak_locs[1].to("cpu").numpy(),
            peak_locs[2].to("cpu").numpy(),
        )

        channel: np.ndarray
        x_locs: np.ndarray
        y_locs: np.ndarray
        peak_values: np.ndarray

        # Construct the output Table
        colnames = ["channel", "x_peak", "y_peak", "detection_flux"]
        coldata: List[np.ndarray] = [
            channel,
            x_locs,
            y_locs,
            peak_values,
        ]

        table = Table(coldata, names=colnames)

        if wcs is not None:
            skycoord_peaks = pixel_to_skycoord(y_locs, x_locs, wcs, origin=0)
            table.add_column(skycoord_peaks, name="coordinate")  # type: ignore

        return table
