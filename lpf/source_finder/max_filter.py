from lpf.bolts.torch.filters import MaxFilter
from scipy import ndimage  # type: ignore
import numpy as np
from typing import Union, List
import torch
from astropy.table import Table  # type: ignore
from astropy.wcs import WCS  # type: ignore
from astropy.wcs.utils import pixel_to_skycoord  # type: ignore


class SourceFinderMaxFilter:
    def __init__(self, image_size: int, duplicate_filter_degrees: int = 1):
        self.maxfilter_torch = MaxFilter(image_size)
        self.duplicate_filter_degrees = 1

    def __call__(
        self, peaks: Union[np.ndarray, torch.Tensor], wcs: Union[None, WCS] = None
    ) -> Table:

        maxima: Union[np.ndarray, torch.Tensor]
        peaks_mask: Union[np.ndarray, torch.Tensor]

        if isinstance(peaks, torch.Tensor):
            maxima = self.maxfilter_torch(peaks[:, None]).squeeze(1)
            peaks_mask = (maxima == peaks) & (peaks > 0)
            peak_locs = peaks_mask.nonzero(as_tuple=True)
            peak_values = maxima[peak_locs].to("cpu").numpy()  # type: ignore
            channel, x_locs, y_locs = (
                peak_locs[0].to("cpu").numpy(),
                peak_locs[1].to("cpu").numpy(),
                peak_locs[2].to("cpu").numpy(),
            )

        else:
            maxima = np.stack([ndimage.maximum_filter(image) for image in peaks])  # type: ignore
            peaks_mask = (maxima == peaks) & (peaks > 0)
            peak_locs = peaks_mask.nonzero()
            peak_values = maxima[peak_locs]

            channel, x_locs, y_locs = ( 
                peak_locs[0],
                peak_locs[1],
                peak_locs[2],
            )  # type: ignore

        channel: np.ndarray
        x_locs: np.ndarray
        y_locs: np.ndarray
        peak_values: np.ndarray

        # Construct the output Table
        colnames = ["channel", "x_peak", "y_peak", "peak_flux"]
        coldata: List[np.ndarray] = [
            channel,
            x_locs,
            y_locs,
            peak_values,
        ]

        table = Table(coldata, names=colnames)

        if wcs is not None:
            skycoord_peaks = pixel_to_skycoord(x_locs, y_locs, wcs, origin=0)
            table.add_column(skycoord_peaks, name="coordinate")  # type: ignore

        return table
