"""
Author: David Ruhe
Date: 2020-12-06
Implements a base class that processes a directory containing .fits files into an iterable survey.
"""
import pandas as pd
from lpf.bolts.io import rglob
from typing import Tuple
import os
import logging

logger = logging.getLogger(__name__)

RAISE_NAN_WARN_FRACTION = 1 / 4


class Survey:
    def __init__(
        self,
        path: str,
        timestamp_start_stop: Tuple[int, int],
        subband_start_stop: Tuple[int, int],
        dt: int,
    ):

        self.survey_length = None
        self.timestamp_start_stop: Tuple[int, int] = timestamp_start_stop
        self.subband_start_stop: Tuple[int, int] = subband_start_stop

        self.data: pd.DataFrame = self._setup(path)
        len_before = len(self.data)
        self.data = self.data.drop_duplicates(["timestamp", "band"])
        len_after = len(self.data)
        if (len_before - len_after) > 0:
            logger.warning(
                "%s timestep-subband duplicates were found and dropped.",
                len_before - len_after,
            )
        self.data = self.data.set_index("timestamp")

        # This resamples the data onto the specified delta T grid. Filling NaN when images are missing.
        self.indexed_data = (
            self.data.pivot(columns="band")
            .resample(f"{dt}S")
            .first()
            .stack(dropna=False)
        )

        self.num_bands = len(self.indexed_data.index.unique(level=-1))

        assert (self.indexed_data.groupby("timestamp").size() == self.num_bands).all()
        self.indexed_data = self.indexed_data.sort_index(ascending=[True, False])
        self.time_index = self.indexed_data.index.unique(level=0)

    # def _filter(self, data: pd.DataFrame) -> pd.DataFrame:
    #     return data.sort_values(["band", "timestamp"]).groupby("band").head(len(self))

    # def _add_time_index(self, data: pd.DataFrame) -> pd.DataFrame:
    #     data["time_index"] = data.groupby("band").cumcount()  # type: ignore
    #     return data

    def _setup(self, path: str) -> pd.DataFrame:

        files = rglob(path, "*.fits")
        assert len(files) > 0
        timestamps = []
        bands = []
        for path in files:
            _, f = os.path.split(path)
            timestamps.append(f[slice(*self.timestamp_start_stop)])
            bands.append(f[slice(*self.subband_start_stop)])

        data = pd.DataFrame(
            {
                "file": files,
                "timestamp_str": timestamps,
                "timestamp": pd.to_datetime(timestamps, errors="coerce"),  # type: ignore
                "band": bands,
            }
        ).dropna()

        assert len(data) > 0, timestamps[:5]
        return data

    def __len__(self):
        return len(self.indexed_data)

    def __getitem__(self, index: int) -> pd.DataFrame:
        # sliced: pd.DataFrame = self.indexed_data.loc[index].dropna()  # type: ignore
        # TODO: for performance these can be removed at runtime.
        # min_t = sliced['timestamp'].min()  # type: ignore
        # diff_endtime = (sliced['timestamp'] - min_t).dt.seconds  # type: ignore
        # assert (diff_endtime < 2).all(), sliced  # type: ignore
        # return sliced
        timestep = self.indexed_data.loc[self.time_index[index]]
        num_nan = timestep.isna().any(axis=1).sum()
        if num_nan > len(timestep) * RAISE_NAN_WARN_FRACTION:
            logger.warning(
                "Encountered %s sub-bands without an image at index %s.", num_nan, index
            )
        return timestep
