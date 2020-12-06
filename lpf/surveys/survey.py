"""
Author: David Ruhe
Date: 2020-12-06
Implements a base class that processes a directory containing .fits files into an iterable survey.
"""
import pandas as pd
from lpf.bolts.io import rglob
from typing import Tuple
import os


class Survey:
    def __init__(self, path: str, timestamp_start_stop: Tuple[int, int], subband_start_stop: Tuple[int, int]):

        self.survey_length = None
        self.timestamp_start_stop: Tuple[int, int] = timestamp_start_stop
        self.subband_start_stop: Tuple[int, int] = subband_start_stop

        data: pd.DataFrame = self._setup(path)
        self.__length: int = data.groupby("band").size().min()  # type: ignore

        self._sanity_checks(data)

        data = self._filter(data)
        self.data = self._add_time_index(data)

        self.indexed_data: pd.DataFrame = self.data.set_index(["time_index", "band"]).sort_index(  # type: ignore
            ascending=[True, False]
        )

    def _sanity_checks(self, data: pd.DataFrame):
        endtimes = data.groupby("band")["timestamp"].max()  # type: ignore
        min_endtime = endtimes.min()  # type: ignore
        diff_endtime = (endtimes - min_endtime).dt.seconds  # type: ignore
        assert (diff_endtime < 2).all()  # type: ignore

    def _filter(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.sort_values(["band", "timestamp"]).groupby("band").head(len(self))

    def _add_time_index(self, data: pd.DataFrame) -> pd.DataFrame:
        data["time_index"] = data.groupby("band").cumcount()  # type: ignore
        return data

    def _setup(self, path: str) -> pd.DataFrame:

        files = rglob(path, "*.fits")
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
                "timestamp": pd.to_datetime(timestamps),  # type: ignore
                "band": bands,
            }
        )
        return data

    def __len__(self):
        return self.__length

    def __getitem__(self, index: int) -> pd.DataFrame:
        sliced: pd.DataFrame = self.indexed_data.loc[index].dropna()  # type: ignore
        # TODO: for performance these can be removed at runtime.
        min_t = sliced['timestamp'].min()  # type: ignore
        diff_endtime = (sliced['timestamp'] - min_t).dt.seconds  # type: ignore
        assert (diff_endtime < 2).all(), sliced  # type: ignore
        return sliced