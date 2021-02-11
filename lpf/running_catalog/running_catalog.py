from astropy import table  # type: ignore
from lpf.source_finder import filter_nan, filter_duplicates, take_measurements
from typing import Union, Any, List
import numpy as np
import torch
import pandas as pd
from numpy.lib.format import open_memmap
import os
import logging

logger = logging.getLogger(__name__)

tensor_or_ndarray = Union[np.ndarray, torch.Tensor]

class RunningCatalog:
    def __init__(self, output_folder: str, box_size: int, mmap_n_sources: int, mmap_n_timesteps: int, monitor_length: int, separation_crit: float):
        # See _initialize
        self.ns_id: int = None  # type: ignore
        self.timesteps: List[table.Table] = None  # type: ignore
        self.image_cache: list = None  # type: ignore
        self.meta = None
        self.flux_data: np.ndarray = None  # type: ignore

        self.output_folder = output_folder
        self.box_size = box_size
        self.mmap_n_sources = mmap_n_sources
        self.mmap_n_timesteps = mmap_n_timesteps
        self.monitor_length = monitor_length
        self.separation_crit = separation_crit

    def _initialize(
        self, detected_sources: table.Table, images: tensor_or_ndarray
    ) -> None:
        detected_sources = filter_nan(detected_sources)
        detected_sources = filter_duplicates(detected_sources)

        # Create new source IDs (starting from 0)
        detected_sources["id"] = range(len(detected_sources))
        detected_sources["new_source"] = True
        detected_sources = take_measurements(detected_sources, images, self.box_size)

        # Update new source ID.
        self.ns_id = len(detected_sources)

        # To keep track of the detected sources in each time-step.
        self.timesteps = [detected_sources]
        self.image_cache = [images.cpu()]
        # Storing source metadata.
        columns = ("start_t", "last_measured")
        meta = np.zeros([len(detected_sources), 2], dtype=int)
        self.meta = pd.DataFrame(meta, columns=columns)

        # Store all source data in a big memory map.
        # TODO: perhaps create a list instead with memory maps (to reduce disk usage).
        n_bands = len(images)
        self.flux_data = open_memmap(
            os.path.join(self.output_folder, "source_data.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(self.mmap_n_sources, n_bands, self.mmap_n_timesteps),
        )
        # Insert peak flux at time-step 0.
        self.flux_data[detected_sources["id"], :, 0] = detected_sources["peak_flux"]

    def _match_sources(self, t: int, prev: table.Table, detected_sources: table.Table) -> table.Table:  # type: ignore
        # Match detected sources to previous ones.
        # detected_sources.sort(keys="detection_flux", reverse=True)  # type: ignore
        # print(detected_sources[:10])
        # idx, d2d, _ = detected_sources["coordinate"].match_to_catalog_sky(  # type: ignore
        # prev["coordinate"]
        # )  # type: ignore
        # print(idx[:10])
        # exit()

        if len(detected_sources) == 0:
            logger.warning("Got empty source list.")
            # Return an empty table.
            return prev.copy()[:0]


        idx, d2d, _ = detected_sources["coordinate"].match_to_catalog_sky(  # type: ignore
            prev["coordinate"]
        )  # type: ignore

        s: np.ndarray = d2d.argsort()  # type: iignore
        idx: np.ndarray = idx[s]
        d2d: np.ndarray = d2d[s]
        detected_sources: table.Table = detected_sources[s]  # type: ignore

        sep_crit: np.ndarray = d2d.deg > self.separation_crit  # type: ignore
        new_sources: table.Table = detected_sources[sep_crit]  # type: ignore
        new_sources = filter_duplicates(new_sources)
        new_sources["id"] = range(self.ns_id, self.ns_id + len(new_sources))
        self.ns_id += len(new_sources)

        if len(new_sources) > 0:
            new_sources["new_source"] = True

        # Contains the index of previous time-step's best matching source for all detections in current time-steps
        # exluding ones that are more than sep_crit degrees separated.
        all_matched_source_idx = idx[~sep_crit]
        # Previous time-step's indices, excluding duplicates.
        matched_source_ids, unique_matched_source_idx = np.unique(
            all_matched_source_idx, return_index=True
        )
        matched_sources: table.Table = detected_sources[~sep_crit][unique_matched_source_idx]  # type: ignore
        matched_sources["id"] = prev[matched_source_ids]["id"].copy()  # type: ignore
        if len(matched_sources) > 0:
            matched_sources["new_source"] = False

        if len(new_sources) > 0:
            curr: table.Table = table.vstack([matched_sources, new_sources])  # type: ignore
        else:
            curr: table.Table = matched_sources  # type: ignore

        return curr


    def _update_metadata(self, t: int, new_sources: table.Table, matched_sources: table.table):
        # Update metadata.
        self.meta.loc[matched_sources["id"], "last_measured"] = t  # type: ignore
        # New sources metadata
        start_t = np.full([len(new_sources)], t - len(self.image_cache), dtype=int)  # type: ignore
        last_measured = np.full([len(new_sources)], t, dtype=int)
        meta = np.stack([start_t, last_measured], axis=-1)
        columns = ("start_t", "last_measured")
        self.meta = pd.concat(  # type: ignore
            [self.meta, pd.DataFrame(meta, columns=columns, index=new_sources["id"])]  # type: ignore
        )  # type: ignore

        # print(detected_sources[:10])
        # print(detected_sources[-10:])
        # exit()

        # idx: np.ndarray
        # d2d: np.ndarray
        # # Sort on distance.
        # s = d2d.argsort()
        # idx = idx[s]
        # print(idx)
        # print(len(idx))
        # exit()
        # d2d = d2d[s]
        # detected_sources: table.Table = detected_sources[s]  # type: ignore

        # # Association prioritised on distance (only 1 to 1).
        # _, s = np.unique(idx, return_index=True)
        # # Treat duplicates (not associated) as new sources.
        # first_mask = np.zeros(len(idx), dtype=bool)
        # first_mask[s] = True

        # # Matches are the closest sources within the criterion.
        # match_mask: np.ndarray = (d2d.deg < SEPARATION_CRIT) & first_mask  # type: ignore

        # # ID assignment vector.
        # id_assignment = np.empty(len(detected_sources))
        # id_assignment[:] = np.nan

        # # Add matched source ids.
        # matched_source_ids: np.ndarray = prev[idx[match_mask]]["id"].copy()  # type: ignore
        # id_assignment[match_mask] = matched_source_ids

        # # New source ids.
        # num_new_sources = len(match_mask[~match_mask])
        # new_idx = range(self.ns_id, num_new_sources + self.ns_id)
        # self.ns_id += num
        # id_assignment[~match_mask] = new_idx

        # return id_assignment  # type: ignore

    def _monitor_sources(self, t: int, prev: table.Table, curr: table.Table) -> table.Table:
        to_monitor: table.Table = prev[  # type: ignore
            (prev["last_detected"] > t - self.monitor_length)  # type: ignore
            & ~np.isin(prev["id"], curr["id"])  # type: ignore
        ].copy()  # type: ignore

        if len(to_monitor) > 0:
            to_monitor["is_monitored"] = True
            to_monitor["new_source"] = False
            if len(curr) > 0:
                curr = table.vstack([curr, to_monitor])  # type: ignore
            else:
                curr = to_monitor

        return curr

    def _backward_fill(self, t: int, new_sources: table.Table):

        new_sources["is_backward_fill"] = True

        for i in range(1, len(self.image_cache) + 1):  # type: ignore
            measurements_t = take_measurements(
                new_sources, self.image_cache[-i], self.box_size  # type: ignore
            )

            self.timesteps[-i] = table.vstack([self.timesteps[-i], measurements_t])  # type: ignore

            self.flux_data[measurements_t["id"], :, t - i] = measurements_t["peak_flux"]

            assert len(table.unique(self.timesteps[-i], keys="id")) == len(  # type: ignore
                self.timesteps[-i]
            )

    def add_timestep(
        self, t: int, detected_sources: table.Table, images: tensor_or_ndarray
    ) -> None:

        # detected_sources = filter_duplicates(detected_sources)


        if len(detected_sources) > 0:
            detected_sources = filter_nan(detected_sources)
            detected_sources["last_detected"] = t
            detected_sources["is_monitored"] = False
            detected_sources["is_backward_fill"] = False
        # Run initialization on first time-step.
        if self.ns_id == None:
            print("Initializing catalog.")
            return self._initialize(detected_sources, images)

        # Otherwise, main loop.
        # Load previous timestep's source table.
        prev: table.Table = self.timesteps[-1]

        # Get current time-step's source-table.
        curr = self._match_sources(t, prev, detected_sources)  # type: ignore


        # Add sources to monitor.
        curr = self._monitor_sources(t, prev, curr)

        # Take measurements
        curr = take_measurements(curr, images, self.box_size)

        # Add measured fluxes to the data array at the correct ids and timestep.
        self.flux_data[curr["id"], :, t] = curr["peak_flux"]

        new_source_mask: np.ndarray = curr['new_source']  # type: ignore
        matched_sources: table.Table = curr[~new_source_mask]  # type: ignore
        new_sources: table.Table = curr[new_source_mask]  # type: ignore
        self._update_metadata(t, new_sources, matched_sources)  # type: ignore

        # Backward fill new sources.
        new_sources: table.Table
        if len(new_sources) > 0:
            self._backward_fill(t, new_sources)

        # Sanity check.
        assert len(table.unique(curr, keys="id")) == len(curr)  # type: ignore

        self.timesteps.append(curr)

        self.image_cache: List[tensor_or_ndarray]
        self.image_cache.append(images.cpu())
        self.image_cache = self.image_cache[-self.monitor_length:]

    def filter_sources_for_analysis(self, t: int, length: int):
        
        runcat_t = self.timesteps[t]
        lengths = (t - self.meta.iloc[runcat_t['id']]['start_t']).values
        
        runcat_t = runcat_t[lengths >= length]
        
        flux_arrays = self.flux_data[runcat_t['id'], :, t - length: t]
        
        print(f"Neural network input shape: {flux_arrays.shape}")
        return runcat_t, flux_arrays
        

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return self.add_timestep(*args, **kwargs)

    def __getitem__(self, index: int):
        return self.timesteps[index]