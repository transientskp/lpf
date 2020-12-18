import os
import pickle
import sys
import time
import warnings
from collections import defaultdict
from types import FunctionType, MethodType
from typing import Any, Callable, Dict, List, Tuple, Union

import astropy.io.fits  # type: ignore
import numpy as np
import pandas as pd
import torch
import yaml
from astropy.utils.exceptions import AstropyWarning  # type: ignore
from astropy.wcs import WCS  # type: ignore
from tqdm import trange  # type: ignore

# from lpf.statistics_estimation import StatisticsEstimator
# from lpf.sigma_clip import SigmaClipper
from lpf._nn import TimeFrequencyCNN
# from lpf.bolts.vis import plot_skymap, catalog_video
from lpf.bolts.vis import catalog_video
from lpf.quality_control import QualityControl
from lpf.running_catalog import RunningCatalog
from lpf.sigma_clip import LocalSigmaClipper
from lpf.source_finder import SourceFinderMaxFilter
from lpf.surveys import Survey

warnings.simplefilter("ignore", category=AstropyWarning)


class LivePulseFinder:
    def __init__(self, config):
        self.config = config

        self.survey = Survey(
            config["fits_directory"],  # type: ignore
            config["timestamp_start_stop"],  # type: ignore
            config["subband_start_stop"],  # type: ignore
        )

        if torch.cuda.is_available():  # type: ignore
            print("Running on GPU.")
            self.cuda: bool = True
        else:
            print("Running on CPU.")
            self.cuda: bool = False

        self.qc = QualityControl()
        # self.statistics = StatisticsEstimator(
        #     config["intensity_map_sigma"],  # type: ignore
        #     config["variability_map_sigma"],  # type: ignore
        #     stride=config["filter_stride"],  # type: ignore
        #     truncate=config["filter_truncate"],  # type: ignore
        # )

        # self.clipper = SigmaClipper(
        #     config["image_shape"], config["kappa"], config["detection_radius"]  # type: ignore
        # )
        self.array_length: int = config["array_length"]

        self.clipper = LocalSigmaClipper(
            config["image_shape"],  # type: ignore
            config["kappa"],  # type: ignore
            config["detection_radius"],  # type: ignore
            config["sigmaclip_kernel_size"],  # type: ignore
            config["sigmaclip_stride"],  # type: ignore
        )

        image_size: int = config["image_shape"][0]
        self.sourcefinder = SourceFinderMaxFilter(image_size)
        os.makedirs(config["output_folder"])  # type: ignore
        monitor_length = self.array_length // 2  # type: ignore
        self.runningcatalog = RunningCatalog(
            config["output_folder"],  # type: ignore
            config["box_size"],  # type: ignore
            config["mmap_n_sources"],  # type: ignore
            config["mmap_n_timesteps"],  # type: ignore
            monitor_length=monitor_length,  # type: ignore
            separation_crit=config["separation_crit"],  # type: ignore
        )

        self.nn = TimeFrequencyCNN(config["nn"])  # type: ignore
        self.nn.load(config["nn"]["checkpoint"])  # type: ignore
        self.nn.eval()

        if self.cuda:
            self.nn.set_device("cuda")
        else:
            self.nn.set_device("cpu")

        self.timings: defaultdict[str, List[float]] = defaultdict(list)

        self.transient_parameters = os.path.join(
            config["output_folder"], "parameters.csv"  # type: ignore
        )

    def _load_data(self, t: int) -> Tuple[Union[np.ndarray, torch.Tensor], WCS]:

        images: List[np.ndarray] = []  # type: ignore
        headers: List[astropy.io.fits.Header] = []

        for f in self.survey[t]["file"]:  # type: ignore
            image, header = astropy.io.fits.getdata(f, header=True)  # type: ignore
            images.append(image)  # type: ignore
            headers.append(header)  # type: ignore

        images: np.ndarray = np.stack(images).squeeze()  # type: ignore

        if self.cuda:
            images: torch.Tensor = torch.from_numpy(images.astype(np.float32)).to("cuda")  # type: ignore

        wcs = WCS(headers[0])

        return images, wcs

    def _infer_parameters(self, x_batch: np.ndarray):
        x_batch_tensor = torch.from_numpy(x_batch).to(self.nn.device)[:, None]  # type: ignore
        predictions = self.nn(
            (x_batch_tensor, None)
        )  # 'None' is placeholder for the targets.
        means, stds = predictions
        means = means.permute(-1, -2).to("cpu").numpy()
        stds = stds.permute(-1, -2).to("cpu").numpy()

        return means, stds

    def _write_to_csv(
        self, timestep: int, runcat_t: np.ndarray, means: np.ndarray, stds: np.ndarray
    ) -> None:
        dm, fluence, width, index = means
        (dm_std,) = stds
        
        data = {
            "timestep": timestep,
            "source_id": runcat_t['id'],
            "ra": runcat_t['coordinate'].ra.deg,
            "dec": runcat_t['coordinate'].dec.deg,
            "x_peak": runcat_t['x_peak'],
            'y_peak': runcat_t['y_peak'],
            'last_detected': runcat_t['last_detected'],
            'is_monitored': runcat_t['is_monitored'],
            'is_backward_fill': runcat_t['is_backward_fill'],
            'is_new_source': runcat_t['new_source'],
            'channel': runcat_t['channel'],
            "peak_flux": runcat_t['peak_flux'].max(),
            "dm": dm,
            "dm_std": dm_std,
            "fluence": fluence,
            "width": width,
            "spectral_index": index,
        }
        
        df = pd.DataFrame.from_dict(data, dtype=np.float32)  # type: ignore
        df.to_csv(
            self.transient_parameters,
            mode="a",
            header=not os.path.exists(self.transient_parameters),
        )

    def call(
        self,
        start_time: float,
        fn: Callable[[Any], Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:

        result = fn(*args, **kwargs)
        if self.cuda:
            torch.cuda.synchronize()  # type: ignore
        end_time = time.time()
        if isinstance(fn, (FunctionType, MethodType)):
            name: str = fn.__name__  # type: ignore
        else:
            name: str = fn.__class__.__name__

        self.timings[name].append(end_time - start_time)
        return result

    def run(self) -> None:

        t: int
        length = len(self.survey)
        with torch.no_grad():
            for t in trange(length):  # type: ignore
                images, wcs = self._load_data(t)
                s = time.time()
                # Quality control
                images: Union[torch.Tensor, np.ndarray] = self.call(s, self.qc, images)

                # Statistics estimation.
#                 intensity_map, variability_map, intensity_subtracted = self.call(
#                     s, self.statistics, images
#                 )

                # Sigma clipping.
                peaks, _ = self.call(s, self.clipper, images)

                detected_sources = self.call(s, self.sourcefinder, peaks, wcs=wcs)
                self.call(s, self.runningcatalog, t, detected_sources, images)

                runcat_t, x_batch = self.call(s, self.runningcatalog.filter_sources_for_analysis, t, self.array_length)  # type: ignore

                if x_batch.shape[0] > 0 and x_batch.shape[-1] == self.array_length:
                    means, stds = self.call(s, self._infer_parameters, x_batch)
                    self.call(s, self._write_to_csv, t, runcat_t, means, stds)  # type: ignore

                if t == length:
                    break

#         anim = catalog_video(self.survey, self.runningcatalog, range(length), n_std=0.5)
#         anim.save(os.path.join(self.config["output_folder"], "catalogue_video.mp4"))  # type: ignore

        # timings: Dict[str, Any] = {k: np.mean(self.timings[k]) for k in self.timings}  # type: ignore
        # print(timings)
        with open(os.path.join(self.config['output_folder'], "timings.pkl"), 'wb') as f:  # type: ignore
            pickle.dump(self.timings, f)  
            
        with open(os.path.join(self.config['output_folder'], "runningcatalog.pkl"), 'wb') as f:  # type: ignore
            pickle.dump(self.runningcatalog, f)  
        


def get_config():

    cmd = sys.argv
    if len(cmd) != 2:
        raise ValueError(
            f"Please provide path to config file. Example: lpf /path/to/config.yml"
        )
    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)  # type: ignore
    return config


def main():
    config = get_config()
    lpf = LivePulseFinder(config)
    lpf.run()


if __name__ == "__main__":
    main()
