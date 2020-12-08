import sys
import warnings
from typing import List, Tuple, Union

import astropy.io.fits  # type: ignore
import numpy as np
import torch
import yaml
from astropy.utils.exceptions import AstropyWarning  # type: ignore
from astropy.wcs import WCS  # type: ignore
from tqdm import trange  # type: ignore

from lpf.quality_control import QualityControl

# from lpf.statistics_estimation import StatisticsEstimator
# from lpf.sigma_clip import SigmaClipper
from lpf.sigma_clip import LocalSigmaClipper
from lpf.surveys import Survey
from lpf.source_finder import SourceFinderMaxFilter
from lpf.running_catalog import RunningCatalog
import time
from collections import defaultdict
from typing import List, Any, Callable, Union
from types import FunctionType

# from lpf.bolts.vis import plot_skymap, catalog_video
from lpf.bolts.vis import catalog_video
import os

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
        monitor_length = config["array_length"] // 2  # type: ignore
        self.runningcatalog = RunningCatalog(
            config["output_folder"],                  # type: ignore
            config["box_size"],                       # type: ignore
            config["mmap_n_sources"],                 # type: ignore
            config["mmap_n_timesteps"],               # type: ignore
            monitor_length=monitor_length,            # type: ignore
            separation_crit=config['separation_crit'] # type: ignore
        )

        self.timings: defaultdict[str, List[float]] = defaultdict(list)

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

    def call(
        self,
        start_time: float,
        fn: Callable[[Any], Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:

        result = fn(*args, **kwargs)
        # if self.cuda:
        # torch.cuda.synchronize()  # type: ignore
        end_time = time.time()
        if isinstance(fn, FunctionType):
            name: str = fn.__name__  # type: ignore
        else:
            name: str = fn.__class__.__name__

        self.timings[name].append(end_time - start_time)
        return result

    def run(self) -> None:

        t: int
        length = 96
        for t in trange(length):  # type: ignore
            images, wcs = self._load_data(t)
            s = time.time()
            # Quality control
            # images:  = self.run_qc(images)
            images: Union[torch.Tensor, np.ndarray] = self.call(s, self.qc, images)
            # Statistics estimation.
            # intensity_map, variability_map, intensity_subtracted = self.call(
            #     s, self.statistics, images
            # )
            # Sigma clipping.
            peaks, _ = self.call(s, self.clipper, images)

            detected_sources = self.call(s, self.sourcefinder, peaks, wcs=wcs)
            self.call(s, self.runningcatalog, t, detected_sources, images)

            self.runningcatalog.filter_sources_for_analysis(t, self.config['array_length'])  # type: ignore

            if t == length:
                break

        anim = catalog_video(self.survey, self.runningcatalog, range(length), n_std=1)
        anim.save(os.path.join(self.config["output_folder"], "catalogue_video.mp4"))  # type: ignore

        # source_idx = 0

        # locs = []

        # for i in range(length):
        #     catitem = self.runningcatalog[i]
        #     source = catitem[catitem["id"] == source_idx]
        #     print(source)

        # source_locations = self.runningcatalog.timesteps[0]

        print(self.timings)


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
