from torch._C import Value
from lpf.sigma_clip.conv_sigma_clipper import ConvSigmaClipper
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict
from types import FunctionType, MethodType
from typing import Any, Callable, List, Tuple
import matplotlib.pyplot as plt
import shutil

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
# from lpf.sigma_clip import ConvSigmaClipper
from lpf._nn import TimeFrequencyCNN
# from lpf.bolts.vis import plot_skymap, catalog_video
# from lpf.bolts.vis import catalog_video
from lpf.quality_control import QualityControl
from lpf.running_catalog import RunningCatalog
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
            config["dt"]
        )

        self.n_timesteps = len(self.survey) if config['n_timesteps'] == -1 else config['n_timesteps']

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

        # self.clipper = LocalSigmaClipper(
        #     config["image_shape"],  # type: ignore
        #     config["kappa"],  # type: ignore
        #     config["detection_radius"],  # type: ignore
        #     config["sigmaclip_kernel_size"],  # type: ignore
        #     config["sigmaclip_stride"],  # type: ignore
        # )
        self.image_size = config["image_size"]

        self.clipper = ConvSigmaClipper(
            self.image_size,              # type: ignore
            config["kappa"],                   # type: ignore
            config["center_sigma"],            # type: ignore
            config["scale_sigma"],             # type: ignore
            config["detection_radius"],                  # type: ignore
            config["sigma_clipping_maxiter"],  # type: ignore
            "cuda" if self.cuda else "cpu"
        )

        self.sourcefinder = SourceFinderMaxFilter(self.image_size)  # type: ignore

        if os.path.exists(config["output_folder"]):
            remove = None
            while remove not in ["y", "n"]:
                remove = input("Run folder exists, delete it? (y/n)")
            if remove == "y":
                shutil.rmtree(config["output_folder"])
            else:
                exit()

        os.makedirs(config["output_folder"])  # type: ignore
        monitor_length = self.array_length // 2  # type: ignore
        self.runningcatalog = RunningCatalog(
            config["output_folder"],  # type: ignore
            config["box_size"],  # type: ignore
            config["mmap_n_sources"],  # type: ignore
            self.n_timesteps,  # type: ignore
            monitor_length=monitor_length,  # type: ignore
            separation_crit=config["separation_crit"],  # type: ignore
        )

        self.nn = TimeFrequencyCNN(config["nn"])  # type: ignore
        if self.cuda:
            self.nn.set_device("cuda")
        else:
            self.nn.set_device("cpu")
        self.nn.load(config["nn"]["checkpoint"])  # type: ignore
        self.nn.eval()

        self.timings: defaultdict[str, List[float]] = defaultdict(list)

        self.transient_parameters = os.path.join(
            config["output_folder"], "parameters.csv"  # type: ignore
        )

    def _load_data(self, t: int) -> Tuple[torch.Tensor, WCS]:

        images: List[np.ndarray] = []  # type: ignore
        # headers: List[astropy.io.fits.Header] = []
        header = None

        for f in self.survey[t]["file"]:  # type: ignore
            if f is not None:
                image, header = astropy.io.fits.getdata(f, header=True)  # type: ignore
                image = image.squeeze().astype(np.float32)
            else:
                image = np.zeros([self.image_size, self.image_size], dtype=np.float32)

            images.append(image)  # type: ignore

        images: np.ndarray = np.stack(images)  # type: ignore
        images: torch.Tensor = torch.from_numpy(images)  # type: ignore
        if self.cuda:
            images = images.cuda()

        images[torch.isnan(images)] = 0

        if header is not None:
            wcs = WCS(header)
        else:
            raise ValueError(f"No images were found in time-step {t}.")

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
        with torch.no_grad():
            for t in trange(self.n_timesteps):  # type: ignore
                images, wcs = self._load_data(t)
                s = time.time()
                # Quality control
                if self.config['use_quality_control']:
                    images: torch.Tensor = self.call(s, self.qc, images)

                # Statistics estimation.
#                 intensity_map, variability_map, intensity_subtracted = self.call(
#                     s, self.statistics, images
#                 )

                # Sigma clipping.
                peaks, center, scale = self.call(s, self.clipper, images)

                # Source localization
                detected_sources = self.call(s, self.sourcefinder, images, peaks, wcs=wcs)  # type: ignore

                # Running catalog
                self.call(s, self.runningcatalog, t, detected_sources, images)

                # Filter sources for analysis
                runcat_t, x_batch = self.call(s, self.runningcatalog.filter_sources_for_analysis, t, self.array_length)  # type: ignore

                if len(x_batch) > 0 and x_batch.shape[-1] == self.array_length:
                    means, stds = self.call(s, self._infer_parameters, x_batch)
                    self.call(s, self._write_to_csv, t, runcat_t, means, stds)  # type: ignore

        # timings: Dict[str, Any] = {k: np.mean(self.timings[k]) for k in self.timings}  # type: ignore
        # print(self.timings)
 
        # anim = catalog_video(self.survey, self.runningcatalog, range(length), n_std=3)
        # anim.save(os.path.join(self.config["output_folder"], "catalogue_video.mp4"))  # type: ignore

        # plt.imsave(os.path.join(self.config["output_folder"], "center.pdf"), center.mean(0).cpu(), vmin=-5, vmax=5)  # type: ignore
        # plt.imsave(os.path.join(self.config["output_folder"], "scale.pdf"), scale.mean(0).cpu(), vmin=-5, vmax=5)  # type: ignore

        # with open(os.path.join(self.config['output_folder'], "timings.pkl"), 'wb') as f:  # type: ignore
        #     pickle.dump(self.timings, f)  
            
        # with open(os.path.join(self.config['output_folder'], "runningcatalog.pkl"), 'wb') as f:  # type: ignore
        #     pickle.dump(self.runningcatalog, f)  


        


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
