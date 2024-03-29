import logging
import os
import pickle
import shutil
import sys
import time
import warnings
from collections import defaultdict
from types import FunctionType, MethodType
from typing import Any, Callable, List, Tuple

import astropy.io.fits  # type: ignore
import numpy as np
import pandas as pd
import torch
import yaml
from astropy.utils.exceptions import AstropyWarning  # type: ignore
from astropy.wcs import WCS  # type: ignore
from tqdm import trange  # type: ignore

from lpf._nn import TimeFrequencyCNN
from lpf.bolts.vis import catalog_video, plot_skymap
from lpf.quality_control import QualityControl
from lpf.running_catalog import RunningCatalog
from lpf.sigma_clip import ConvSigmaClipper
from lpf.source_finder import SourceFinderMaxFilter
from lpf.surveys import Survey
from lpf.bolts.math import create_circular_mask

logger = logging.getLogger(__name__)

warnings.simplefilter("ignore", category=AstropyWarning)


class LivePulseFinder:
    def __init__(self, config):
        self.config = config

        self.survey = Survey(
            config["fits_directory"],  # type: ignore
            config["timestamp_start_stop"],  # type: ignore
            config["subband_start_stop"],  # type: ignore
            config["delta_t"],
        )

        self.n_timesteps = (
            len(self.survey) if config["n_timesteps"] == -1 else config["n_timesteps"]
        )

        if torch.cuda.is_available():  # type: ignore
            logger.info("Running on GPU.")
            self.cuda: bool = True
        else:
            logger.info("Running on CPU.")
            self.cuda: bool = False

        self.qc = QualityControl()

        self.array_length: int = config["array_length"]

        self.image_size = config["image_size"]

        self.clipper = ConvSigmaClipper(
            self.image_size,  # type: ignore
            config["kappa"],  # type: ignore
            config["center_sigma"],  # type: ignore
            config["scale_sigma"],  # type: ignore
            config["detection_radius"],  # type: ignore
            config["sigma_clipping_maxiter"],  # type: ignore
            "cuda" if self.cuda else "cpu",
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
        cache_size = (
            monitor_length if config["cache_size"] == -1 else config["cache_size"]
        )
        self.runningcatalog = RunningCatalog(
            config["output_folder"],  # type: ignore
            config["box_size"],  # type: ignore
            config["mmap_n_sources"],  # type: ignore
            self.n_timesteps,  # type: ignore
            monitor_length=monitor_length,  # type: ignore
            cache_size=cache_size,
            separation_crit=config["separation_crit"],  # type: ignore
        )

        self.nn = TimeFrequencyCNN([len(config["frequencies"]), self.array_length])  # type: ignore
        if self.cuda:
            self.nn.set_device("cuda")
        else:
            self.nn.set_device("cpu")
        self.nn.load(config["nn_checkpoint"])  # type: ignore
        self.nn.eval()

        self.timings: defaultdict[str, List[float]] = defaultdict(list)

        self.transient_parameters = os.path.join(
            config["output_folder"], "parameters.csv"  # type: ignore
        )

    def _load_data(self, t: int) -> Tuple[torch.Tensor, WCS]:

        images: List[np.ndarray] = []  # type: ignore
        header = None

        for f in self.survey[t]["file"]:  # type: ignore
            if f is not None:
                try:
                    image, header = astropy.io.fits.getdata(f, header=True)  # type: ignore
                    image = image.squeeze().astype(np.float32)
                except (ValueError, OSError) as e:
                    print(f"Got error at time-step {t}.")
                    print(e)
                    image = np.zeros(
                        [self.image_size, self.image_size], dtype=np.float32
                    )
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
            logger.warning("No images were found in time-step %s", t)
            wcs = None

        return images, wcs

    def _infer_parameters(self, x_batch: np.ndarray):
        x_batch_tensor = torch.from_numpy(x_batch).to(self.nn.device)[:, None]  # type: ignore
        py_x = self.nn((x_batch_tensor, None))  # 'None' is placeholder for the targets.
        means = py_x.mean
        covariance_matrix = py_x.covariance_matrix
        m = means.shape[-1]
        stds = torch.sqrt(covariance_matrix[:, range(m), range(m)])
        means = means.permute(-1, -2).to("cpu").numpy()
        stds = stds.permute(-1, -2).to("cpu").numpy()
        return means, stds

    def _write_to_csv(
        self, timestep: int, runcat_t: np.ndarray, means: np.ndarray, stds: np.ndarray
    ) -> None:
        dm, peak_flux, width, index, t0 = means
        (dm_std, peak_flux_std, width_std, index_std, t0_std) = stds

        data = {
            "timestep": timestep,
            "source_id": runcat_t["id"],
            "ra": runcat_t["coordinate"].ra.deg,
            "dec": runcat_t["coordinate"].dec.deg,
            "x_peak": runcat_t["x_peak"],
            "y_peak": runcat_t["y_peak"],
            "last_detected": runcat_t["last_detected"],
            "is_monitored": runcat_t["is_monitored"],
            "is_backward_fill": runcat_t["is_backward_fill"],
            "is_new_source": runcat_t["new_source"],
            "channel": runcat_t["channel"],
            "peak_flux": runcat_t["peak_flux"].max(),
            "dm": dm,
            "dm_std": dm_std,
            "peak_flux": peak_flux,
            "peak_flux_std": peak_flux_std,
            "width": width,
            "width_std": width_std,
            "spectral_index": index,
            "spectral_index_std": index_std,
            "t0": t0,
            "t0_std": t0_std,
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
        self.mask: np.ndarray = create_circular_mask(self.config["image_size"], self.config["image_size"], 
                                                     radius=self.config["detection_radius"])
        t: int
        with torch.no_grad():
            for t in trange(self.n_timesteps):  # type: ignore
                images, wcs = self._load_data(t)

                value_limit =  torch.mean(images[:,self.mask]) + 100*torch.std(images[:,self.mask])                

                for i in range(len(images)):
                    if torch.max(np.abs(images[i,self.mask])) > value_limit:
                        images[i,:,:] *= 0.

                images = torch.clip(images, -value_limit, value_limit)

                s = time.time()
                # Quality control
                if self.config["use_quality_control"]:
                    images: torch.Tensor = self.call(s, self.qc, images)

                # Sigma clipping.
                peaks, residual, center, scale = self.call(s, self.clipper, images)

                # Source localization
                detected_sources = self.call(s, self.sourcefinder, images, peaks, wcs=wcs)  # type: ignore

                # Running catalog
                self.call(s, self.runningcatalog, t, detected_sources, images)

                # Filter sources for analysis
                runcat_t, x_batch = self.call(s, self.runningcatalog.filter_sources_for_analysis, t, self.array_length)  # type: ignore

                if len(x_batch) > 0 and x_batch.shape[-1] == self.array_length:
                    means, stds = self.call(s, self._infer_parameters, x_batch)
                    self.call(s, self._write_to_csv, t, runcat_t, means, stds)  # type: ignore

                if t == min(32, self.n_timesteps - 1):
                    logger.warning(
                        "Making catalog video and example background and RMS estimations. One moment..."
                    )
                    anim = catalog_video(
                        self.survey,
                        self.runningcatalog,
                        range(t),
                        n_std=3,
                    )
                    if self.config['catalog_video']:
                        anim.save(os.path.join(self.config["output_folder"], "catalogue_video.mp4"))  # type: ignore

                    if self.config['background_rms_maps']:

                        plot_skymap(
                            center.mean(0).cpu(),
                            fname=os.path.join(self.config["output_folder"], "center.pdf"),
                            n_std=3,
                        )
                        plot_skymap(
                            scale.mean(0).cpu(),
                            fname=os.path.join(self.config["output_folder"], "scale.pdf"),
                            n_std=3,
                        )

        # timings: Dict[str, Any] = {k: np.mean(self.timings[k]) for k in self.timings}  # type: ignore
        # print(self.timings)

        # plt.imsave(os.path.join(self.config["output_folder"], "center.pdf"), center.mean(0).cpu(), vmin=-5, vmax=5)  # type: ignore
        # plt.imsave(os.path.join(self.config["output_folder"], "scale.pdf"), scale.mean(0).cpu(), vmin=-5, vmax=5)  # type: ignore

        with open(os.path.join(self.config["output_folder"], "timings.pkl"), "wb") as f:  # type: ignore
            pickle.dump(self.timings, f)

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
