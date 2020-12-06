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
from lpf.surveys import Survey
import time

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

        self.timings = {
            'quality_control': []
        }

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

    def run(self) -> None:

        t: int
        for t in trange(len(self.survey)):  # type: ignore
            images, wcs = self._load_data(t)
            # Quality control
            s = time.time()
            images = self.qc(images)
            if self.cuda:
                torch.cuda.synchronize()  # type: ignore
            e = time.time()
            self.timings['quality_control'].append(e - s)


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
