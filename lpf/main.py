import sys
import yaml
from lpf.surveys import Survey
from tqdm import trange  # type: ignore
import torch
import astropy.io.fits  # type: ignore
import numpy as np
from typing import List, Union, Tuple
from astropy.wcs import WCS  # type: ignore
from lpf.quality_control import QualityControl
from astropy.utils.exceptions import AstropyWarning  # type: ignore
import warnings

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
            images = self.qc(images)
            
            break


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