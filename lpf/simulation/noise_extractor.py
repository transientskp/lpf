from lpf.surveys import Survey
from numpy.lib.format import open_memmap
import os
import numpy as np
from tqdm import tqdm  # type: ignore
from typing import List
import astropy.io.fits  # type: ignore


class NoiseExtractor:
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.n_arrays: int = config["n_arrays"]
        self.num_patches_per_image: int = config["num_patches_per_image"]
        self.nfreq: int = len(config["frequencies"])
        self.array_length: int = config["array_length"]
        self.radius: int = config["detection_radius"]
        self.image_size: int = config["image_size"]
        self.box_size: int = config["box_size"]

        self.survey = Survey(
            config["fits_directory"],  # type: ignore
            config["timestamp_start_stop"],  # type: ignore
            config["subband_start_stop"],  # type: ignore
            config["delta_t"],  # type: ignore
        )

        os.makedirs(config["noise_output_folder"])  # type: ignore
        self.mmap: np.ndarray = open_memmap(
            os.path.join(config["noise_output_folder"], "noise.npy"),  # type: ignore
            dtype=np.float32,
            mode="w+",
            shape=(self.n_arrays, self.nfreq, self.array_length),
        )

    def integrate_random_sequence(self, to_integrate: int):
        tf_array = np.zeros([to_integrate, self.nfreq, self.array_length])
        # Get random locations.
        image_center = self.image_size // 2
        a = np.random.rand(to_integrate) * 2 * np.pi
        r = np.sqrt(np.random.rand(to_integrate) * self.radius ** 2)
        x_l = (r * np.cos(a) + image_center).astype(int)
        y_l = (r * np.sin(a) + image_center).astype(int)

        t = np.random.randint(0, len(self.survey) - self.array_length)
        for i in tqdm(range(self.array_length)):  # type: ignore
            i: int
            survey_timestep = self.survey[t + i]
            files: List[np.ndarray] = [astropy.io.fits.getdata(f) for f in survey_timestep["file"]]  # type: ignore
            files: np.ndarray = np.stack(files).squeeze()  # type: ignore
            patches = np.stack(
                [
                    files[
                        :,
                        x_l[j] - self.box_size // 2 : x_l[j] + self.box_size // 2,
                        y_l[j] - self.box_size // 2 : y_l[j] + self.box_size // 2,
                    ]
                    for j in range(to_integrate)
                ]
            )

            integrated: np.ndarray = patches.sum(axis=(-1, -2))  # type: ignore
            tf_array[:, :, i] = integrated

        return tf_array

    def run(self):
        counter = 0
        pbar = tqdm(total=self.n_arrays)

        while counter < self.n_arrays:
            to_integrate: int = min(self.n_arrays - counter, self.num_patches_per_image)

            tf_array = self.integrate_random_sequence(to_integrate)
            assert not np.isnan(tf_array).any()
            self.mmap[counter : counter + to_integrate] = tf_array
            counter += to_integrate
            pbar.update(to_integrate)  # type: ignore

            if counter == self.n_arrays:
                break
