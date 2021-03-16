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

    def integrate_random_sequence(self, counter, to_integrate: int):
        # Get random locations.
        image_center = self.image_size // 2
        a = np.random.rand(to_integrate) * 2 * np.pi
        r = np.sqrt(np.random.rand(to_integrate) * self.radius ** 2)
        x_l = (r * np.cos(a) + image_center).astype(int)
        y_l = (r * np.sin(a) + image_center).astype(int)

        t = np.random.randint(0, len(self.survey) - self.array_length)
        for i in tqdm(range(self.array_length)):  # type: ignore
            survey_timestep = self.survey[t + i]
            for l, f in enumerate(survey_timestep["file"]):
                try:
                    image = astropy.io.fits.getdata(f, memmap_mode=True)  # type: ignore
                    image = image.squeeze().astype(np.float32)  # type: ignore
                except (ValueError, OSError) as e:
                    print(f"Got loading error error at time-step {t}.")
                    print(e)
                    image = np.zeros(
                        [self.image_size, self.image_size], dtype=np.float32
                    )                    
                
                for j in range(to_integrate):
                    patch = image[
                            x_l[j] - self.box_size // 2 : x_l[j] + self.box_size // 2,
                            y_l[j] - self.box_size // 2 : y_l[j] + self.box_size // 2,
                        ]
                    
                    integrated = patch.sum()
                    self.mmap[counter + j, l, i] = integrated


    def run(self):
        counter = 0
        pbar = tqdm(total=self.n_arrays)

        while counter < self.n_arrays:
            to_integrate: int = min(self.n_arrays - counter, self.num_patches_per_image)

            self.integrate_random_sequence(counter, to_integrate)

            assert not np.isnan(self.mmap).any()
            counter += to_integrate
            pbar.update(to_integrate)  # type: ignore

            if counter == self.n_arrays:
                break
