from typing import List, Tuple
from numpy.core.numeric import full
from numpy.lib.format import open_memmap
import os
import numpy as np
from numpy.lib.function_base import disp
from numpy.ma.core import _DomainCheckInterval
from tqdm import trange  # type: ignore
from lpf.simulation.simutils import disp_delay
import logging
from tqdm import trange

logger = logging.getLogger(__name__)

DT = 1e-2
DF = 1e-2


def any_intersection_sorted(intervals, reverse=False):
    for i in range(len(intervals) - 1):
        if reverse:
            if intervals[i][0] < intervals[i + 1][0]:
                return True
        else:
            if intervals[i][0] > intervals[i + 1][0]:
                return True
    return False


class Telescope:
    def __init__(
        self,
        frequencies: List[float],
        delta_f: float,
        delta_t: float,
        array_length: int,
    ):

        self.frequencies = np.sort(frequencies)
        self.array_length = array_length
        self.delta_t: float = delta_t
        self.delta_f = delta_f

        self.passbands = [
            (f - self.delta_f / 2, f + self.delta_f / 2) for f in self.frequencies
        ]

        assert not any_intersection_sorted(
            self.passbands, reverse=False
        ), "Overlapping passbands not supported."

    def check_config(self, dm_range):

        # Lowest
        f0 = self.frequencies[0]
        # Highest
        f1 = self.frequencies[-1]
        max_dm = dm_range[-1]
        delay = disp_delay(f0, max_dm) - disp_delay(f1, max_dm)

        logger.warning(
            "The delay between the highest and lowest frequency can be %.3f. The specified array length is %s with a ∆t of %s, covering %s seconds.",
            delay,
            self.array_length,
            self.delta_t,
            self.delta_t * self.array_length,
        )
        if delay >= 2 * self.array_length * self.delta_t:
            logger.warning(
                "The maximum delay is much larger than the specified array length. Consider enlargening it."
            )


class Event(object):
    def __init__(
        self,
        f_ref: float,
        dm: float,
        fluence: float,
        width: float,
        spec_ind: float,
        disp_ind: float = 2,
    ):
        self.f_ref = f_ref
        self.dm = dm
        self.fluence = fluence
        self.width = width
        self.spec_ind = spec_ind
        self.disp_ind = disp_ind

    def gaussian_profile(self, ntime: int, width: float):
        x: np.ndarray = np.linspace(-ntime // 2, ntime // 2, ntime)  # type: ignore
        g = np.exp(-((x / width) ** 2)) + 1e-9
        return g

    def arrival_time(self, f: float) -> float:
        delay: float = disp_delay(f, self.dm, self.disp_ind)  # seconds
        # Subtract the f_ref time to center the event (reference freq arrives at t)
        t = delay - disp_delay(self.f_ref, self.dm, self.disp_ind)
        return t

    def simulate(self, telescope: Telescope):
        t_osr = int(telescope.delta_t / DT)
        length_full_burst = telescope.array_length * t_osr
        pulse = self.gaussian_profile(length_full_burst, self.width * t_osr)

        t0 = int(np.random.rand() * length_full_burst)
        spectrum = []

        for band in reversed(telescope.passbands):
            frequencies = np.arange(*reversed(band), -DF)
            delays = self.arrival_time(frequencies) * t_osr
            delays = delays.astype(int)
            burst = []
            for i in range(len(delays)):
                delay = delays[i]
                frequency = frequencies[i]
                shift = t0 - length_full_burst // 2 + delay
                val = pulse.copy() * self.fluence / self.width
                val = val * (frequency / self.f_ref) ** self.spec_ind
                val = np.roll(val, shift)
                if shift < 0:
                    val[shift:] *= 0
                else:
                    val[:shift] *= 0

                burst.append(val)
            burst = np.stack(burst)
            # First average frequencies, then average time.
            integrated_burst = burst.mean(0).reshape(-1, t_osr).mean(1)
            spectrum.append(integrated_burst)

        spectrum = np.stack(spectrum)
        return spectrum


class EventSimulator:
    def __init__(
        self,
        dm_range: Tuple[float, float],
        fluence_range: Tuple[float, float],
        width_range: Tuple[float, float],
        spec_ind_range: Tuple[float, float],
    ) -> None:
        super().__init__()

        self.dm_range = dm_range
        self.fluence_range = fluence_range
        self.width_range = width_range
        self.spec_ind_range = spec_ind_range

    def draw_event_parameters(self) -> Tuple[float, float, float, float]:
        dm = np.random.uniform(*self.dm_range)
        fluence = np.random.uniform(*self.fluence_range)
        width = np.random.uniform(*self.width_range)
        spec_ind = np.random.uniform(*self.spec_ind_range)
        return dm, fluence, width, spec_ind

    def __call__(self, telescope: Telescope) -> Tuple[np.ndarray, np.ndarray]:
        dm, fluence, width, spec_ind = self.draw_event_parameters()
        f_ref = np.median(telescope.frequencies)
        event = Event(f_ref, dm, fluence, width, spec_ind)
        data = event.simulate(telescope)
        return data, np.array([dm, fluence, width, spec_ind])


class TransientSimulator:
    def __init__(self, config):

        self.telescope = Telescope(
            # type: ignore
            config["frequencies"],
            config["delta_f"],
            config["delta_t"],
            config["array_length"],
        )

        self.telescope.check_config(config["dm_range"])

        self.event_simulator = EventSimulator(
            config["dm_range"],  # type: ignore
            config["fluence_range"],  # type: ignore
            config["width_range"],  # type: ignore
            config["spec_ind_range"],  # type: ignore
        )

        self.output_folder: str = config["simulation_output_folder"]
        os.makedirs(self.output_folder)

        self.data_mmap: np.ndarray = open_memmap(
            os.path.join(self.output_folder, "data.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(
                config["nevents"],
                len(self.telescope.frequencies),
                config["array_length"],
            ),  # type: ignore
        )
        self.param_mmap: np.ndarray = open_memmap(
            os.path.join(self.output_folder, "parameters.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(config["nevents"], config["nparams"]),
        )

        self.config = config

    def run(self):
        i: int
        for i in trange(self.config["nevents"]):  # type: ignore
            data, parameters = self.event_simulator(self.telescope)
            self.data_mmap[i] = data
            self.param_mmap[i] = parameters

        del self.data_mmap

        self.data_mmap: np.ndarray = open_memmap(
            os.path.join(self.output_folder, "data.npy"),
            mode="r",
            # dtype=np.float32,
            # shape=(config["nevents"], len(self.telescope.frequencies), config["array_length"]),  # type: ignore
        )
