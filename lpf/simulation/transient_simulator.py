from typing import List, Tuple
from numpy.core.numeric import full
from numpy.lib.format import open_memmap
import os
import numpy as np
from numpy.lib.function_base import disp
from tqdm import trange  # type: ignore
from lpf.simulation.simutils import disp_delay
import logging
from tqdm import trange

logger = logging.getLogger(__name__)

DT = 2
DF = 0.1953125

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
        self, frequencies: List[float], delta_f: float, delta_t: float, array_length: int
    ):

        # n_bands = int((frequency_range[1] - frequency_range[0]) // df)
        # self.frequencies: np.ndarray = np.linspace(*frequency_range, n_bands + 1)  # type: ignore
        self.frequencies = np.sort(frequencies)[::-1]
        self.array_length = array_length
        self.delta_t: float = delta_t
        self.delta_f = delta_f

        print(self.frequencies)

        self.bandpasses = [(f - self.delta_f / 2, f + self.delta_f / 2) for f in self.frequencies]
        assert not any_intersection_sorted(self.bandpasses, reverse=True), "Overlapping bandpasses not supported."

        

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
            self.dt,
            self.dt * self.array_length,
        )
        if delay >= 2 * self.array_length * self.dt:
            logger.warning(
                "The maximum delay is much larger than the specified array length. Consider enlargening it."
            )


class Event(object):
    def __init__(
        self,
        t_ref: float,
        f_ref: float,
        dm: float,
        fluence: float,
        width: float,
        spec_ind: float,
        disp_ind: float = 2,
    ):
        self.t_ref = t_ref
        self.f_ref = f_ref
        self.dm = dm
        self.fluence = fluence
        self.width = width
        self.spec_ind = spec_ind
        self.disp_ind = disp_ind

    def gaussian_profile(self, ntime: int, width: float):
        x: np.ndarray = np.linspace(-ntime // 2,
                                    ntime // 2, ntime)  # type: ignore
        g = np.exp(-((x / width) ** 2)) + 1e-9
        return g

    def arrival_time(self, f: float) -> float:
        delay: float = disp_delay(f, self.dm, self.disp_ind)  # seconds
        # Subtract the f_ref time to center the event (reference freq arrives at t)
        t = delay - disp_delay(self.f_ref, self.dm, self.disp_ind)
        return self.t_ref + t

    def simulate(self, telescope: Telescope):
        data = np.zeros(
            shape=(len(telescope.frequencies), telescope.array_length))

        # First, simulate the full unintegrated burst.
        sample_frequency = int(telescope.delta_t / DT)
        length_full_burst = telescope.array_length * sample_frequency
        pulse = self.gaussian_profile(length_full_burst, self.width)

        all_frequencies = np.concatenate([np.arange(*bandpass, DF) for bandpass in telescope.bandpasses])
        print(len(all_frequencies))

        arrival_times = self.arrival_time(
            all_frequencies) * sample_frequency

        # full_burst = np.zeros([len(frequency_range), length_full_burst])
        # print(full_burst.shape)
        # Discretize
        arrival_times = arrival_times.astype(int)
        # Cut everything that falls outside the spectrum.
        # arrival_times = arrival_times[(arrival_times > 0) & (arrival_times < full_burst.shape[1])]

        full_burst = np.zeros([len(all_frequencies), np.max(
            arrival_times) - np.min(arrival_times) + length_full_burst])
        arrival_times = arrival_times - np.min(arrival_times)

        # print(np.min(arrival_times), np.max(arrival_times))

        for i in trange(len(arrival_times)):
            t = arrival_times[i]
            # s = t - length_full_burst // 2
            # e = t + length_full_burst // 2
            # slice_range = slice(max((0, s)), min((length_full_burst), e))
            # print(pulse.shape, s, e)
            full_burst[i, t: t + length_full_burst] += pulse
            # Center the pulse on t_index
            # shift = int(t - length_full_burst // 2)
            # shift = min(length_full_burst, max(-length_full_burst, shift))
            # if shift < 0:
            #     pulse_centered = np.concatenate(
            #         [np.zeros((-shift)), pulse.copy()[:shift]]
            #     )
            # else:
            #     pulse_centered = np.concatenate(
            #         [pulse.copy()[shift:], np.zeros((shift))]
            #     )
            # full_burst[i] += pulse_centered

        # Integrate over time
        print(sample_frequency, full_burst.shape, telescope.delta_t)
        full_burst = full_burst.reshape(
            len(full_burst), -1, sample_frequency).mean(-1)
        print(full_burst.shape)
        return full_burst

        # Find the range that should be integrated over frequency.
        delta_f_rel = telescope.delta_f / (fh - fl)
        range_delta_f = int(delta_f_rel * len(full_burst)) // 2

        burst = []

        for f in telescope.frequencies:
            f_rel = (fh - f) / (fh - fl)
            ix = int(f_rel * len(full_burst))
            integrated_pulse = full_burst[ix -
                                          range_delta_f: ix + range_delta_f].mean(0)
            burst.append(integrated_pulse)

        burst = np.stack(burst)
        print(burst.shape)

        return full_burst

        raise

        for i, f in enumerate(telescope.frequencies):
            # Get arrival time relevant to t_ref (probably half of ntime and median)
            # frequency.
            t_index: int = int(self.arrival_time(telescope.dt, f))

            # Center the pulse on t_index
            shift = int(t_index - telescope.array_length / 2)
            shift = min(telescope.array_length,
                        max(-telescope.array_length, shift))
            if shift < 0:
                pulse_centered = np.concatenate(
                    [np.zeros((-shift)), pulse.copy()[:shift]]
                )
            else:
                pulse_centered = np.concatenate(
                    [pulse.copy()[shift:], np.zeros((shift))]
                )

            # if pulse_centered.max() > 0:
                # pulse_centered = pulse_centered / pulse_centered.max()
            assert np.all(pulse_centered >= 0)

            val = pulse_centered.copy()

            val = val * self.fluence / self.width
            val = val * (f / self.f_ref) ** self.spec_ind
            data[i] = data[i] + val

        return data


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
        # Sample reference t uniformly.
        t_ref = np.random.randint(telescope.array_length)
        event = Event(t_ref, f_ref, dm, fluence, width, spec_ind)
        data = event.simulate(telescope)
        return data, np.array([dm, fluence, width, spec_ind])


class TransientSimulator:
    def __init__(self, config):

        self.telescope = Telescope(
            # type: ignore
            config["frequencies"], config["df"], config["dt"], config["array_length"]
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
            shape=(config["nevents"], len(self.telescope.frequencies),
                   config["array_length"]),  # type: ignore
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
