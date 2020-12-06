import numpy as np
from scipy.special import j1  # type: ignore
from lpf.bolts.misc import odd_root


def h(n_1: np.ndarray, n_2: np.ndarray, w: float) -> np.ndarray:
    norm = np.sqrt(n_1 ** 2 + n_2 ** 2)
    return w / (2 * np.pi * norm) * j1(w * norm)


def circular_lowpass_filter(size: int, w: float, order: int = 1):
    assert w <= np.pi

    n_1 = np.arange(-size // 2, size // 2)
    n_2 = np.arange(-size // 2, size // 2)
    f = h(n_1[:, None], n_2[None, :], w)
    f[size // 2, size // 2] = w ** 2 / (4 * np.pi)
    if order != 1:
        f: np.ndarray = odd_root(f, order)

    f = f / f.sum()  # type: ignore
    return f
