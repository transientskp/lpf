#type: ignore
import numpy as np


def odd_root(x, k):
    return np.sign(x) * np.abs(x) ** (1 / k)