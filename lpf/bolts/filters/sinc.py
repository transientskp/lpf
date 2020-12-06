# type: ignore
import numpy as np
from scipy.ndimage import _ni_support
from scipy.ndimage.filters import correlate1d

def sinc(x, s=1):
    sinc = np.sin(x / s) / (x / s)
    sinc[x == 0] = 1
    return sinc


def _sinc_kernel1d(sigma, radius):
    x = np.arange(-radius, radius + 1)
    weights = sinc(x, sigma)
    weights = weights / weights.sum()
    return weights


def sinc_filter1d(
    input, sigma, axis=-1, order=0, output=None, mode="reflect", cval=0.0, truncate=6.0
):
    lw = int(truncate * sigma + 0.5)
    weights = _sinc_kernel1d(sigma, lw)[::-1]
    return correlate1d(input, weights, axis, output, mode, cval, 0)


def sinc_filter(input, sigma, order=0, output=None,
                    mode="reflect", cval=0.0, truncate=6.0):
    input = np.asarray(input)
    output = _ni_support._get_output(output, input)
    orders = _ni_support._normalize_sequence(order, input.ndim)
    sigmas = _ni_support._normalize_sequence(sigma, input.ndim)
    modes = _ni_support._normalize_sequence(mode, input.ndim)
    axes = list(range(input.ndim))
    axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii])
            for ii in range(len(axes)) if sigmas[ii] > 1e-15]
    if len(axes) > 0:
        for axis, sigma, order, mode in axes:
            sinc_filter1d(input, sigma, axis, order, output,
                              mode, cval, truncate)
            input = output
    else:
        output[...] = input[...]
    return output
