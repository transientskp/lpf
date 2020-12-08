# type: ignore
from scipy.constants import e, pi, epsilon_0, parsec, m_e, c

# MHz^2 cm^3 ms
# 4148.806423890001
DM_CONST = 1e-6 * (e ** 2) / (8 * pi ** 2 * epsilon_0 * m_e * c) * parsec


def disp_delay(freq, dm, disp_ind=2.0):
    """Compute the dispersion delay (s) as a function of frequency (MHz) and DM"""
    return DM_CONST * dm / (freq ** disp_ind)
