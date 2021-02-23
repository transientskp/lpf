# type: ignore
from astropy.table import Table
import numpy as np


def filter_nan(catalog: Table) -> Table:
    nan = np.isnan(catalog["coordinate"].ra.deg) | np.isnan(
        catalog["coordinate"].dec.deg
    )
    return catalog[~nan]


def filter_duplicates(catalog: Table, separation: float = 1) -> Table:
    catalog.sort(keys="detection_flux", reverse=True)
    i = 0
    while i < len(catalog):
        # Perhaps adding an index is faster.
        s = catalog["coordinate"][i]
        idx = s.separation(catalog["coordinate"]).deg > separation
        idx[i] = True
        catalog = catalog[idx]
        i += 1

    return catalog