# type: ignore
from astropy.table import Table
import numpy as np


def filter_nan(catalog: Table) -> Table:
    nan = np.isnan(catalog['coordinate'].ra.deg) | np.isnan(catalog['coordinate'].dec.deg)
    return catalog[~nan]

def filter_duplicates(catalog: Table, separation: float = 1) -> Table:
    catalog.sort(keys='detection_flux', reverse=True)
    i = 0
    while i < len(catalog):
        # Perhaps adding an index is faster.
        s = catalog["coordinate"][i]
        idx = s.separation(catalog["coordinate"]).deg > separation
        idx[i] = True
        catalog = catalog[idx]
        i += 1

    return catalog

# def filter_duplicates_and_nan(catalog: Table):
#     nan = np.isnan(catalog['coordinate'].ra.deg) | np.isnan(catalog['coordinate'].dec.deg)
#     catalog = catalog[~nan]
#     catalog.sort(keys='peak_flux', reverse=True)
#     ix, _, _ = catalog['coordinate'].match_to_catalog_sky(catalog['coordinate'])
#     unique = np.unique(ix)
#     return catalog[unique]