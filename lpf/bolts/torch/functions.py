# type: ignore
import torch

def nanmax(data: torch.Tensor, axis=None):
    nan_mask = torch.isnan(data)
    # Probably there's a more efficient way than imputing with the minimum.
    data[nan_mask] = torch.min(data[~nan_mask])
    maxima = torch.amax(data, axis=axis)
    return maxima


def nanmean(v, *args, is_nan=None, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
        
    if is_nan is None:
        is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)