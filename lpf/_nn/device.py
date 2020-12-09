# type: ignore
from typing import Iterable, Dict
import torch


def send_to_device_recursively(iterable, device=torch.device('cpu')):
    if not isinstance(iterable, Dict):
        keys = range(len(iterable))
    else:
        keys = iterable

    for k in keys:
        if isinstance(iterable[k], torch.Tensor):
            iterable[k] = iterable[k].to(device)

        elif isinstance(iterable[k], Iterable):
            iterable[k] = send_to_device_recursively(iterable[k], device)

    return iterable