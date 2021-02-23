import torch
from torch import nn
from typing import List, Dict, Any

from torch.nn import Flatten, Softplus
import numpy as np

MAX_CHANNELS = 512
N_OUTPUT = 5


def build_neural_network(data_shape):
    # ['conv', in_channels, out_channels, kernel_size, stride, padding]
    channels = 8
    architecture = [
        ["conv", 1, channels, 1, [1, 1], 0],
        ["act"],
    ]

    min_dim, max_dim = np.argsort(data_shape)
    if max_dim == 0:
        stride = np.array([2, 1])
    elif max_dim == 1:
        stride = np.array([1, 2])
    else:
        raise ValueError(f"Data shape {data_shape} incorrect.")

    current_shape = np.array(data_shape)

    # While halving the max_dim brings it closer to the min_dim
    # print(current_shape)
    while abs(current_shape[max_dim] / 2 - current_shape[min_dim]) < abs(
        current_shape[max_dim] - current_shape[min_dim]
    ):
        out_channels = min(MAX_CHANNELS, channels * 2)
        architecture.append(["conv", channels, out_channels, 3, tuple(stride), 1])
        architecture.append(["act"])
        channels = out_channels
        current_shape = np.ceil(current_shape / stride)

    stride = np.array([2, 2])
    while np.any(current_shape > 1):
        out_channels = min(MAX_CHANNELS, channels * 2)
        architecture.append(["conv", channels, out_channels, 3, tuple(stride), 1])
        architecture.append(["act"])
        channels = out_channels
        current_shape = np.ceil(current_shape / np.array(stride))

        if np.any(current_shape == 1):
            assert np.all(current_shape == 1)

    architecture.append(["flatten"])
    architecture.append(["linear", channels, N_OUTPUT])

    return architecture


class TimeFrequencyCNN(nn.Module):
    def __init__(self, data_shape):
        super(TimeFrequencyCNN, self).__init__()
        self.device = None

        self.act = nn.LeakyReLU(inplace=False)

        self.architecture = build_neural_network(data_shape)

        self.layers = []
        for layer in self.architecture:
            name: str = layer[0]  # type: ignore
            if name == "act":
                self.layers.append(self.act)
            elif name == "flatten":
                self.layers.append(Flatten())
            elif name == "linear":
                in_features: int = layer[1]  # type: ignore
                out_features: int = layer[2]  # type: ignore
                self.layers.append(nn.Linear(in_features, out_features))
            elif name == "conv":
                in_channels: int = layer[1]  # type: ignore
                out_channels: int = layer[2]  # type: ignore
                kernel_size: int = layer[3]  # type: ignore
                stride: tuple = layer[4]  # type: ignore
                padding: int = layer[5]  # type: ignore
                self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))  # type: ignore

        self.softplus = Softplus(beta=1e-1)  # type: ignore
        self.nn = nn.Sequential(*self.layers)

    def set_device(self, device: str):
        self.device = device
        self.to(self.device)

    def forward(self, batch: List[torch.Tensor]):  # type: ignore

        input, _ = batch  # type: ignore
        input: torch.Tensor = (input - input.mean(dim=(-1, -2), keepdims=True)) / input.std(dim=(-1, -2), keepdims=True)  # type: ignore
        input: torch.Tensor = torch.clamp(input, -5, 5)  # type: ignore
        output = input
        # for module in self.nn:
        # output = module(output)
        output = self.nn(input.to(self.device))
        # Second output is standard deviation.
        means = output[:, :-1]
        variances = self.softplus(output[:, -1:])

        return means, variances

    def load(self, path: str):
        if self.device is None:
            raise ValueError("Please set device first.")
        state_dict: Dict[str, Any] = torch.load(path, map_location=torch.device(self.device))  # type: ignore
        self.load_state_dict(state_dict["model_state_dict"])
