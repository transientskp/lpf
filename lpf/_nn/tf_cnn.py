import torch
from torch import nn
from typing import List, Dict, Any

from torch.nn import Flatten, Softplus
import numpy as np
from math import e
from torch import distributions
from torchvision.models import resnet50

MAX_CHANNELS = 512

def build_neural_network(data_shape):
    # ['conv', in_channels, out_channels, kernel_size, stride, padding]
    architecture = []

    assert len(data_shape) == 2

    array_length = data_shape[-1]

    channels = 1
    stride = (1, 2)
    while array_length > 1:
        out_channels = channels * 2
        architecture.append(["conv", channels, out_channels, 3, tuple(stride), 1])
        architecture.append(["act"])
        channels = out_channels
        array_length = np.ceil(array_length / stride[-1])

    architecture.append(["flatten"])

    return architecture, channels


def get_layers(architecture, act):

    layers = []
    for layer in architecture:
        name: str = layer[0]  # type: ignore
        if name == "act":
            layers.append(act)
        elif name == "flatten":
            layers.append(Flatten())
        elif name == "linear":
            in_features: int = layer[1]  # type: ignore
            out_features: int = layer[2]  # type: ignore
            layers.append(nn.Linear(in_features, out_features))
        elif name == "conv":
            in_channels: int = layer[1]  # type: ignore
            out_channels: int = layer[2]  # type: ignore
            kernel_size: int = layer[3]  # type: ignore
            stride: tuple = layer[4]  # type: ignore
            padding: int = layer[5]  # type: ignore
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))  # type: ignore

    return nn.Sequential(*layers)



class TimeFrequencyCNN(nn.Module):
    def __init__(self, data_shape):
        super(TimeFrequencyCNN, self).__init__()
        self.device = None

        assert len(data_shape) == 2
        self.act = nn.LeakyReLU(inplace=False)

        num_parameters = 4
        num_variances = (num_parameters * (num_parameters + 1) // 2)

        architecture, channels = build_neural_network(data_shape)
        self.cnn = get_layers(architecture, self.act)

        features_in = channels * data_shape[0]
        self.nn_mu = nn.Sequential(nn.Linear(features_in, 256), nn.LeakyReLU(), nn.Linear(256, num_parameters))
        self.nn_sigma = nn.Sequential(nn.Linear(features_in, 256), nn.LeakyReLU(), nn.Linear(256, num_variances))

        ix = torch.tril_indices(num_parameters, num_parameters)
        self.ix = (ix[0], ix[1])
                       

    def set_device(self, device: str):
        self.device = device
        self.to(self.device)

    def forward(self, batch: List[torch.Tensor]):  # type: ignore

        input, _ = batch  # type: ignore
        input = input.to(self.device)
        input: torch.Tensor = (input - input.mean(dim=(-1, -2), keepdims=True)) / input.std(dim=(-1, -2), keepdims=True)  # type: ignore
        input: torch.Tensor = torch.clamp(input, -5, 5)  # type: ignore

        z = self.cnn(input)
        means = self.nn_mu(z)
        variances = self.nn_sigma(z)
        p = means.shape[-1]
        l = torch.zeros([p, p, len(means)], device=means.device)
        l[self.ix] = variances.permute(1, 0)
        l[range(p), range(p)] = torch.exp(
            l[range(p), range(p)] + e)

        l = l.permute(2, 0, 1)

        py_x = distributions.MultivariateNormal(means, scale_tril=l)
        return py_x

    def load(self, path: str):
        if self.device is None:
            raise ValueError("Please set device first.")
        state_dict: Dict[str, Any] = torch.load(path, map_location=torch.device(self.device))  # type: ignore
        self.load_state_dict(state_dict["model_state_dict"])
