import torch
from torch import nn
from typing import List, Union, Tuple

from torch.nn import Flatten, Softplus


class TimeFrequencyCNN(nn.Module):

    def __init__(self, config):
        super(TimeFrequencyCNN, self).__init__()
        self.config = config
        self.device = None

        self.act = nn.LeakyReLU(inplace=False)

        self.architecture: List[Union[str, int, Tuple[int, int]]] = config['architecture']
        self.layers = []
        for layer in self.architecture:
            name: str = layer[0]  # type: ignore
            if name == 'act':
                self.layers.append(self.act)
            elif name == 'flatten':
                self.layers.append(Flatten())
            elif name == 'linear':
                in_features: int = layer[1]  # type: ignore
                out_features: int = layer[2] # type: ignore
                self.layers.append(nn.Linear(in_features, out_features))
            elif name == 'conv':
                in_channels: int = layer[1]  # type: ignore
                out_channels: int = layer[2] # type: ignore
                kernel_size: int = layer[3]  # type: ignore
                stride: tuple = layer[4]     # type: ignore
                padding: int = layer[5]      # type: ignore
                self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))  # type: ignore

        self.softplus = Softplus(beta=1e-2)  # type: ignore
        self.nn = nn.Sequential(*self.layers)

    def set_device(self, device: str):
        self.device = device
        self.to(self.device)

    def forward(self, batch: List[torch.Tensor]):  # type: ignore

        input, _ = batch  # type: ignore
        input = (input - input.mean(dim=(-1, -2), keepdims=True)) / input.std(dim=(-1, -2), keepdims=True)
        input: torch.Tensor = torch.clamp(input, -5, 5)  # type: ignore
        output = self.nn(input.to(self.device))
        # Second output is standard deviation.
        means = output[:, :-1]
        variances = self.softplus(output[:, -1:])

        return means, variances

    def load(self, path: str):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict['model_state_dict'])
