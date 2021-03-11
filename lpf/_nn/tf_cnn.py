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
        # self.architecture_mu = build_neural_network(data_shape, num_parameters)
        # self.nn_mu = get_layers(self.architecture_mu, self.act)

        num_variances = (num_parameters * (num_parameters + 1) // 2)
        # self.architecture_sigma = build_neural_network(data_shape, num_variances)
        # self.nn_sigma = get_layers(self.architecture_sigma, self.act)

        architecture, channels = build_neural_network(data_shape)
        self.cnn = get_layers(architecture, self.act)

        # self.softplus = Softplus(0.1)  # type: ignore

        # print(self.cnn)

        # self.nn_mu = nn.Sequential(nn.Linear(channels, channels), self.act, nn.Linear(channels, 4))
        # self.nn_sigma = nn.Sequential(nn.Linear(channels, channels), self.act, nn.Linear(channels, 1))

        # self.nn_mu = nn.Sequential(nn.Flatten(), nn.Linear(8192, 1024), nn.LeakyReLU(), nn.Linear(1024, 4))
        # self.nn_sigma = nn.Sequential(nn.Flatten(), nn.Linear(8192, 1024), nn.LeakyReLU(), nn.Linear(1024, num_variances))

        # self.nn_mu = resnet50(pretrained=False)
        # self.nn_mu.fc = nn.Linear(2048, 5)
        # self.nn_mu.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
        #                 bias=False)

        # self.nn_sigma = resnet50(pretrained=True)
        # self.nn_sigma.fc = nn.Linear(2048, 1) 
        # self.nn_sigma.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # self.cnn = resnet50(pretrained=False)
        # self.cnn.fc = nn.Linear(2048, 512)
        # self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
        #                 bias=False)
        # self.nn_mu = nn.Sequential(self.cnn, nn.LeakyReLU(), nn.Linear(512, 4))
        # self.nn_sigma = nn.Sequential(self.cnn, nn.LeakyReLU(), nn.Linear(512, 1))

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(1, 2, (3, 3), padding=1, stride=(1, 2)),
        #     nn.LeakyReLU(),
        #     # nn.MaxPool2d((1, 2), (1, 2)),
        #     nn.Conv2d(2, 4, (3, 3), padding=1, stride=(1, 2)),
        #     nn.LeakyReLU(),
        #     # nn.MaxPool2d((1, 2), (1, 2)),
        #     nn.Conv2d(4, 8, (3, 3), padding=1, stride=(1, 2)),
        #     nn.LeakyReLU(),
        #     # nn.MaxPool2d((1, 2), (1, 2)),
        #     nn.Conv2d(8, 16, (3, 3), padding=1, stride=(1, 2)),
        #     nn.LeakyReLU(),
        #     # nn.MaxPool2d((1, 2), (1, 2)),
        #     nn.Conv2d(16, 32, (3, 3), padding=1, stride=(1, 2)),
        #     nn.LeakyReLU(),
        #     # nn.MaxPool2d((1, 2), (1, 2)),
        #     nn.Conv2d(32, 64, (3, 3), padding=1, stride=(1, 2)),
        #     nn.LeakyReLU(),
        #     # nn.MaxPool2d((1, 2), (1, 2)),
        #     nn.Conv2d(64, 128, (3, 3), padding=1, stride=(1, 2)),
        #     nn.LeakyReLU(),
        #     # nn.MaxPool2d((1, 2), (1, 2)),
        #     nn.Conv2d(128, 256, (3, 3), padding=1, stride=(1, 2)),
        #     nn.LeakyReLU(),
        #     # nn.MaxPool2d((1, 2), (1, 2)),
        #     nn.Conv2d(256, 512, (3, 3), padding=1, stride=(1, 2)),
        #     nn.LeakyReLU(),
        #     # nn.MaxPool2d((1, 2), (1, 2))
        # )

        features_in = channels * data_shape[0]
        self.nn_mu = nn.Sequential(nn.Linear(features_in, 256), nn.LeakyReLU(), nn.Linear(256, 4))
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

        # input = input.flatten(start_dim=1)
        # print(input.mean(), input.std(), input.max(), input.min())
        # for module in self.nn:
        # output = module(output)
        # print(input.shape)
        # raise
        z = self.cnn(input)
        # z = torch.flatten(z, 1)
        # means = self.nn_mu(z_mu)
        # z_var = self.cnn_sigma(input)
        means = self.nn_mu(z)
        variances = self.nn_sigma(z)

        # means = self.nn_mu(input)
        # variances = self.nn_sigma(input)

        # means = means[..., :-1]
        # variances = means[..., :-1:].detach()
        # variances = self.softplus(self.nn_sigma(z))
        # variances = self.nn_sigma(z).exp()


        # input = input.flatten(start_dim=1)
        # means = self.nn_mu(input)
        # variances = self.nn_sigma(input)
        # variances = self.nn_sigma(input.flatten(start_dim=1)).exp()
        # variances = self.nn_sigma(input).abs() ** 0.1
        # variances = self.nn_sigma(input) ** 2
        # variances = 1.00000000001 ** self.nn_sigma(input)
        p = means.shape[-1]
        l = torch.zeros([p, p, len(means)], device=means.device)
        # variances = (variances + e).exp()
        l[self.ix] = variances.permute(1, 0)
        l[range(p), range(p)] = torch.exp(
            l[range(p), range(p)] + e)

        l = l.permute(2, 0, 1)

        # variances = variances.exp() ** 2
        # variances = self.nn_sigma(input).abs()
        
        # Second output is standard deviation.
        # means = output[:, :-1]
        # variances = self.softplus(output[:, -1:])
        # variances = torch.sqrt(torch.exp(output[:, -1:]))
        # l = torch.diag_embed(variances)
        # print(l.shape)
        # n = distributions.MultivariateNormal(means, scale_tril=l)
        # cov = n.covariance_matrix
        # print(cov.shape)
        # print(l[0])
        # print(cov[0])
        # print(variances[0])
        # raise

        return means, l

    def load(self, path: str):
        if self.device is None:
            raise ValueError("Please set device first.")
        state_dict: Dict[str, Any] = torch.load(path, map_location=torch.device(self.device))  # type: ignore
        self.load_state_dict(state_dict["model_state_dict"])
