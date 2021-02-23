# type: ignore
from torch import nn
import os
from lpf._nn import Trainer
from lpf._nn.tf_cnn_auto import TimeFrequencyCNN
import yaml
import sys
import random
import numpy as np
from torch.utils.data import Dataset
from numpy.lib.format import open_memmap
import torch
from lpf._nn import configure_dataloaders
from torch import distributions
from torch.nn import functional as F
import shutil
import logging

logger = logging.getLogger(__name__)

def get_config():

    cmd = sys.argv
    if len(cmd) != 2:
        raise ValueError(
            f"Please provide path to config file. Example: train.py /path/to/config.yml"
        )
    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)  # type: ignore
    return config


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def log_likelihood(self, t, y_mu, y_sigma):
        d = distributions.Independent(distributions.Normal(y_mu, y_sigma), 1)
        return d.log_prob(t).mean(0)

    def forward(self, model_output, batch, batch_idx):
        means = model_output[0]
        variances = model_output[1]
        target = batch[1]
        dm_t = target[:, :1]
        dm_p = means[:, :1]

        other_outputs = means[:, 1:]
        other_targets = target[:, 1:]

        other_loss = F.mse_loss(other_outputs, other_targets.to(other_outputs.device))
        dm_nll = -self.log_likelihood(dm_t.to(dm_p.device), dm_p, variances)
        loss = other_loss + dm_nll

        if batch_idx == 0:
            result = {
                "loss": loss,
            }

        else:
            result = {"loss": loss}

        return result


class CustomTransientDataset(Dataset):  # type: ignore
    def __init__(
        self,
        transient_path,
        parameter_path,
        noise_multiplier,
        noise_path=None,
    ):
        super(CustomTransientDataset, self).__init__()

        self.transient_mmap = open_memmap(transient_path, mode="r")
        self.parameter_mmap = open_memmap(parameter_path, mode="r")
        self.noise_multiplier = noise_multiplier
        if noise_path is not None:
            self.noise_mmap = open_memmap(noise_path, mode="r")
            self.noise_len = len(self.noise_mmap)
        else:
            self.noise_mmap = None

        self.shape = self.transient_mmap.shape[-2:]

        self.length = len(self.transient_mmap)

        print(f"Transient shapes: {self.transient_mmap.shape}")
        if self.noise_mmap is not None:
            print(f"Noise shapes: {self.noise_mmap.shape}")

    def __getitem__(self, index: int):
        tr_data = self.transient_mmap[index]
        param_data = self.parameter_mmap[index]

        if self.noise_mmap is not None:
            noise_index = random.randint(0, self.noise_len - 1)
            noise_data = self.noise_mmap[noise_index]
        else:
            # Will be added in collate_fn
            noise_data = None

        return (
            tr_data.astype(np.float32),
            noise_data,
            param_data.astype(np.float32),
            self.noise_multiplier
        )

    def __len__(self):
        return self.length


def collate_fn(batch):
    tr_data, noise_data, param_data, noise_multiplier = zip(*batch)
    tr_data = np.stack(tr_data)
   
    param_data = np.stack(param_data)
    mult_data = np.stack(noise_multiplier).astype(np.float32)

    if noise_data[0] is not None:
        noise_data = np.stack(noise_data).astype(np.float32)
    else:
        noise_data = np.random.randn(*tr_data.shape).astype(np.float32)

    noise_data = (
        noise_data - noise_data.mean(axis=(-1, -2), keepdims=True)
    ) / noise_data.std(axis=(-1, -2), keepdims=True)

    to_return = torch.from_numpy(noise_data * mult_data[:, None, None] + tr_data)

    return to_return[:, None], torch.from_numpy(param_data)


def load_data(config, train_split=0.7, val_split=0.2):

    if train_split + val_split > 0.9:
        print("WARNING: test set less than 10% of total data.")
    dataset = CustomTransientDataset(
        os.path.join(config["simulation_output_folder"], 'data.npy'),
        os.path.join(config["simulation_output_folder"], 'parameters.npy'),
        config["noise_multiplier"],
        config["noise_path"],
    )
    train_len = int(train_split * len(dataset))
    val_len = int(val_split * len(dataset))
    test_len = len(dataset) - train_len - val_len
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_len, val_len, test_len)
    )
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0
    assert len(test_dataset) > 0
    return dataset, train_dataset, val_dataset, test_dataset


def main():
    config = get_config()
    output_folder = config['nn_output_folder']
    os.makedirs(output_folder)
    shutil.copy(sys.argv[1], output_folder)
    dataset, train_dataset, val_dataset, test_dataset = load_data(config)

    train_loader = configure_dataloaders(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader, test_loader = configure_dataloaders(
        val_dataset,
        test_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        collate_fn=collate_fn
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device %s", device)

    tf_cnn = TimeFrequencyCNN([len(config["frequencies"]), config["array_length"]])
    tf_cnn.set_device(device)
    loss_fn = CustomLoss()
    trainer = Trainer(
       config, output_folder, tf_cnn, loss_fn, train_loader, val_loader, test_loader
    )
    trainer.run()


if __name__ == "__main__":
    main()