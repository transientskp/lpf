from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from definitions import opt_int
from tqdm import tqdm  # type: ignore
from typing import List, Dict, Any, Union
import torch
from lpf._nn.device import send_to_device_recursively
from definitions import opt_int
from torch.utils import tensorboard
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os


class Trainer:
    def __init__(
        self,
        config,  # type: ignore
        output_folder: str,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,  # type: ignore
        val_loader: DataLoader,  # type: ignore
        test_loader: DataLoader,  # type: ignore
        early_stopping_patience: int = 5,
    ) -> None:
        super().__init__()
        self.optim = Adam(model.parameters(), config["lr"])  # type: ignore
        self.train_loader: DataLoader = train_loader  # type: ignore
        self.val_loader: DataLoader = val_loader  # type: ignore
        self.model = model
        self.loss_fn = loss_fn
        self.current_epoch: int = 0
        self.epochs: int = config["epochs"]
        self.output_folder = output_folder
        self.writer = tensorboard.SummaryWriter(output_folder)
        self.best_val_loss = float('inf')
        self.early_stopping_patience = early_stopping_patience
        self.patience = early_stopping_patience

        self.is_training: bool = True

    def handle_log(
        self, result: Dict["str", Union[float, Any]], divider: str, index: int
    ):
        for k in result:
            k: str
            if isinstance(result[k], (torch.Tensor, np.ndarray, float, int, np.int, np.float)):  # type: ignore
                self.writer.add_scalar(f"{k}/{divider}", result[k], index)  # type: ignore
            elif isinstance(result[k], plt.Figure):
                self.writer.add_figure(f"examples/{divider}", result[k], index)  # type: ignore

    def save_model(self, path: str):
        torch.save(  # type: ignore
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),  # type: ignore
            },
            path,
        )

    def _validate_loop(self, num_iterations: opt_int = None):

        self.model.eval()
        num_iterations = (
            num_iterations if num_iterations is not None else len(self.val_loader)  # type: ignore
        )

        pbar = tqdm(
            self.val_loader,  # type: ignore
            position=0,
            total=num_iterations,
        )

        total_loss: float = 0
        results = []
        with torch.no_grad():  # type: ignore
            for batch_idx, batch in enumerate(pbar):  # type: ignore
                batch: List[torch.Tensor]
                model_output = self.model(batch)
                result = self.loss_fn(model_output, batch, batch_idx)

                total_loss: float = total_loss + result["loss"].item()
                desc = (
                    f"Epoch {self.current_epoch} - Validating - Loss: "
                    f"{total_loss / (batch_idx + 1):7.3f}"
                )
                pbar.set_description(desc)  # type: ignore
                result: Dict[str, Any] = send_to_device_recursively(
                    result, torch.device("cpu")
                )
                results.append(result)

                if pbar.n >= pbar.total:  # type: ignore
                    break

        return results

    def train(self):

        self.model.train()

        pbar = tqdm(
            self.train_loader,  # type: ignore
            position=0,
        )

        total_loss: float = 0

        for batch_idx, batch in enumerate(pbar):  # type: ignore
            batch: List[torch.Tensor]
            model_output = self.model(batch)

            self.optim.zero_grad()
            result = self.loss_fn(model_output, batch, batch_idx)
            result["loss"].backward()
            self.optim.step()

            total_loss: float = total_loss + result["loss"].item()
            desc = (
                f"Epoch {self.current_epoch} - Training - Loss: "
                f"{total_loss / (batch_idx + 1):7.3f}"
            )
            pbar.set_description(desc)  # type: ignore

            train_idx: int = len(self.train_loader) * (self.current_epoch - 1) + batch_idx  # type: ignore
            self.handle_log(result, "train", train_idx)

    def validate(self, num_iterations: opt_int = None):

        self.model.eval()
        results = self._validate_loop(num_iterations)

        epoch_results: defaultdict[str, Any] = defaultdict(list)
        for r in results:
            for k in r:
                epoch_results[k].append(r[k])

        for k in epoch_results:
            if isinstance(epoch_results[k][0], (torch.Tensor, np.ndarray, float, int)):
                epoch_results[k] = np.mean(epoch_results[k])  # type: ignore
            elif isinstance(epoch_results[k][0], (plt.Figure)):
                epoch_results[k] = epoch_results[k][0]
            else:
                raise ValueError(
                    f"Don't know how to log {type(epoch_results[k][0])} for key {k}"
                )

        self.handle_log(epoch_results, "validation", self.current_epoch)

        loss = epoch_results["loss"]

        if loss < self.best_val_loss:
            self.best_val_loss = loss
            if self.writer:
                self.save_model(
                    os.path.join(
                        self.writer.log_dir,  # type: ignore
                        f"epoch_{self.current_epoch}.ckpt",
                    ),
                )
            self.patience = self.early_stopping_patience

        elif self.patience > 0:
            self.patience -= 1
        else:
            print(
                f"Early stopping reached at validation loss "
                f"{self.best_val_loss:.3f}"
            )
            self.is_training = False

        self.current_epoch += 1

    def run(self):

        print("Testing validation routine.")
        self.validate(num_iterations=1)

        while self.is_training:
            self.train()
            self.validate()
