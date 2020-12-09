import torch
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset, DataLoader
import os
import glob
import shutil
from typing import List
import numpy as np


class MemMapDataset(Dataset):  # type: ignore
    def __init__(
        self,
        directory,
        scratch_dir=os.environ["TMPDIR"],
    ):
        """Initialize Memory Mapped Dataset

        Parameters
        ----------
        directory : str
            directory with memory-mapped files (.npy)
        scratch_dir : str, optional
            copy files here before training, by default os.environ['TMPDIR']
        """
        super(MemMapDataset, self).__init__()

        self.npy_files = glob.glob(
            os.path.join(directory, "**", "*.npy"), recursive=True
        )
        self.scratch_dir = scratch_dir
        self.scratch_files = [
            os.path.join(self.scratch_dir, os.path.split(f)[-1]) for f in self.npy_files
        ]

        for i in range(len(self.scratch_files)):
            shutil.copyfile(self.npy_files[i], self.scratch_files[i])

        self.mmaps: List[np.ndarray] = [open_memmap(f, mode="r") for f in self.scratch_files]
        self.len = len(self.mmaps[0])
        assert all(self.len == len(mmap) for mmap in self.mmaps)

    def __getitem__(self, index: int):
        return [torch.from_numpy(mmap[index : index + 1].copy()).float() for mmap in self.mmaps]  # type: ignore

    def __len__(self):
        return self.len