# type: ignore
from torch.utils.data import DataLoader


def configure_dataloaders(*datasets, batch_size, num_workers, shuffle, collate_fn=None):
    """Configure dataloaders from datasets

    Parameters
    ----------
    shuffle : bool
        tuple of whether to shuffle or not
    """
    kwargs = {
        "pin_memory": True,
        "drop_last": True,
    }
    loaders = [
        DataLoader(
            ds,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            batch_size=batch_size,
            **kwargs
        )
        for ds in datasets
    ]
    return loaders if len(loaders) > 1 else loaders[0]