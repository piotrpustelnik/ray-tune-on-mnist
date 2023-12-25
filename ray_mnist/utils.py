from enum import Enum

import torch
from torch.utils.data import Subset, random_split

from ray_mnist.exceptions import CUDANotDiscovered


class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda"


def _validate_device(v: Device) -> Device | str:
    if v == Device.CPU:
        return v

    elif v == Device.CUDA:
        if torch.cuda.is_available():
            device = "cuda:0"
            return device
        raise CUDANotDiscovered(
            "CUDA device passed to Trainer but CUDA is not discoverable by Pytorch."
        )


def split_dataset(trainset, train_ratio: float = 0.8) -> (Subset, Subset):
    test_abs = int(len(trainset) * train_ratio)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    return train_subset, val_subset
