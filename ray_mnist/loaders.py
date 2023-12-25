from pathlib import Path

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from .interfaces import TrainDataLoader, ValDataLoader
from .utils import split_dataset


def load_data(data_dir: Path):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir.as_posix(), train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir.as_posix(), train=False, download=True, transform=transform
    )

    return trainset, testset


def get_train_and_val_loaders(
    dataset: CIFAR10, batch_size: int
) -> (TrainDataLoader, ValDataLoader):
    train_subset, val_subset = split_dataset(dataset, 0.8)

    trainloader: TrainDataLoader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    valloader: ValDataLoader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=True, num_workers=8
    )

    return trainloader, valloader
