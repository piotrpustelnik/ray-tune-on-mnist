from enum import Enum
from typing import Any

import torch.nn as nn
from pydantic import BaseModel, field_validator
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from ray_mnist.exceptions import CUDANotDiscovered
from ray_mnist.utils import Device, _validate_device


class Trainer(BaseModel):
    model: Any
    device: Device
    criterion: Any
    optimizer: Any

    @field_validator("device")
    @classmethod
    def set_device(cls, v: Device):
        _validate_device(v)


class TrainDataLoader(DataLoader):
    ...


class ValDataLoader(DataLoader):
    ...
