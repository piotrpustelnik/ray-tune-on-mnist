from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from ray.air import session
from ray.train import Checkpoint

from ray_mnist.constants import DATA_DOWNLOAD_ROOT
from ray_mnist.interfaces import (Device, TrainDataLoader, Trainer,
                                  ValDataLoader)
from ray_mnist.loaders import get_train_and_val_loaders, load_data
from ray_mnist.models import Net


def donwload_data_and_hypertune(config: Dict[str, Any]):
    trainset, testset = load_data(DATA_DOWNLOAD_ROOT)

    train_loader, val_loader = get_train_and_val_loaders(
        trainset,
        batch_size=100,
    )

    hypertune(config, (train_loader, val_loader))


def hypertune(config: Dict[str, Any], loaders: (TrainDataLoader, ValDataLoader)):
    model: nn.Module = Net(config["l1"], config["l2"])
    device = Device.CUDA
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    trainer = Trainer(
        model=model, device=device, criterion=criterion, optimizer=optimizer
    )

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    train(trainer, loaders=loaders, start_epoch=start_epoch)


def train(trainer: Trainer, loaders: [TrainDataLoader, ValDataLoader], start_epoch=0):
    net: nn.Module = trainer.model
    net.to(trainer.device)

    train_loader, val_loader = loaders

    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
        net, loss, optimizer = train_one_epoch(
            trainer, (train_loader, val_loader), epoch
        )
        val_loss, val_steps, correct, total = validate_model(
            net, val_loader, trainer.criterion, trainer.device
        )

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        # checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"loss": val_loss / val_steps, "accuracy": correct / total},
            # checkpoint=checkpoint,
        )

    print("Finished Training")


def validate_model(net, valloader, criterion, device):
    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    for i, data in enumerate(valloader, 0):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            val_loss += loss.cpu().numpy()
            val_steps += 1

    return (val_loss, val_steps, correct, total)


def train_one_epoch(trainer: Trainer, loaders, epoch: int):
    trainloader, valloader = loaders
    device = trainer.device
    optimizer = trainer.optimizer
    net = trainer.model
    criterion = trainer.criterion

    running_loss = 0.0
    epoch_steps = 0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs.to(device)
        labels.to(device)

        net, loss, optimizer = train_one_batch(
            net, criterion, optimizer, (inputs, labels)
        )

        # print statistics
        running_loss += loss.item()
        epoch_steps += 1
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(
                "[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / epoch_steps)
            )
            running_loss = 0.0

    return net, loss, optimizer


def train_one_batch(net, criterion, optimizer, batch):
    inputs, labels = batch

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    return net, loss, optimizer
