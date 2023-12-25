from functools import partial
from typing import Any, Dict

from ray import tune
from ray.tune.schedulers import ASHAScheduler

from ray_mnist.constants import DATA_DOWNLOAD_ROOT
from ray_mnist.interfaces import TrainDataLoader, ValDataLoader
from ray_mnist.loaders import get_train_and_val_loaders, load_data
from ray_mnist.training import donwload_data_and_hypertune, hypertune


def main(
    config: Dict[Any, str],
    max_num_epochs=100,
    num_samples=10,
):
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    gpus_per_trial = 1

    result = tune.run(
        donwload_data_and_hypertune,
        resources_per_trial={"cpu": 6, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    config = {
        "l1": tune.choice([2**i for i in range(9)]),
        "l2": tune.choice([2**i for i in range(9)]),
        "lr": tune.loguniform(1e-4, 1e-1),
    }

    # model = Net(config["l1"], config["l2"])
    # device = Device.CUDA
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    # trainer = Trainer(model, device, criterion, optimizer)

    main(config)
