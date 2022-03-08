"""[summary]

TODO:
    - run experiments to check model performances
    - reservoirs at the same endpoint, how to implement this?
    - app.py (vertex ai implementation to serve the model)
Returns:
    [type]: [description]
"""
import os
from glob import glob
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from h2ox.ai.dataset import DatasetFactory
from h2ox.ai.dataset.dataset import train_validation_test_split
from h2ox.ai.dataset.utils import calculate_errors
from h2ox.ai.experiment import ex
from h2ox.ai.model import initialise_model
from h2ox.ai.plots import plot_horizon_losses, plot_losses, plot_timeseries_over_horizon
from h2ox.ai.train import initialise_training, test, train


@ex.automain
def main(
    _run,
    dataset_parameters: dict,
    data_parameters: dict,
    model_parameters: dict,
    training_parameters: dict,
) -> int:

    # make temporary path if not exists
    if not os.path.exists("tmp"):
        os.mkdir(os.path.join(os.getcwd(), "tmp"))

    dd = DatasetFactory(
        {"data_parameters": data_parameters, "dataset_parameters": dataset_parameters}
    ).build_dataset()

    # train-validation split
    train_dd, validation_dd, test_dd = train_validation_test_split(
        dd,
        cfg=dataset_parameters,
        time_dim="date",
    )

    # build dataloaders
    train_dl = DataLoader(
        train_dd,
        batch_size=training_parameters["batch_size"],
        shuffle=False,
        num_workers=training_parameters["num_workers"],
    )
    val_dl = DataLoader(
        validation_dd,
        batch_size=training_parameters["batch_size"],
        shuffle=False,
        num_workers=training_parameters["num_workers"],
    )
    test_dl = DataLoader(
        test_dd,
        batch_size=training_parameters["batch_size"],
        shuffle=False,
        num_workers=training_parameters["num_workers"],
    )

    item = dd.__getitem__(0)

    # initialise model
    model = initialise_model(
        item,
        hidden_size=model_parameters["hidden_size"],
        num_layers=model_parameters["num_layers"],
        dropout=model_parameters["dropout"],
    )

    # force float
    model = model.to(torch.float)

    # setup tensorboard writer
    writer = SummaryWriter(
        os.path.join(os.getcwd(), "experiments", "tensorboard", f"tb-{_run._id}")
    )

    # train
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer, scheduler, loss_fn = initialise_training(
        model, device=device, loss_rate=1e-3
    )

    losses, val_losses = train(
        model,
        train_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        writer=writer,
        loss_fn=loss_fn,
        log_every_n_steps=training_parameters["log_every_n_steps"],
        checkpoint_every_n=training_parameters["checkpoint_every_n"],
        epochs=training_parameters["n_epochs"],
        val_dl=val_dl,
        validate_every_n=training_parameters["validate_every_n"],
        experiment=ex,
    )

    filepath = Path("tmp")

    logger.info("Archiving model and optimizer checkpoints")
    ex.add_artifact(filepath / f"model_epoch{training_parameters['n_epochs']-1:03d}.pt")
    ex.add_artifact(
        filepath / f"optimizer_state_epoch{training_parameters['n_epochs']-1:03d}.pt"
    )

    logger.info("Calculating test performance")
    preds = test(model, test_dl)
    errors = calculate_errors(
        preds,
        var="Y",
        site_dim="site",
        horizon_dim="step",
        model_str="s2s-ohe",
    )

    logger.info(f"Generating test performance figures at {filepath}")
    plot_losses(filepath=filepath, losses=losses, val_losses=val_losses)
    plot_horizon_losses(filepath, error=errors["rmse"], identifier="rmse")
    plot_horizon_losses(filepath, error=errors["pearson-r"], identifier="pearson-r")
    plot_timeseries_over_horizon(filepath=filepath, preds=preds)

    logger.info(f"Writing prediction and error datasets to .nc at {filepath}")
    errors.to_netcdf(filepath / "errors.nc")
    preds.to_netcdf(filepath / "preds.nc")

    logger.info("Archiving plot and data artifacts")
    artifacts = glob(str(filepath / "*"))
    select_artifacts = [
        f for f in artifacts if "model" not in f or "optimizer" not in f
    ]
    for artifact in select_artifacts:
        ex.add_artifact(artifact)

    # cleanup
    if training_parameters["cleanup"]:
        for artifact in artifacts:
            os.remove(artifact)

    writer.close()

    return 1
