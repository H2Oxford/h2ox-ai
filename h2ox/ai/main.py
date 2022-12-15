"""[summary]

TODO:
    - run experiments to check model performances
    - reservoirs at the same endpoint, how to implement this?
    - app.py (vertex ai implementation to serve the model)
Returns:
    [type]: [description]
"""
import json
import os
import pickle
from glob import glob
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from h2ox.ai.dataset import DatasetFactory, maybe_load
from h2ox.ai.dataset.dataset import train_validation_test_split
from h2ox.ai.dataset.utils import calculate_errors, revert_to_levels
from h2ox.ai.experiment import ex
from h2ox.ai.model import initialise_model as initialise_s2s2s
from h2ox.ai.model_bayesian import initialise_bayesian
from h2ox.ai.model_gnn import initialise_gnn
from h2ox.ai.plots import (
    plot_horizon_losses,
    plot_losses,
    plot_test_preds,
    plot_timeseries_over_horizon,
)
from h2ox.ai.train import initialise_training
from h2ox.ai.train import test as test_s2s2s
from h2ox.ai.train import train as train_s2s2s
from h2ox.ai.train_bayesian import test as test_bayesian
from h2ox.ai.train_bayesian import train as train_bayesian


def dict_xr(xrd):
    return dict(zip(xrd["coords"]["global_sites"]["data"], xrd["data"]))


def var_norms_to_json(var_norms):
    var_norms_json = {}
    for norm_type in var_norms.keys():
        var_norms_json[norm_type] = {}
        for var in var_norms[norm_type].keys():
            var_norms_json[norm_type][var] = {}
            for param in var_norms[norm_type][var].keys():
                var_norms_json[norm_type][var][param] = dict_xr(
                    var_norms[norm_type][var][param].to_dict()
                )

    return var_norms_json


@ex.automain
def main(
    _run,
    dataset_parameters: dict,
    data_parameters: dict,
    model_parameters: dict,
    training_parameters: dict,
) -> int:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make temporary path if not exists
    if not os.path.exists("tmp"):
        os.mkdir(os.path.join(os.getcwd(), "tmp"))

    dd = DatasetFactory(
        {"data_parameters": data_parameters, "dataset_parameters": dataset_parameters}
    ).build_dataset()

    print("dataset len")
    print(len(dd))

    if dataset_parameters["norm_difference"]:
        var_norms = dd.augment_dict
        target_var = dd.target_var[0]

        std_target = dict(
            zip(
                var_norms["std_norm"]["shift_targets_WATER_VOLUME"]["std"].to_dict()[
                    "coords"
                ]["global_sites"]["data"],
                var_norms["std_norm"]["shift_targets_WATER_VOLUME"]["std"].to_dict()[
                    "data"
                ],
            )
        )
    else:
        var_norms = None
        target_var = None

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
    if model_parameters["model_str"] == "s2s2s":
        model = initialise_s2s2s(
            item,
            hidden_size=model_parameters["hidden_size"],
            num_layers=model_parameters["num_layers"],
            dropout=model_parameters["dropout"],
        )
        test = test_s2s2s
        train = train_s2s2s

    elif model_parameters["model_str"] == "bayesian":
        model = initialise_bayesian(
            item,
            device=device,
            hidden_size=model_parameters["hidden_size"],
            num_layers=model_parameters["num_layers"],
            dropout=model_parameters["dropout"],
            bayesian=model_parameters["bayesian_lstm"],
            lstm_params=model_parameters["lstm_params"],
        )
        test = test_bayesian
        train = train_bayesian

    elif model_parameters["model_str"] == "gnn":
        model = initialise_gnn(
            item,
            sites=maybe_load(dataset_parameters["select_sites"]),
            sites_edges=maybe_load(dataset_parameters["sites_edges"]),
            flow_std=std_target,
            device=device,
            graph_conv=model_parameters["graph_conv"],
            hidden_size=model_parameters["hidden_size"],
            num_layers=model_parameters["num_layers"],
            dropout=model_parameters["dropout"],
            bayesian_linear=model_parameters["bayesian_linear"],
            bayesian_lstm=model_parameters["bayesian_lstm"],
            lstm_params=model_parameters["lstm_params"],
        )
        test = test_bayesian
        train = train_bayesian

    # force float
    model = model.to(torch.float)

    # setup tensorboard writer
    writer = SummaryWriter(
        os.path.join(
            os.getcwd(), "experiments_no_gconv", "tensorboard", f"tb-{_run._id}"
        )
    )

    # train

    optimizer, scheduler, loss_fn = initialise_training(
        model,
        device=device,
        loss_rate=training_parameters["learning_rate"],
        schedule_params=training_parameters["schedule_params"],
    )

    losses, val_losses = train(
        model,
        train_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        denorm=var_norms,
        denorm_var=target_var,
        writer=writer,
        loss_fn=loss_fn,
        log_every_n_steps=training_parameters["log_every_n_steps"],
        checkpoint_every_n=training_parameters["checkpoint_every_n"],
        epochs=training_parameters["n_epochs"],
        epochs_loss_cliff=training_parameters["epochs_loss_cliff"],
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
    preds = test(model, test_dl, denorm=var_norms, denorm_var=target_var)
    errors = calculate_errors(
        preds,
        obs_var="obs",
        sim_var="sim" if model_parameters["model_str"] == "s2s2s" else "sim-frozen",
        site_dim="site",
    )

    logger.info(f"Generating test performance figures at {filepath}")
    plot_losses(filepath=filepath, losses=losses, val_losses=val_losses)
    plot_horizon_losses(filepath, error=errors["rmse"], identifier="rmse")
    plot_horizon_losses(filepath, error=errors["pearson-r"], identifier="pearson-r")
    plot_timeseries_over_horizon(
        filepath=filepath,
        preds=preds,
        prediction_dim="sim"
        if model_parameters["model_str"] == "s2s2s"
        else "sim-mean",
    )

    if dataset_parameters["target_difference"]:
        preds_levels = revert_to_levels(
            data=dd.xr_ds,
            preds=preds.copy(deep=True),
            target_var=dataset_parameters["target_var"][
                0
            ],  # TODO: if there is more than one target var...
        )
        plot_test_preds(
            filepath=filepath / "test_sites_levels.png",
            preds=preds_levels,
            test_chunks=dataset_parameters["test_date_ranges"],
            site_dim="site",
            main_dim="sim" if model_parameters["model_str"] == "s2s2s" else "sim-mean",
            ci_dims=["ci-95+", "ci-95-"]
            if model_parameters["model_str"] != "s2s2s"
            else None,
        )
        plot_test_preds(
            filepath=filepath / "test_sites_nolevels.png",
            preds=preds,
            test_chunks=dataset_parameters["test_date_ranges"],
            site_dim="site",
            main_dim="sim" if model_parameters["model_str"] == "s2s2s" else "sim-mean",
            ci_dims=["ci-95+", "ci-95-"]
            if model_parameters["model_str"] != "s2s2s"
            else None,
        )
    else:
        plot_test_preds(
            filepath=filepath,
            preds=preds,
            test_chunks=dataset_parameters["test_date_ranges"],
            site_dim="site",
            main_dim="sim" if model_parameters["model_str"] == "s2s2s" else "sim-mean",
            ci_dims=["ci-95+", "ci-95-"]
            if model_parameters["model_str"] != "s2s2s"
            else None,
        )

    logger.info(f"Writing prediction and error datasets to .nc at {filepath}")
    errors.to_netcdf(filepath / "errors.nc")
    preds.to_netcdf(filepath / "preds.nc")
    pickle.dump(item, open(filepath / "dummy_item.pkl", "wb"))
    if var_norms is not None:
        json.dump(var_norms_to_json(var_norms), open(filepath / "var_norms.json", "w"))

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
