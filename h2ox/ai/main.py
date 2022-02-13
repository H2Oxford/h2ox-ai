"""[summary]

TODO:
    - run experiments to check model performances
    - reservoirs at the same endpoint, how to implement this?
    - app.py (vertex ai implementation to serve the model)
Returns:
    [type]: [description]
"""
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

from h2ox.ai.dataset import DatasetFactory
from h2ox.ai.experiment import ex
from h2ox.ai.experiment_utils import plot_losses
from h2ox.ai.model import initialise_model
from h2ox.ai.train import initialise_training, test, train, train_validation_test_split


@ex.automain
def main(
    dataset_parameters: dict,
    data_parameters: dict,
    model_parameters: dict,
    training_parameters: dict,
) -> int:
    # load data
    Path.cwd() / "data"

    dd = DatasetFactory(
        {"data_parameters": data_parameters, "dataset_parameters": dataset_parameters}
    ).build_dataset()

    # train-validation split
    train_dd, validation_dd, test_dd = train_validation_test_split(
        dd,
        cfg=dataset_parameters,
        time_dim="date",
    )

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

    item = dd.__getitem__(0)

    # initialise model
    model = initialise_model(
        item,
        hidden_size=model_parameters["hidden_size"],
        num_layers=model_parameters["num_layers"],
        dropout=model_parameters["dropout"],
    )

    # # train
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer, scheduler, loss_fn = initialise_training(
        model, device=device, loss_rate=1e-3
    )

    losses, val_losses = train(
        model,
        train_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=training_parameters["n_epochs"],
        val_dl=val_dl,
        validate_every_n=training_parameters["validate_every_n"],
        experiment=ex,
    )

    # get filepath for experiment dir from the Sacred Experiment
    # TODO(tl): is ex a global variable?
    filepath = Path(ex.observers[0].dir) if ex.observers[0].dir is not None else None
    if filepath is not None:
        logger.info(f"Saving losses.png to {filepath}")
        plot_losses(filepath=filepath, losses=losses, val_losses=val_losses)

    logger.info("Generate Preds")
    pred_ds = test(model, val_dl)
    pred_ds.to_netcdf("./preds_interim.nc")

    # # test
    """ TODO: change test_dd to come from split above
    if eval_test:
        # load dataset
        test_dd = FcastDataset(
            target=test_target,  # target,
            history=test_history.sel(location=sites),  # history,
            forecast=test_forecast,  # forecast,
            encode_doy=encode_doy,
            historical_seq_len=seq_len,
            future_horizon=future_horizon,
            target_var=target_var,
            mode="test",
            history_variables=history_variables,
            forecast_variables=forecast_variables,
            include_ohe=include_ohe,
        )

        test_dl = DataLoader(
            test_dd, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    else:
        test_dl = val_dl

    preds = test(model, test_dl)
    # preds = unnormalize_preds(preds, target_mn, target_std, target=target_var, sample=)
    errors = calculate_errors(preds, target_var, model_str="s2s2s")

    print(errors)


    # if how_well_do_we_do_on_train_data:
    #     train_preds = test(model, train_dl)
    #     train_errors = calculate_errors(train_preds, target_var, model_str="s2s2s")

    if filepath is not None:
        logger.info(f"Saving horizon_losses.png to {filepath}")
        plot_horizon_losses(filepath, error=errors["rmse"], identifier="rmse")
        plot_horizon_losses(filepath, error=errors["pearson-r"], identifier="pearson-r")

        logger.info(f"Saving *_demo_timeseries.png.png to {filepath}")
        plot_timeseries_over_horizon(filepath=filepath, preds=preds)

        logger.info(f"Saving errors.nc to {filepath}")
        errors.to_netcdf(filepath / "errors.nc")

        logger.info(f"Saving preds.nc to {filepath}")
        preds.to_netcdf(filepath / "preds.nc")

    # TODO: create a summary table and save to .tex file ?
    """

    return 1
