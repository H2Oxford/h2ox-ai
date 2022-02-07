"""[summary]

TODO:
    - run experiments to check model performances
    - reservoirs at the same endpoint, how to implement this?
    - app.py (vertex ai implementation to serve the model)
Returns:
    [type]: [description]
"""
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from loguru import logger
import xarray as xr

from h2ox.ai.dataset.dataset import FcastDataset
from h2ox.ai.experiment import ex
from h2ox.ai.model import initialise_model
from h2ox.ai.dataset.utils import load_zscore_data
from h2ox.ai.train import initialise_training, train, train_validation_split, test
from h2ox.ai.dataset.utils import normalize_data, unnormalize_preds
from h2ox.ai.experiment_utils import (
    plot_losses,
    plot_horizon_losses,
    plot_timeseries_over_horizon,
)
from h2ox.ai.dataset.utils import calculate_errors


def _main(
    seq_len: int = 60,
    future_horizon: int = 76,
    target_var: str = "volume_bcm",
    train_end_date: str = "2018-12-31",
    train_start_date: str = "2010-01-01",
    test_start_date: str = "2019-01-01",
    test_end_date: str = "2022-01-01",
    history_variables: List[str] = ["tp", "t2m"],
    forecast_variables: List[str] = ["tp", "t2m"],
    encode_doy: bool = True,
    sites: List[str] = ["kabini"],
    batch_size: int = 32,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.4,
    num_workers: int = 1,
    random_val_split: bool = True,
    eval_test: bool = True,
    n_epochs: int = 30,
    normalize: bool = False,
    validate_every_n: int = 3,
    include_ohe: bool = True,
) -> int:
    # load data
    data_dir = Path.cwd() / "data"
    target, history, forecast = load_zscore_data(data_dir)
    history = history.merge(target)
    # sam_data = load_samantha_updated_data(data_dir)
    # target = sam_data[[target_var]]
    # history = sam_data
    # forecast = None

    # train-test split
    # normalize data
    if normalize:
        train_history, (history_mn, history_std) = normalize_data(
            history.sel(time=slice(train_start_date, train_end_date)),
            static_or_global=True,
        )
        train_target, (target_mn, target_std) = normalize_data(
            target.sel(time=slice(train_start_date, None)), static_or_global=True
        )
        train_forecast, (forecast_mn, forecast_std) = normalize_data(
            forecast.sel(initialisation_time=slice(train_start_date, None)),
            time_dim="initialisation_time",
            static_or_global=True,
        )
        test_history, _ = normalize_data(
            history.sel(time=slice(test_start_date, test_end_date)),
            mean_=history_mn,
            std_=history_std,
        )
        test_forecast, _ = normalize_data(
            forecast.sel(initialisation_time=slice(test_start_date, None)),
            mean_=target_mn,
            std_=target_std,
        )
        test_target, _ = normalize_data(
            target.sel(time=slice(test_start_date, None)),
            mean_=forecast_mn,
            std_=forecast_std,
        )
    else:
        train_history = history.sel(time=slice(train_start_date, train_end_date))
        train_forecast = forecast.sel(initialisation_time=slice(train_start_date, None))
        train_target = target.sel(time=slice(train_start_date, None))

        test_history = history.sel(time=slice(test_start_date, test_end_date))
        test_forecast = forecast.sel(initialisation_time=slice(test_start_date, None))
        test_target = target.sel(time=slice(test_start_date, None))

    dd = FcastDataset(
        target=train_target,  # target,
        history=train_history.sel(location=sites),  # history,
        forecast=train_forecast,  # forecast,
        encode_doy=encode_doy,
        historical_seq_len=seq_len,
        future_horizon=future_horizon,
        target_var=target_var,
        mode="train",
        history_variables=history_variables,
        forecast_variables=forecast_variables,
        include_ohe=include_ohe,
    )

    print(dd)

    # train-validation split
    train_dd, validation_dd = train_validation_split(
        dd, random_val_split=random_val_split, validation_proportion=0.8
    )

    train_dl = DataLoader(
        train_dd, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_dl = DataLoader(
        validation_dd, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # initialise model
    model = initialise_model(
        dd, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout
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
        epochs=n_epochs,
        val_dl=val_dl,
        validate_every_n=validate_every_n,
        experiment=ex,
    )

    # get filepath for experiment dir from the Sacred Experiment
    # TODO(tl): is ex a global variable?
    filepath = Path(ex.observers[0].dir) if ex.observers[0].dir is not None else None
    if filepath is not None:
        logger.info(f"Saving losses.png to {filepath}")
        plot_losses(filepath=filepath, losses=losses, val_losses=val_losses)

    # # test
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

    return 1


@ex.automain
def main(
    seq_len: int,
    future_horizon: int,
    target_var: str,
    train_end_date: str,
    train_start_date: str,
    test_start_date: str,
    test_end_date: str,
    history_variables: List[str],
    forecast_variables: List[str],
    encode_doy: bool,
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    num_workers: int,
    random_val_split: bool,
    eval_test: bool,
    n_epochs: int,
    sites: List[str] = ["kabini"],
    normalize: bool = False,
) -> int:

    out = _main(
        seq_len=seq_len,
        future_horizon=future_horizon,
        target_var=target_var,
        train_end_date=train_end_date,
        train_start_date=train_start_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        history_variables=history_variables,
        forecast_variables=forecast_variables,
        encode_doy=encode_doy,
        sites=sites,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_workers=num_workers,
        random_val_split=random_val_split,
        eval_test=eval_test,
        n_epochs=n_epochs,
        normalize=normalize,
    )
    return out
