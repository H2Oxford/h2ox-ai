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

from h2ox.ai.dataset import FcastDataset
from h2ox.ai.experiment import ex
from h2ox.ai.model import initialise_model
from h2ox.ai.scripts.utils import load_zscore_data
from h2ox.ai.train import initialise_training, train, train_validation_split
from h2ox.ai.data_utils import normalize_data
from h2ox.ai.experiment_utils import plot_losses


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
    site: str = "kabini",
    batch_size: int = 32,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.4,
    num_workers: int = 1,
    random_val_split: bool = True,
    eval_test: bool = True,
    n_epochs: int = 30,
) -> int:
    # load data
    data_dir = Path.cwd() / "data"
    target, history, forecast = load_zscore_data(data_dir)
    history = history.merge(target)
    
    # sam_data = load_samantha_updated_data(data_dir)
    # target = sam_data[[target_var]]
    # history = sam_data
    # forecast = None

    # select site
    # site_target = target.sel(location=[site])
    # site_history = history.sel(location=[site])
    # site_forecast = forecast.sel(location=[site]) if forecast is not None else None

    # train-test split
    # normalize data
    train_history, (history_mn, history_std) = normalize_data(
        history.sel(time=slice(train_start_date, train_end_date)), static_or_global=True
    )
    train_target, (target_mn, target_std) = normalize_data(
        target.sel(time=slice(train_start_date, train_end_date)), static_or_global=True
    )
    train_forecast, (forecast_mn, forecast_std) = normalize_data(
        forecast.sel(initialisation_time=slice(train_start_date, train_end_date)),
        time_dim="initialisation_time",
        static_or_global=True,
    )

    dd = FcastDataset(
        target=train_target,  # target,
        history=train_history.sel(location=[site]),  # history,
        forecast=train_forecast,  # forecast,
        encode_doy=encode_doy,
        historical_seq_len=seq_len,
        future_horizon=future_horizon,
        target_var=target_var,
        mode="train",
        history_variables=history_variables,
        forecast_variables=forecast_variables,
    )

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
    )

    # get filepath for experiment dir 
    filepath = Path(ex.observers[0].dir)
    if filepath is not None:
        logger.info(f"Saving losses.png to {filepath}")
        plot_losses(filepath=filepath, losses=losses, val_losses=val_losses)
    
    # # test
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
    site: str,
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    num_workers: int,
    random_val_split: bool,
    eval_test: bool,
    n_epochs: int,
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
        site=site,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_workers=num_workers,
        random_val_split=random_val_split,
        eval_test=eval_test,
        n_epochs=n_epochs,
    )
    return out
