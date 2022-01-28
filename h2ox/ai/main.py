from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from h2ox.ai.dataset import FcastDataset
from h2ox.ai.experiment import ex
from h2ox.ai.model import initialise_model
from h2ox.ai.scripts.utils import load_zscore_data
from h2ox.ai.train import initialise_training, train, train_validation_split


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
    print("I am on line36 in main.py")
    data_dir = Path(Path.cwd() / "data")
    target, history, forecast = load_zscore_data(data_dir)
    history = history.merge(target)

    # select site
    site_target = target.sel(location=[site])
    site_history = history.sel(location=[site])
    site_forecast = forecast.sel(location=[site])

    # train-test split
    train_forecast = site_forecast.sel(
        initialisation_time=slice(train_start_date, train_end_date)
    )
    
    dd = FcastDataset(
        target=site_target,  # target,
        history=site_history,  # history,
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

    return 1
