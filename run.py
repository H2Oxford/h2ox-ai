from typing import List
from pathlib import Path
from sacred import Experiment
import torch 
from torch.utils.data import DataLoader
from h2ox.ai.train import train, test, train_validation_split, initialise_training
from h2ox.ai.dataset import FcastDataset
from h2ox.ai.model import initialise_model
from h2ox.scripts.utils import load_zscore_data
from definitions import ROOT_DIR


# instantiate the Experiment class
ex = Experiment("fcast", interactive=True)


# TODO: normalize/unnormalize data
# TODO: save experiments in reproducible way
# TODO: test dataloader 

def initialise_experiment():
    pass



@ex.main
def main(
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
    random_val_split: bool  = True,
    eval_test: bool = True,
    n_epochs: int = 30,
):
    # load data
    data_dir = Path(ROOT_DIR / "data")
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
    train_dd, validation_dd = train_validation_split(dd, random_val_split=random_val_split, validation_proportion=0.8)

    train_dl = DataLoader(
        train_dd, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_dl = DataLoader(
        validation_dd, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # initialise model
    model = initialise_model(
        train_dl, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout
    )

    # # train
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer, scheduler, loss_fn = initialise_training(model, device=device, loss_rate=1e-3)

    losses, val_losses = train(model, train_dl, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, epochs=n_epochs, val_dl=val_dl)
    # plt.plot(losses)




if __name__ == "__main__":
    # parameters from the yaml file
    ex.add_config('conf.yaml')
    
    ex.run_commandline()
    # main()
