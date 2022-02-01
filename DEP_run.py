from pathlib import Path
from typing import List, Dict, Any, Callable

import torch
from sacred import Experiment
from torch.utils.data import DataLoader

from h2ox.ai.dataset import FcastDataset
from h2ox.ai.model import initialise_model
from h2ox.ai.train import initialise_training, train, train_validation_split
from h2ox.ai.scripts.utils import load_zscore_data, load_samantha_updated_data
from h2ox.ai.data_utils import normalize_data
from h2ox.ai.main import _main

# instantiate the Experiment class
ex = Experiment("fcast", interactive=True)

# TODO: normalize/unnormalize data
# TODO: save experiments in reproducible way
# TODO: assert that experiment settings make sense (i.e. variables can be found etc.) def pre_experiment_checks
# TODO: ensuring config file sets up your data
# TODO: specify the reservoirs (wrap in a loop for all reservoirs)
# TODO: uncertainty [stretch goal]
# TODO: build test dataloaders etc.


@ex.main
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
    sites: List[str],
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    num_workers: int,
    random_val_split: bool,
    eval_test: bool,
    n_epochs: int,
):
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
    )
    return out
    


def get_correct_keys(conf: Dict[str, Any], func: Callable) -> Dict[str, Any]:
    varnames = func.__code__.co_varnames
    return {k: conf[k] for k in varnames if k in conf.keys()}


if __name__ == "__main__":
    # parameters from the yaml file
    # ex.add_config("conf.yaml")
    ex.add_config("tests/test.yaml")

    # get the correct keys from the config file to pass to main()
    config_obj = ex.configurations[0]._conf
    conf = get_correct_keys(config_obj, main)

    # assert False
    # ex.run_commandline()
    main(**conf)
