from typing import List, Dict, Any
from pathlib import Path
from sacred import Experiment
import torch 
from torch.utils.data import DataLoader
from h2ox.ai.train import train, test, train_validation_split, initialise_training
from h2ox.ai.dataset import FcastDataset
from h2ox.ai.model import initialise_model
from h2ox.scripts.utils import load_zscore_data
from h2ox.ai.experiment_utils import create_model_experiment_folder, dump_config
from definitions import ROOT_DIR


# instantiate the Experiment class
ex = Experiment("H2Ox", interactive=True)
# parameters from the yaml file
ex.add_config('conf.yaml')


# TODO: normalize/unnormalize data
# TODO: save experiments in reproducible way
# TODO: test dataloader 


def initialise_experiment(config: Dict[str, Any]) -> Path:
    if config["path_to_runs_folder"] is not None:
        run_dir = Path(config["path_to_runs_folder"])
    else:
        run_dir = ROOT_DIR / "runs"

    run_dir.mkdir(exist_ok=True)
    assert run_dir.exists(), f'Expect the given runs folder to exist [{run_dir}]'
    
    # create experiment folder
    experiment_name = config["name"]
    experiment_dir = create_model_experiment_folder(run_dir, experiment_name, add_datetime=False)

    # dump yaml 
    dump_config(config, experiment_dir / "config.yaml")
    
    # return path to experiment folder
    return experiment_dir


@ex.main
def main(
    _config,
):
    # initialise experiment
    experiment_dir = initialise_experiment(dict(_config))

    # load data
    data_dir = Path(ROOT_DIR / "data")
    target, history, forecast = load_zscore_data(data_dir)
    history = history.merge(target)

    # select site
    site_target = target.sel(location=[_config["site"]])
    site_history = history.sel(location=[_config["site"]])
    site_forecast = forecast.sel(location=[_config["site"]])

    # train-test split
    train_forecast = site_forecast.sel(
        initialisation_time=slice(_config["train_start_date"], _config["train_end_date"])
    )

    dd = FcastDataset(
        target=site_target,  # target,
        history=site_history,  # history,
        forecast=train_forecast,  # forecast,
        encode_doy=_config["encode_doy"],
        historical_seq_len=_config["seq_len"],
        future_horizon=_config["future_horizon"],
        target_var=_config["target_var"],
        mode="train",
        history_variables=_config["history_variables"],
        forecast_variables=_config["forecast_variables"],
        experiment_dir=experiment_dir,
    )

    # train-validation split
    train_dd, validation_dd = train_validation_split(dd, random_val_split=_config["random_val_split"], validation_proportion=0.8)

    train_dl = DataLoader(
        train_dd, batch_size=_config["batch_size"], shuffle=False, num_workers=_config["num_workers"]
    )
    val_dl = DataLoader(
        validation_dd, batch_size=_config["batch_size"], shuffle=False, num_workers=_config["num_workers"]
    )

    # initialise model
    model = initialise_model(
        train_dl, hidden_size=_config["hidden_size"], num_layers=_config["num_layers"], dropout=_config["dropout"]
    )

    # # train
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer, scheduler, loss_fn = initialise_training(model, device=device, loss_rate=1e-3)

    losses, val_losses = train(model, train_dl, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, epochs=_config["n_epochs"], val_dl=val_dl)
    # plt.plot(losses)




if __name__ == "__main__":    
    # ex.run_commandline()
    # main()
    ex.run()
