from pathlib import Path
from datetime import datetime
import yaml
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr 


def create_model_experiment_folder(
    data_dir: Path, experiment_name: str, add_datetime: bool = False
) -> Path:
    if add_datetime:
        date_str = datetime.now().strftime("%Y%M%d_%H%m")
        experiment_name += "_" + date_str

    expt_dir = data_dir / experiment_name
    if not expt_dir.exists():
        expt_dir.mkdir()
    return expt_dir


def dump_config(config: dict, config_path: Path) -> None:
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def plot_losses(filepath: Path, losses: np.ndarray, val_losses: np.ndarray, val_every: int = 3):
    f, ax = plt.subplots(figsize=(12, 4))
    ax.plot(losses, label="Training")
    # plot validation losses
    X = np.arange(len(losses))[::val_every][:len(val_losses)]
    ax.plot(X, val_losses, label="Validation")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("Epoch")
    ax.set_xlabel("Loss")
    ax.legend()
    f.savefig(filepath / "losses.png")
    plt.close("all")


def plot_horizon_losses(filepath: Path, error: xr.DataArray):
    # make the forecast horizon plot
    f, ax = plt.subplots(figsize=(12, 6))
    for sample in error.sample.values:
        error.sel(sample=sample).plot(ax=ax, label=sample)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(error.name if error.name is not None else "Error")
    ax.set_xlabel("Horizon")
    ax.set_title("Performance over forecast horizon")
    ax.legend()

    f.savefig(filepath / "horizon_losses.png")
    plt.close("all")


def plot_timeseries_over_horizon(filepath: Path, preds: xr.Dataset):
    for sample in np.unique(preds.sample.values):
        # make the timeseries plots
        preds_ = preds.sel(initialisation_time=preds["sample"] == sample)
        f, axs = plt.subplots(3, 4, figsize=(6*4, 2*3), tight_layout=True, sharey=True, sharex=True)
        f.tight_layout(rect=[0, 0.03, 1, 0.95])

        random_times = np.random.choice(preds_["initialisation_time"].values, size=12, replace=False)

        for ix, time in enumerate(random_times):
            ax = axs[np.unravel_index(ix, (3, 4))]
            ax.plot(preds_.sel(initialisation_time=time)["obs"], label="obs")
            ax.plot(preds_.sel(initialisation_time=time)["sim"], label="sim")
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            ax.set_title(time)
        
        ax.legend()

        f.suptitle(f"{sample} Timeseries")
        f.savefig(filepath / f"{sample}_demo_timeseries.png")
        plt.close("all")