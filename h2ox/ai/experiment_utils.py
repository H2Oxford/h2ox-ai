from pathlib import Path
from datetime import datetime
import yaml
import matplotlib.pyplot as plt
import numpy as np

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
