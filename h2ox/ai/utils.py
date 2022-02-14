from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import torch
import xarray as xr
import yaml
from loguru import logger
from torch import nn


def _process_metadata(
    data: Dict[str, torch.Tensor], meta_lookup: Dict[int, Tuple[str, int]]
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    # Convert metadata (init_time, target_time, location) into lists
    idxs = data["meta"]["index"].detach().cpu().numpy().flatten()
    samples = [meta_lookup[idx][0] for idx in idxs]
    forecast_init_times = [meta_lookup[idx][1] for idx in idxs]
    # NOTE: these times need cleaning (conversion errors @ minute resolution)...
    target_times = np.array(
        data["meta"]["target_times"].detach().cpu().numpy().astype("datetime64[ns]"),
        dtype="datetime64[m]",
    )

    return samples, forecast_init_times, target_times


def _eval_data_to_ds(
    eval_data: DefaultDict[str, List[np.ndarray]],
    assign_sample: bool = False,
    time_dim: str = "date",
    horizon_dim: str = "step",
    sample_dim: str = "sample",
) -> xr.Dataset:
    # get correct shapes for arrays as output
    obs = np.concatenate(eval_data["obs"], axis=0)
    sim = np.concatenate(eval_data["sim"], axis=0)
    sample = np.concatenate(eval_data["sample"], axis=0)
    time = np.concatenate(eval_data["time"], axis=0)
    init_time = np.concatenate(eval_data["init_time"], axis=0)

    coords = {
        time_dim: init_time,
        horizon_dim: np.arange(1, sim.shape[-1] + 1 if sim.ndim > 1 else 1),
    }
    ds = xr.Dataset(
        {
            "obs": (
                (time_dim, horizon_dim),
                obs if obs.ndim > 1 else obs[:, np.newaxis],
            ),
            "sim": (
                (time_dim, horizon_dim),
                sim if sim.ndim > 1 else sim[:, np.newaxis],
            ),
            "valid_time": ((time_dim, horizon_dim), time),
            sample_dim: ((time_dim), sample),
        },
        coords=coords,
    )

    # assign `sample_dim` as a dimension to the dataset
    if assign_sample:
        df = ds.to_dataframe().reset_index()
        ds_with_sample_dim = df.set_index(
            [time_dim, horizon_dim, sample_dim]
        ).to_xarray()

        return ds_with_sample_dim
    else:
        return ds


def load_model_optimizer_from_checkpoint(
    run_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    weight_path: Optional[Path] = None,
    device: str = "cpu",
) -> nn.Module:
    if weight_path is None:
        assert (
            run_dir is not None
        ), "If weight_path not provided, run_dir must be provided"
        weight_path = [x for x in sorted(list(run_dir.glob("model_epoch*.pt")))][-1]

    epoch = weight_path.name[-6:-3]
    optimizer_path = run_dir / f"optimizer_state_epoch{epoch}.pt"

    logger.info(f"Loading model & optimizer from Epoch: {epoch}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    optimizer.load_state_dict(torch.load(str(optimizer_path), map_location=device))


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


def _save_weights_and_optimizer(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    save_path: str = "tmp",
):

    weight_path = Path(save_path) / f"model_epoch{epoch:03d}.pt"
    torch.save(model.state_dict(), weight_path)

    optimizer_path = Path(save_path) / f"optimizer_state_epoch{epoch:03d}.pt"
    torch.save(optimizer.state_dict(), str(optimizer_path))

    return weight_path, optimizer_path


def get_exponential_weights(horizon: int) -> torch.Tensor:
    # exponential weighting
    wt = np.exp(np.linspace(0, 5, horizon))[::-1]
    wt = wt / np.linalg.norm(wt)
    wt = torch.from_numpy(wt)
    return wt
