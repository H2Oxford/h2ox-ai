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

    [meta_lookup[idx] for idx in idxs]

    forecast_init_times = [meta_lookup[idx]["date"] for idx in idxs]
    # NOTE: these times need cleaning (conversion errors @ minute resolution)...
    target_times = np.array(
        data["meta"]["target_times"].detach().cpu().numpy().astype("datetime64[ns]"),
        dtype="datetime64[m]",
    )

    eval_metadata = {"init_time": forecast_init_times, "time": target_times}

    if "site" in data["meta"].keys():
        eval_metadata["site"] = [meta_lookup[idx]["site"] for idx in idxs]

    return eval_metadata


def _eval_data_to_ds(
    eval_data: DefaultDict[str, List[np.ndarray]],
    site_keys: Optional[List[str]] = None,
    assign_sample: bool = False,
    time_dim: str = "date",
    horizon_dim: str = "step",
    site_dim: str = "site",
) -> xr.Dataset:
    # get correct shapes for arrays as output
    obs = np.concatenate(eval_data["obs"], axis=0)
    sim = np.concatenate(eval_data["sim"], axis=0)
    time = np.concatenate(eval_data["time"], axis=0)
    init_time = np.concatenate(eval_data["init_time"], axis=0)

    if site_dim in eval_data.keys():
        # is one-hot-encoded, deshape:
        sites = np.concatenate(eval_data[site_dim], axis=0)

        obs = np.stack([obs[sites == site, :] for site in np.unique(sites)], axis=-1)
        sim = np.stack([sim[sites == site, :] for site in np.unique(sites)], axis=-1)
        time = np.stack([time[sites == site, :] for site in np.unique(sites)], axis=-1)
        init_time = np.stack(
            [init_time[sites == site] for site in np.unique(sites)], axis=-1
        )

        # assert (np.diff(init_time,axis=1)==timedelta(days=0)).all(), ""
        time = time[:, :, 0]
        init_time = init_time[:, 0]

    # recover a 4d array: time x site x horizon x variable
    if site_keys is None:
        site_keys = np.arange(obs.shape[-1])

    coords = {
        time_dim: init_time,
        horizon_dim: np.arange(sim.shape[1]),
        "site": site_keys,
    }
    ds = xr.Dataset(
        {
            "obs": (
                (time_dim, horizon_dim, "site"),
                obs,
            ),
            "sim": (
                (time_dim, horizon_dim, "site"),
                sim,
            ),
            "valid_time": ((time_dim, horizon_dim), time),
        },
        coords=coords,
    )

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
