from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import torch
import xarray as xr
from loguru import logger
from sacred import Experiment
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from h2ox.ai.dataset.dataset import FcastDataset
from h2ox.ai.dataset.utils import calculate_errors
from h2ox.ai.experiment import ex
from h2ox.ai.train_utils import get_exponential_weights


def weighted_mse_loss(
    input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return torch.sum(weight * (input - target) ** 2)


def initialise_training(
    model, device: str, loss_rate: float = 5e-2
) -> Tuple[Any, Any, Any]:
    # use ADAM optimizer
    optimizer = optim.Adam([pam for pam in model.parameters()], lr=loss_rate)  # 0.05

    # reduce loss rate every \step_size epochs by \gamma
    #  from initial \lr
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # use MSE Loss function
    loss_fn = nn.MSELoss().to(device)
    # loss_fn = nn.SmoothL1Loss().to(device)
    # loss_fn = weighted_mse_loss

    return optimizer, scheduler, loss_fn


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


@ex.capture
def train(
    model: nn.Module,
    train_dl: DataLoader,
    optimizer: Any,
    loss_fn: nn.Module,
    log_every_n_steps: int,
    checkpoint_every_n: int = 5,
    writer: Optional[SummaryWriter] = None,
    scheduler: Optional[Any] = None,
    epochs: int = 5,
    val_dl: Optional[DataLoader] = None,
    validate_every_n: int = 3,
    catch_nans: bool = False,
    cache_model: bool = False,
    experiment: Optional[Experiment] = None,
) -> Tuple[List[float], ...]:
    # TODO (tl): add early stopping
    # TODO (tl): save model checkpoints & optimizer checkpoints
    # move onto GPU (if exists)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_losses = []
    all_val_losses = []
    count_nans = 0
    for epoch in range(epochs):
        #  train the model (turn dropout on ...)
        model.train()

        epoch_losses = []
        pbar = tqdm(train_dl, f"Epoch {epoch + 1}")
        for data in pbar:
            # move onto GPU (if exists)
            for key in [k for k in data.keys() if k != "meta"]:
                data[key] = data[key].to(device)

            optimizer.zero_grad()

            # forward pass
            yhat = model(data).to(device)
            y = data["y"]

            # REMOVE NANS (by batch-axis)
            nans = (
                torch.isnan(yhat).any(axis=1).squeeze()
                | torch.isnan(y).any(axis=1).squeeze()
            )
            yhat, y = yhat[~nans], y[~nans]

            #  SKIP loss if nans
            if y.nelement() == 0:
                count_nans += 1
                if catch_nans:
                    raise NotImplementedError
            else:  #  calculate loss
                if "weighted_mse_loss" in loss_fn.__repr__():
                    wt = get_exponential_weights(horizon=model.target_horizon).to(
                        device
                    )
                    loss = loss_fn(yhat.squeeze(), y.squeeze(), wt)
                else:
                    loss = loss_fn(yhat.squeeze(), y.squeeze())
                if torch.isnan(loss):
                    raise NotImplementedError

                #  calculate gradients and change weights
                loss.backward()
                optimizer.step()

            #  return info to user
            learning_rate = optimizer.param_groups[0]["lr"]

            loss_float = float(loss.detach().cpu().numpy())

            epoch_losses.append(loss_float)
            epoch_loss = np.mean(epoch_losses)

            pbar.set_postfix_str(
                f"Loss: {epoch_loss:.2f}  Lr: {learning_rate:.4f}  nans:  {count_nans}"
            )

        writer.add_scalar("Loss/train", epoch_loss, epoch)
        ex.log_scalar("Loss/train", epoch_loss, epoch)

        # Scheduler for reducing the learning rate loss
        if scheduler is not None:
            scheduler.step()

        all_losses.append(epoch_loss)
        if epoch % validate_every_n == 0:
            # print(f"Current Losses: {all_losses}")
            if val_dl is not None:
                val_loss = validate(
                    model, log_every_n_steps, val_dl, loss_fn, epoch, writer
                )
                all_val_losses.append(val_loss)

        # checkpoint training or save final model
        if (epoch % checkpoint_every_n == 0) or (epoch == epochs - 1):
            _save_weights_and_optimizer(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
            )

    return (all_losses, all_val_losses)


def validate(
    model: nn.Module,
    log_every_n_steps: int,
    validation_dl: DataLoader,
    loss_fn: nn.Module,
    epoch: int,
    writer: SummaryWriter,
) -> float:
    # move onto GPU (if exists)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if isinstance(validation_dl.dataset, Subset):
        meta_lookup = validation_dl.dataset.dataset.sample_lookup
    else:
        meta_lookup = validation_dl.dataset.sample_lookup

    model.eval()
    pbar = tqdm(validation_dl, "Validation")

    losses = []
    eval_data = defaultdict(list)
    for data in pbar:
        # move to device
        for key in [k for k in data.keys() if k != "meta"]:
            data[key] = data[key].to(device)

        # forward pass
        yhat = model(data).to(device)
        y = data["y"]

        # calculate loss
        if "weighted_mse_loss" in loss_fn.__repr__():
            wt = get_exponential_weights(horizon=model.target_horizon).to(device)
            loss = loss_fn(yhat.squeeze(), y.squeeze(), wt)
        else:
            loss = loss_fn(yhat.squeeze(), y.squeeze())

        samples, forecast_init_times, target_times = _process_metadata(
            data, meta_lookup
        )

        # save the predictions and the observations
        obs = y.squeeze().detach().cpu().numpy()
        sim = yhat.squeeze().detach().cpu().numpy()

        # Create a dictionary of the results
        eval_data["obs"].append(obs)
        eval_data["sim"].append(sim)
        eval_data["sample"].append(samples)
        eval_data["time"].append(target_times)
        eval_data["init_time"].append(forecast_init_times)

        losses.append(loss.detach().cpu().numpy())

    ds = _eval_data_to_ds(eval_data, assign_sample=False)
    ds = calculate_errors(ds, var="y")

    target_horizon = ds["step"].shape[0]

    scalar_dict = (
        ds.mean(dim="sample")
        .sel({"step": range(1, target_horizon, log_every_n_steps)})
        .to_dict()
    )
    scalar_dict = {kk: vv["data"] for kk, vv in scalar_dict["data_vars"].items()}

    valid_loss = np.mean(losses)

    writer.add_scalar("Loss/val", valid_loss, epoch)
    ex.log_scalar("Loss/val", valid_loss, epoch)

    for kk, vv in scalar_dict.items():
        log_data = dict(
            zip(
                [str(ii) for ii in range(1, target_horizon, log_every_n_steps)],
                np.array(vv).squeeze(),
            )
        )

        for kk2, vv2 in log_data.items():
            ex.log_scalar(f"Metric/{kk}/{kk2}", vv2, epoch)

        writer.add_scalars(
            f"Metrics/{kk}",
            log_data,
            epoch,
        )

    return valid_loss


def test(model: nn.Module, test_dl: DataLoader) -> xr.Dataset:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if isinstance(test_dl.dataset, Subset):
        meta_lookup = test_dl.dataset.dataset.sample_lookup
    else:
        meta_lookup = test_dl.dataset.sample_lookup

    eval_data = defaultdict(list)
    for data in tqdm(test_dl, "Running Evaluation"):
        # move to device
        for key in [k for k in data.keys() if k != "meta"]:
            data[key] = data[key].to(device)

        # get the metadata
        samples, forecast_init_times, target_times = _process_metadata(
            data, meta_lookup
        )

        # save the predictions and the observations
        obs = data["y"].squeeze().detach().cpu().numpy()
        sim = model(data).squeeze().detach().cpu().numpy()

        # Create a dictionary of the results
        eval_data["obs"].append(obs)
        eval_data["sim"].append(sim)
        eval_data["sample"].append(samples)
        eval_data["time"].append(target_times)
        eval_data["init_time"].append(forecast_init_times)

    ds = _eval_data_to_ds(eval_data, assign_sample=False)
    return ds


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


def train_validation_test_split(
    train_dataset: FcastDataset,
    cfg: Dict[str, Any],
    time_dim: str = "date",
) -> Tuple[FcastDataset, ...]:
    """Create train, validation, test dataset objects as a subset of original

    Args:
        train_dataset (FcastDataset): [description]
        cfg (Dict[str, Any]): [description]
        time_dim (str, optional): [description]. Defaults to "date".

    Returns:
        Tuple[FcastDataset, ...]: train, validation, test datasets
    """
    train_start_date = cfg["train_start_date"]
    train_end_date = cfg["train_end_date"]
    val_start_date = cfg["val_start_date"]
    val_end_date = cfg["val_end_date"]
    test_start_date = cfg["test_start_date"]
    test_end_date = cfg["test_end_date"]

    # SEQUENTIAL = train from 1:N; validation from N:-1
    # (NOTE: INDEXED BY TIME NOT SPACE - first sort the index_df by time)
    # TODO: pass in date objects to slice the dataset appropriately
    index_df = train_dataset._get_meta_dataframe()
    index_df = index_df.sort_values(time_dim)
    # reindex by date so that can use pandas slicing functionality
    index_df = (
        index_df.reset_index().rename(columns={"index": "pt_index"}).set_index(time_dim)
    )

    train_indexes = index_df.loc[train_start_date:train_end_date]["pt_index"]
    val_indexes = index_df.loc[val_start_date:val_end_date]["pt_index"]
    test_indexes = index_df.loc[test_start_date:test_end_date]["pt_index"]

    assert not any(
        np.isin(train_indexes, test_indexes)
    ), "Leakage: train indexes in test data"
    assert not any(
        np.isin(val_indexes, test_indexes)
    ), "Leakage: validation indexes in test data"

    # TODO: for pretty printing etc. create a derived class from Subset
    #  with a custom __repr__ method from the original FcastDataset class
    train_dd = Subset(train_dataset, train_indexes)
    validation_dd = Subset(train_dataset, val_indexes)
    test_dd = Subset(train_dataset, test_indexes)

    return train_dd, validation_dd, test_dd
