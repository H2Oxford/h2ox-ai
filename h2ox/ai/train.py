from collections import defaultdict
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import xarray as xr
from sacred import Experiment
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from h2ox.ai.dataset.utils import calculate_errors
from h2ox.ai.experiment import ex
from h2ox.ai.utils import (
    _eval_data_to_ds,
    _process_metadata,
    _save_weights_and_optimizer,
    get_exponential_weights,
)


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

            # calculate loss, maybe with weights
            if "weighted_mse_loss" in loss_fn.__repr__():
                wt = get_exponential_weights(horizon=model.target_horizon).to(device)
                loss = loss_fn(yhat.squeeze(), y.squeeze(), wt)
            else:
                loss = loss_fn(yhat.squeeze(), y.squeeze())

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
