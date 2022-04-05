from collections import defaultdict
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import xarray as xr
from sacred import Experiment
from torch import nn
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


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")


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
    denorm: Optional[dict] = None,
    denorm_var: Optional[str] = None,
    epochs: int = 5,
    epochs_loss_cliff: int = 5,
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
    for epoch in range(epochs):
        #  train the model (turn dropout on ...)
        model.train()

        epoch_losses = []
        pbar = tqdm(train_dl, f"Epoch {epoch + 1}")
        for ii, data in enumerate(pbar):
            # move onto GPU (if exists)
            # for key in [k for k in data.keys() if k != "meta"]:
            #    data[key] = data[key].to(device)
            optimizer.zero_grad()

            # X = {kk:vv.to(device) for kk,vv in data.items() if kk in ['x_d','x_f','x_ff']}
            # X = {kk:torch.zeros(vv.shape).to(device) for kk,vv in data.items() if kk in ['x_d','x_f','x_ff']}

            data = move_to(data, device)

            data["y"] = data["y"].squeeze().to(device)
            # Y = torch.zeros(data['y'].shape).to(device)
            # yhat = model.forward_deterministic(X).to(device)

            # loss = loss_fn(yhat.squeeze(), data['y'].squeeze())

            (
                yhat,
                loss,
                likelihood_loss,
                complexity_loss,
            ) = model.sample_elbo_detailed_loss(
                inputs=data,
                labels=data["y"],
                criterion=loss_fn,
                sample_nbr=3,
                complexity_cost_weight=1.0 / data["y"].shape[0] / 500,
            )

            if ii == 0:
                print(
                    "loss:",
                    "combined",
                    loss.detach().cpu().numpy(),
                    "likelihood",
                    likelihood_loss.detach().cpu().numpy(),
                    "complexity",
                    complexity_loss,
                )  # .detach().cpu().numpy())
                print(
                    "Y",
                    "max",
                    torch.amax(data["y"], dim=(0, 1)).cpu().numpy(),
                    "min",
                    torch.amin(data["y"], dim=(0, 1)).cpu().numpy(),
                    "mean",
                    torch.mean(data["y"], dim=(0, 1)).cpu().numpy(),
                )
                print(
                    "yhat",
                    "max",
                    yhat[0, ...].max(axis=(0, 1)),
                    "min",
                    yhat[0, ...].min(axis=(0, 1)),
                    "mean",
                    yhat[0, ...].mean(axis=(0, 1)),
                )

                # for name, p in model.named_parameters():
                #    if 'head' in name or 'pre' in name:
                #        print (name, p.requires_grad, p.data.mean())

            loss.backward()
            optimizer.step()

            #  return info to user
            learning_rate = optimizer.param_groups[0]["lr"]

            loss_float = float(loss.detach().cpu().numpy())

            epoch_losses.append(loss_float)
            epoch_loss = np.mean(epoch_losses)

            pbar.set_postfix_str(f"Loss: {epoch_loss:.5f}  Lr: {learning_rate:.4f}")

        epoch_loss = np.mean(epoch_losses)
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        ex.log_scalar("Loss/train", epoch_loss, epoch)

        # Scheduler for reducing the learning rate loss
        if scheduler is not None:
            scheduler.step()

        all_losses.append(epoch_loss)

        if epoch % validate_every_n == 0:

            if val_dl is not None:
                val_loss = validate(
                    model,
                    log_every_n_steps,
                    val_dl,
                    loss_fn,
                    epoch,
                    epochs_loss_cliff,
                    denorm,
                    denorm_var,
                    writer,
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


def mc_sample(model, X, N_samples=100, ci_std=2):
    preds = [model(X) for _ in range(N_samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)

    # maybe need to go to and back from levels vs. deltas

    ci_upper = means + (ci_std * stds)
    ci_lower = means - (ci_std * stds)
    return means, stds, ci_upper, ci_lower


def mc_sample_paths(model, X, N_samples=100, ci_std=2):
    def reverse_cumsum(arr, dim):
        shp = list(arr.shape)
        shp[dim] = 1
        return torch.diff(arr, dim=dim, prepend=torch.zeros(*list(shp)).to(arr.device))

    preds = [model(X) for _ in range(N_samples)]

    preds = torch.stack(preds)

    paths = preds.cumsum(dim=2)

    means = paths.mean(axis=0)
    path_stds = paths.std(axis=0)

    # maybe need to go to and back from levels vs. deltas

    ci_upper = means + (ci_std * path_stds)
    ci_lower = means - (ci_std * path_stds)

    means = reverse_cumsum(means, dim=1)
    ci_upper = reverse_cumsum(ci_upper, dim=1)
    ci_lower = reverse_cumsum(ci_lower, dim=1)

    return means, path_stds, ci_upper, ci_lower


def validate(
    model: nn.Module,
    log_every_n_steps: int,
    validation_dl: DataLoader,
    loss_fn: nn.Module,
    epoch: int,
    epochs_loss_cliff: int,
    denorm: Optional[dict],
    denorm_var: Optional[str],
    writer: SummaryWriter,
) -> float:
    # move onto GPU (if exists)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if isinstance(validation_dl.dataset, Subset):
        meta_lookup = validation_dl.dataset.dataset.sample_lookup
        site_keys = validation_dl.dataset.dataset.site_keys
    else:
        meta_lookup = validation_dl.dataset.sample_lookup
        site_keys = validation_dl.dataset.site_keys

    model.eval()
    pbar = tqdm(validation_dl, "Validation")

    losses = []
    eval_data = defaultdict(list)
    for data in pbar:
        # move to device
        data = move_to(data, device)

        # X = {kk:vv.to(device) for kk,vv in data.items() if kk in ['x_d','x_f','x_ff']}
        data["y"] = data["y"].squeeze().to(device)

        # forward pass - deterministic
        yhat = model.forward(data).to(device)
        y = data["y"]

        # calculate loss
        if "weighted_mse_loss" in loss_fn.__repr__():
            wt = get_exponential_weights(
                horizon=model.target_horizon,
                clip=min((epoch + 1) / epochs_loss_cliff, 1),
            ).to(device)
            wt = torch.stack([wt] * y.squeeze().shape[-1], dim=1)
            loss = loss_fn(yhat.squeeze(), y.squeeze(), wt)
        else:
            loss = loss_fn(yhat.squeeze(), y.squeeze())

        eval_meta = _process_metadata(data, meta_lookup)

        # save the predictions and the observations
        obs = y.squeeze().detach().cpu().numpy()
        sim = yhat.squeeze().detach().cpu().numpy()

        # Create a dictionary of the results
        eval_data["obs"].append(obs)
        eval_data["sim-frozen"].append(sim)

        means, stds, ci_upper, ci_lower = mc_sample(model, data, N_samples=50)

        eval_data["sim-mean"].append(means.squeeze().detach().cpu().numpy())
        eval_data["sim-std"].append(stds.squeeze().detach().cpu().numpy())
        eval_data["ci-95+"].append(ci_upper.squeeze().detach().cpu().numpy())
        eval_data["ci-95-"].append(ci_lower.squeeze().detach().cpu().numpy())

        for kk, vv in eval_meta.items():
            eval_data[kk].append(vv)

        losses.append(loss.detach().cpu().numpy())

    ds = _eval_data_to_ds(
        eval_data,
        data_keys=["obs", "sim-frozen", "sim-mean", "sim-std", "ci-95+", "ci-95-"],
        assign_sample=False,
        denorm=denorm,
        denorm_var=denorm_var,
        site_keys=site_keys,
    )

    ds_errors = calculate_errors(
        ds, obs_var="obs", sim_var="sim-frozen", site_dim="site"
    )

    target_horizon = ds_errors["step"].shape[0]

    std_dict = dict(
        zip(
            [str(ii) for ii in range(1, target_horizon, log_every_n_steps)],
            ds["sim-std"]
            .mean(dim=("site", "date"))
            .sel({"step": range(1, target_horizon, log_every_n_steps)})
            .to_dict()["data"],
        )
    )

    for kk, vv in std_dict.items():
        ex.log_scalar(f"Bayesian/std/{kk}", np.array(vv), epoch)

    writer.add_scalars(
        "Bayesian/std",
        std_dict,
        epoch,
    )

    scalar_dict = (
        ds_errors.mean(dim="site")
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


def test(
    model: nn.Module,
    test_dl: DataLoader,
    denorm: Optional[dict] = None,
    denorm_var: Optional[str] = None,
) -> xr.Dataset:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if isinstance(test_dl.dataset, Subset):
        meta_lookup = test_dl.dataset.dataset.sample_lookup
        site_keys = test_dl.dataset.dataset.site_keys
    else:
        meta_lookup = test_dl.dataset.sample_lookup
        site_keys = test_dl.dataset.site_keys

    eval_data = defaultdict(list)
    for data in tqdm(test_dl, "Running Evaluation"):
        # move to device
        # X = {kk:vv.to(device) for kk,vv in data.items() if kk in ['x_d','x_f','x_ff']}
        # X = {kk:torch.zeros(vv.shape).to(device) for kk,vv in data.items() if kk in ['x_d','x_f','x_ff']}

        data = move_to(data, device)

        y = data["y"].to(device)

        # forward pass - deterministic
        yhat = model.forward(data).to(device)

        eval_meta = _process_metadata(data, meta_lookup)

        # save the predictions and the observations
        obs = y.squeeze().detach().cpu().numpy()
        sim = yhat.squeeze().detach().cpu().numpy()

        # Create a dictionary of the results
        eval_data["obs"].append(obs)
        eval_data["sim-frozen"].append(sim)

        means, stds, ci_upper, ci_lower = mc_sample_paths(model, data, N_samples=50)

        eval_data["sim-mean"].append(means.squeeze().detach().cpu().numpy())
        eval_data["sim-std"].append(stds.squeeze().detach().cpu().numpy())
        eval_data["ci-95+"].append(ci_upper.squeeze().detach().cpu().numpy())
        eval_data["ci-95-"].append(ci_lower.squeeze().detach().cpu().numpy())

        for kk, vv in eval_meta.items():
            eval_data[kk].append(vv)

    ds = _eval_data_to_ds(
        eval_data,
        data_keys=["obs", "sim-frozen", "sim-mean", "sim-std", "ci-95+", "ci-95-"],
        assign_sample=False,
        denorm=denorm,
        denorm_var=denorm_var,
        site_keys=site_keys,
    )

    return ds
