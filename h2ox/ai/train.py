import socket
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
import xarray as xr
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from loguru import logger

from h2ox.ai.dataset import FcastDataset
from h2ox.ai.train_utils import get_exponential_weights
from sacred import Experiment


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
    experiment: Experiment,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
):
    run_dir = (
        Path(experiment.observers[0].dir)
        if experiment.observers[0].dir is not None
        else None
    )

    if run_dir is not None:
        logger.info(f"Saving model_epoch{epoch:03d}.pt to {run_dir.as_posix()}")
        weight_path = run_dir / f"model_epoch{epoch:03d}.pt"
        torch.save(model.state_dict(), str(weight_path))

        logger.info(
            f"Saving optimizer_state_epoch{epoch:03d}.pt to {run_dir.as_posix()}"
        )
        optimizer_path = run_dir / f"optimizer_state_epoch{epoch:03d}.pt"
        torch.save(optimizer.state_dict(), str(optimizer_path))
    else:
        logger.info(
            f"No run_dir found in experiment observers. Not saving model or optimizer state."
        )


def train(
    model: nn.Module,
    train_dl: DataLoader,
    optimizer: Any,
    loss_fn: nn.Module,
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

                _save_weights_and_optimizer(
                    epoch=epoch,
                    experiment=experiment,
                    model=model,
                    optimizer=optimizer,
                )

            #  return info to user
            learning_rate = optimizer.param_groups[0]["lr"]

            loss_float = float(loss.detach().cpu().numpy())
            epoch_losses.append(loss_float)
            epoch_loss = np.mean(epoch_losses)

            pbar.set_postfix_str(
                f"Loss: {epoch_loss:.2f}  Lr: {learning_rate:.4f}  nans:  {count_nans}"
            )

        # Scheduler for reducing the learning rate loss
        if scheduler is not None:
            scheduler.step()

        all_losses.append(epoch_loss)
        if epoch % validate_every_n == 0:
            # print(f"Current Losses: {all_losses}")
            if val_dl is not None:
                val_loss = validate(model, val_dl, loss_fn)
                print(f"-- Validation Loss: {val_loss:.3f} --")
                all_val_losses.append(val_loss)

        if cache_model:
            # Save model checkpoint
            # save optimizer checkpoint
            pass

    return (all_losses, all_val_losses)


def validate(model: nn.Module, validation_dl: DataLoader, loss_fn: nn.Module) -> float:
    # move onto GPU (if exists)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    pbar = tqdm(validation_dl, "Validation")

    losses = []
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

        losses.append(loss.detach().cpu().numpy())
    valid_loss = np.mean(losses)

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

    print("Converting to xarray object")
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
    eval_data: DefaultDict[str, List[np.ndarray]], assign_sample: bool = False
) -> xr.Dataset:
    # get correct shapes for arrays as output
    obs = np.concatenate(eval_data["obs"], axis=0)
    sim = np.concatenate(eval_data["sim"], axis=0)
    sample = np.concatenate(eval_data["sample"], axis=0)
    time = np.concatenate(eval_data["time"], axis=0)
    init_time = np.concatenate(eval_data["init_time"], axis=0)
    coords = {
        "initialisation_time": init_time,
        "horizon": np.arange(sim.shape[-1] if sim.ndim > 1 else 1),
    }
    ds = xr.Dataset(
        {
            "obs": (
                ("initialisation_time", "horizon"),
                obs if obs.ndim > 1 else obs[:, np.newaxis],
            ),
            "sim": (
                ("initialisation_time", "horizon"),
                sim if sim.ndim > 1 else sim[:, np.newaxis],
            ),
            "time": (("initialisation_time", "horizon"), time),
            "sample": (("initialisation_time"), sample),
        },
        coords=coords,
    )

    # assign "sample" as a dimension to the dataset
    if assign_sample:
        df = ds.to_dataframe().reset_index()
        ds_with_sample_dim = df.set_index(
            ["initialisation_time", "horizon", "sample"]
        ).to_xarray()

        return ds_with_sample_dim
    else:
        return ds


def train_validation_split(
    train_dataset: FcastDataset,
    random_val_split: bool,
    validation_proportion: float = 0.8,
) -> Tuple[FcastDataset, FcastDataset]:
    train_size = int(validation_proportion * len(train_dataset))
    validation_size = len(train_dataset) - train_size
    if random_val_split:
        train_dd, validation_dd = torch.utils.data.random_split(
            train_dataset, [train_size, validation_size]
        )
    else:
        # SEQUENTIAL
        # train from 1:N; validation from N:-1
        train_dd = Subset(train_dataset, np.arange(train_size))
        validation_dd = Subset(train_dataset, np.arange(train_size, len(train_dataset)))

    return train_dd, validation_dd


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt

    from h2ox.ai.data_utils import calculate_errors
    from h2ox.ai.dataset import FcastDataset
    from h2ox.ai.model import initialise_model
    from h2ox.ai.scripts.utils import load_zscore_data

    # parameters for the yaml file
    ENCODE_DOY = True
    SEQ_LEN = 60
    FUTURE_HORIZON = 76
    SITE = "kabini"
    TARGET_VAR = "volume_bcm"
    HISTORY_VARIABLES = ["tp", "t2m"]
    FORECAST_VARIABLES = ["tp", "t2m"]
    BATCH_SIZE = 32
    TRAIN_END_DATE = "2018-12-31"
    TRAIN_START_DATE = "2010-01-01"
    TEST_START_DATE = "2019-01-01"
    TEST_END_DATE = "2022-01-01"
    HIDDEN_SIZE = 64
    NUM_LAYERS = 1
    DROPOUT = 0.4
    NUM_WORKERS = 4
    N_EPOCHS = 30
    RANDOM_VAL_SPLIT = False
    EVAL_TEST = True

    if socket.gethostname() == "Tommy-Lees-MacBook-Air.local":
        # if on tommy laptop then only running tests
        TRAIN_END_DATE = "2011-01-01"
        TRAIN_START_DATE = "2010-01-01"
        EVAL_TEST = False
        N_EPOCHS = 10
        NUM_WORKERS = 1

    # load data
    data_dir = Path(Path.cwd() / "data")
    target, history, forecast = load_zscore_data(data_dir)

    # # select site
    site_target = target.sel(location=[SITE])
    site_history = history.sel(location=[SITE])
    site_forecast = forecast.sel(location=[SITE])

    # get train data
    # train_target = site_target.sel(time=slice(TRAIN_START_DATE, TRAIN_END_DATE))
    train_history = site_history.sel(time=slice(TRAIN_START_DATE, TRAIN_END_DATE))
    train_forecast = site_forecast.sel(
        initialisation_time=slice(TRAIN_START_DATE, TRAIN_END_DATE)
    )

    # normalize data
    # norm_target, (mean_target, std_target) = normalize_data(site_target)
    # norm_history, (mean_history, std_history) = normalize_data(site_history)
    # norm_train_forecast, (mean_forecast, std_forecast) = normalize_data(train_forecast, time_dim="initialisation_time")

    # load dataset
    dd = FcastDataset(
        target=site_target,  # target,
        history=train_history,  # history,
        forecast=None,  # forecast,
        encode_doy=ENCODE_DOY,
        historical_seq_len=SEQ_LEN,
        future_horizon=FUTURE_HORIZON,
        target_var=TARGET_VAR,
        mode="train",
        history_variables=HISTORY_VARIABLES,
        forecast_variables=FORECAST_VARIABLES,
    )

    # train-validation split
    train_dd, validation_dd = train_validation_split(
        dd, random_val_split=RANDOM_VAL_SPLIT, validation_proportion=0.8
    )

    train_dl = DataLoader(
        train_dd, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    val_dl = DataLoader(
        validation_dd, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # initialise model
    model = initialise_model(
        dd, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT
    )

    # # train
    # TODO: how to config the loss_fn // optimizer etc. ?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer, scheduler, loss_fn = initialise_training(
        model, device=device, loss_rate=1e-3
    )

    losses, _ = train(
        model,
        train_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=N_EPOCHS,
        val_dl=val_dl,
    )
    # plt.plot(losses)

    # # test
    if EVAL_TEST:
        # get test data
        # test_target = site_target.sel(time=slice(TEST_START_DATE, TEST_END_DATE))
        # test_history = site_history.sel(time=slice(TEST_START_DATE, TEST_END_DATE))
        test_forecast = site_forecast.sel(
            initialisation_time=slice(TEST_START_DATE, TEST_END_DATE)
        )
        # norm_test_forecast = (test_forecast - mean_forecast) / std_forecast

        # load dataset
        test_dd = FcastDataset(
            target=site_target,  # target,
            history=site_history,  # history,
            forecast=test_forecast,  # forecast,
            encode_doy=ENCODE_DOY,
            historical_seq_len=SEQ_LEN,
            future_horizon=FUTURE_HORIZON,
            target_var=TARGET_VAR,
            mode="test",
            history_variables=HISTORY_VARIABLES,
            forecast_variables=FORECAST_VARIABLES,
        )

        test_dl = DataLoader(
            test_dd, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

    else:
        test_dl = val_dl

    preds = test(model, test_dl)

    # unnormalize preds
    # preds = unnormalize_preds(preds, mean_target, std_target, target=TARGET_VAR, sample=SITE)

    errors = calculate_errors(preds, TARGET_VAR, model_str="s2s2s")
    print(errors["rmse"])
    print(errors["pearson-r"])

    # make the timeseries plots
    # f, axs = plt.subplots(3, 4, figsize=(6*4, 2*3), tight_layout=True, sharey=True, sharex=True)
    # random_times = np.random.choice(preds["initialisation_time"].values, size=12, replace=False)

    # for ix, time in enumerate(random_times):
    #     ax = axs[np.unravel_index(ix, (3, 4))]
    #     ax.plot(preds.sel(initialisation_time=time)["obs"], label="obs")
    #     ax.plot(preds.sel(initialisation_time=time)["sim"], label="sim")
    #     ax.set_title(time)

    # # make the forecast horizon plot
    f, ax = plt.subplots(figsize=(12, 6))
    errors.squeeze()["rmse"].plot(ax=ax)
