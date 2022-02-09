from pathlib import Path
import matplotlib.pyplot as plt
import socket
import torch
from torch.utils.data import DataLoader

from h2ox.ai.dataset.utils import calculate_errors
from h2ox.ai.dataset.dataset import FcastDataset
from h2ox.ai.model import initialise_model
from h2ox.ai.dataset.utils import load_zscore_data
from h2ox.ai.train import train, train_validation_test_split, initialise_training, test


if __name__ == "__main__":
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
    train_dd, validation_dd = train_validation_test_split(
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
