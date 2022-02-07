from h2ox.ai.dataset.dataset import FcastDataset
from h2ox.ai.dataset.utils import load_zscore_data
from pathlib import Path
from torch.utils.data import DataLoader
import torch 
from h2ox.ai.model import S2S2SModel


if __name__ == "__main__":
    # load data
    # load dataset
    # load dataloader
    # initialise model
    # run model forward

    # parameters for the yaml file
    ENCODE_DOY = True
    SEQ_LEN = 60
    FUTURE_HORIZON = 76
    SITE = "kabini"
    TARGET_VAR = "volume_bcm"
    HISTORY_VARIABLES = ["tp", "t2m"]
    FORECAST_VARIABLES = ["tp", "t2m"]
    FUTURE_VARIABLES = []
    BATCH_SIZE = 32
    TRAIN_END_DATE = "2011-01-01"
    TRAIN_START_DATE = "2010-01-01"
    HIDDEN_SIZE = 64
    NUM_LAYERS = 1
    DROPOUT = 0.4
    NUM_WORKERS = 1
    N_EPOCHS = 10
    RANDOM_VAL_SPLIT = True

    # load data
    data_dir = Path(Path.cwd() / "data")
    target, history, forecast = load_zscore_data(data_dir)

    # get train data
    train_target = target.sel(time=slice(TRAIN_START_DATE, TRAIN_END_DATE))
    train_history = history.sel(time=slice(TRAIN_START_DATE, TRAIN_END_DATE))
    train_forecast = forecast.sel(
        initialisation_time=slice(TRAIN_START_DATE, TRAIN_END_DATE)
    )

    # # select site
    y = train_target.sel(location=[SITE])
    x_d = train_history.sel(location=[SITE])
    x_f = train_forecast.sel(location=[SITE])

    # load dataset
    dd = FcastDataset(
        target=y,  # target,
        history=x_d,  # history,
        forecast=None,  # forecast,
        encode_doy=ENCODE_DOY,
        historical_seq_len=SEQ_LEN,
        future_horizon=FUTURE_HORIZON,
        target_var=TARGET_VAR,
        history_variables=HISTORY_VARIABLES,
        forecast_variables=FORECAST_VARIABLES,
    )
    dl = DataLoader(dd, batch_size=BATCH_SIZE, shuffle=False)

    # initialise model shapes
    # initialise model shapes
    forecast_horizon = dd.forecast_horizon
    future_horizon = dd.future_horizon

    historical_input_size = dd.historical_input_size
    forecast_input_size = dd.forecast_input_size
    future_input_size = dd.future_input_size

    model = S2S2SModel(
        forecast_horizon=forecast_horizon,
        future_horizon=future_horizon,
        historical_input_size=historical_input_size,
        forecast_input_size=forecast_input_size,
        future_input_size=future_input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    data_example = dl.__iter__().__next__()
    yhat = model(data_example)
    # assert yhat.shape == (
    #     BATCH_SIZE,
    #     total_horizon,
    # ), f"Expected {(BATCH_SIZE, total_horizon)} Got: {yhat.shape}"

    # train -- validation split (for testing hyperparameters)
    train_size = int(0.8 * len(dd))
    validation_size = len(dd) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dd, [train_size, validation_size]
    )