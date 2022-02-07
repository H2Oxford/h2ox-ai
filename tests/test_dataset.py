from pathlib import Path

from h2ox.ai.dataset.dataset import FcastDataset
from h2ox.ai.dataset.utils import load_zscore_data


if __name__ == "__main__":

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
    TRAIN_END_DATE = "2012-01-01"
    TRAIN_START_DATE = "2011-01-01"
    HIDDEN_SIZE = 64
    NUM_LAYERS = 1
    DROPOUT = 0.4
    NUM_WORKERS = 1
    N_EPOCHS = 10
    RANDOM_VAL_SPLIT = True

    # load data
    data_dir = Path(Path.cwd() / "data")
    target, history, forecast = load_zscore_data(data_dir)

    # create future data (x_ff)
    # date and location columns
    # min_date = target["time"].min().values
    # max_date = target["time"].max().values + (FUTURE_HORIZON * pd.Timedelta("1D"))
    # future_date_index = pd.date_range(min_date, max_date, freq="D")
    # future = pd.concat(
    #     [
    #         pd.DataFrame({"time": future_date_index, "location": loc})
    #         for loc in target.location.values
    #     ]
    # ).set_index("time")

    # feature engineering
    # doys_sin, doys_cos = encode_doys(future.index.dayofyear)
    # future["doy_sin"] = doys_sin[0]
    # future["doy_cos"] = doys_cos[0]

    # # select site
    site_target = target.sel(location=[SITE])
    site_history = history.sel(location=[SITE])
    site_forecast = forecast.sel(location=[SITE])

    # get train data
    train_target = site_target.sel(time=slice(TRAIN_START_DATE, TRAIN_END_DATE))
    train_history = site_history.sel(time=slice(TRAIN_START_DATE, TRAIN_END_DATE))
    train_forecast = site_forecast.sel(
        initialisation_time=slice(TRAIN_START_DATE, TRAIN_END_DATE)
    )

    # load dataset
    dd = FcastDataset(
        target=train_target,  # target,
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

    print([(k, dd[0][k].shape) for k in dd[0].keys() if not isinstance(dd[0][k], dict)])

    # load dataloader
    # get individual/batched samples

    final_t = dd[0]["x_d"][-1, :]
    first_t = dd[0]["y"][0, :]
    location = dd.get_meta(0)[0]
    time = dd.get_meta(0)[1]
    # check numbers match
    print("History")
    print(final_t[:2])
    print(history.sel(location=location, time=time).to_array().values)
    print()
    print("Target")
    print(first_t)
    print(target.sel(location=location, time=time).to_array().values)

    data_y = site_target
    target_horizon_td = pd.Timedelta(f"{dd.target_horizon}D")
    dd._get_target_data(data_y, forecast_init_time=time, horizon_td=target_horizon_td)
    forecast.sel(location=location, initialisation_time=time)