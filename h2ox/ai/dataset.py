from typing import Dict, Union, List, Optional, DefaultDict, Tuple
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
from definitions import ROOT_DIR
from torch.utils.data import Dataset
from torch import Tensor
import torch
import xarray as xr
from h2ox.ai.data_utils import create_doy


# ASSUMES: "time" is the named dimension/index column
# ASSUMES: assign doy to all datasets
class FcastDataset(Dataset):
    def __init__(
        self,
        history: xr.Dataset,
        forecast: xr.Dataset,
        target: xr.Dataset,
        historical_seq_len: int = 60,
        future_horizon: int = 76,
        target_var: str = "PRESENT_STORAGE_TMC",
        history_variables: List[str] = ["t2m"],
        forecast_variables: List[str] = ["t2m"],
        encode_doy: bool = True,
        mode: str = "train",
        spatial_dim: str = "location",
        forecast_initialisation_dim: str = "initialisation_time",
        forecast_horizon_dim: str = "forecast_horizon",
        cache: bool = False,
        experiment_dir: Optional[Path] = None,
    ):
        # TODO: check that date column is saved as "time"
        # store data in memory
        # TODO: how do we do this with cacheing? what stage to cache?
        self.history = history
        self.forecast = forecast
        self.target = target
        # self.future = future

        # ATTRIBUTES
        self.mode = mode
        self.cache = cache
        if self.cache:
            assert experiment_dir is not None, "Must specify an experiment directory if cache is True"
        self.experiment_dir = experiment_dir
        # variables
        self.target_var = target_var
        self.history_variables = history_variables
        self.forecast_variables = forecast_variables
        self.future_variables = []
        # dimension names
        self.spatial_dim = spatial_dim
        self.forecast_initialisation_dim = forecast_initialisation_dim
        self.forecast_horizon_dim = forecast_horizon_dim
        # engineered features
        self.encode_doy = encode_doy
        # size of arrays
        self.seq_len = historical_seq_len
        self.future_horizon = future_horizon
        self.forecast_horizon = pd.Timedelta(
            forecast[forecast_horizon_dim].max().values
        ).days
        self.target_horizon = self.future_horizon + self.forecast_horizon
        self.times = np.ndarray([])

        # TODO: do we want to add engineered features in the torch.Dataset?
        # TODO: how shall we specify what data is used in the future dataframe?
        # add engineered features
        self.history_variables = (
            self.history_variables + ["doy_sin", "doy_cos"]
            if encode_doy
            else self.history_variables
        )

        self.forecast_variables = (
            self.forecast_variables + ["doy_sin", "doy_cos"]
            if encode_doy
            else self.forecast_variables
        )

        self.future_variables = (
            ["doy_sin", "doy_cos"] if encode_doy else self.future_variables
        )

        # turn the data into a dictionary for model input
        self.engineer_arrays()

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        total_str = f"FcastDataset\n"
        total_str += f"------------\n"
        total_str += f"\n"
        total_str += f"Size: {self.__len__()}\n"
        total_str += f"Dataset: {self.mode}\n"
        total_str += f"target: {self.target_var}\n"
        total_str += f"history_variables: {self.history_variables}\n"
        total_str += f"forecast_variables: {self.forecast_variables}\n"
        total_str += "future_variables: [doy_sin, doy_cos]\n"  # NOTE: hardcoded
        total_str += f"seq_len: {self.seq_len}D\n"
        total_str += f"future_horizon: {self.future_horizon}\n"
        total_str += f"forecast_horizon: {self.forecast_horizon}\n"
        total_str += f"target_horizon: {self.target_horizon}\n"

        # plot data shapes
        data_eg = self[1]
        total_str += f'x_d shape: {data_eg["x_d"].shape}\n'
        total_str += f'x_f shape: {data_eg["x_f"].shape}\n'
        total_str += f'x_ff shape: {data_eg["x_ff"].shape}\n'
        total_str += f'y shape:   {data_eg["y"].shape}\n'

        # time and space dimensions
        tmin = pd.Timestamp(
            self.forecast[self.forecast_initialisation_dim].min().values
        )
        tmax = pd.Timestamp(
            self.forecast[self.forecast_initialisation_dim].max().values
        )
        total_str += f"PERIOD: {tmin}: {tmax}\n"
        total_str += f"LOCATIONS: {self.forecast[self.spatial_dim].values}\n"

        return total_str


    def engineer_arrays(self):
        """Create an `all_data` attribute which stores all the data
            DefaultDict[int, Dict[str, np.ndarray]]
        Create a `sample_lookup` attribute which stores the
            location & initialisation_date info.
            Dict[int, Tuple[str, pd.Timestamp]]
        """
        # Timedelta objects describing horizons / seq_length
        seq_length_history_td = pd.Timedelta(f"{self.seq_len}D")
        # forecast_horizon_td = pd.Timedelta(f"{self.forecast_horizon}D")
        future_horizon_td = pd.Timedelta(f"{self.future_horizon}D")
        target_horizon_td = pd.Timedelta(f"{self.target_horizon}D")

        # initialise the dictionary storing all the data
        self.all_data: DefaultDict[int, Dict] = defaultdict(dict)
        self.sample_lookup: Dict[int, Tuple[str, pd.Timestamp]] = {}

        # initialise the loop for building the data arrays
        COUNTER = 0
        NAN_COUNTER = 0

        # TODO: is this definitely the time axis we want to loop through?
        # get the initialisation dates for looping through the data
        forecast_init_times = self.forecast[self.forecast_initialisation_dim].values

        # TODO: what happens if the timeseries is not complete? i.e. missing dates
        # TODO: what if all the spatial locations in a dataset are not there?
        for sample in self.forecast[self.spatial_dim].values:

            # get the data for the sample
            data_h = self.history.sel({self.spatial_dim: sample})
            data_f = self.forecast.sel({self.spatial_dim: sample})
            # TODO: how to include the future data?
            # data_ff = self.future.loc[np.isin(self.future[self.spatial_dim], sample)]
            data_y = self.target.sel({self.spatial_dim: sample})

            # create data samples for each initialisation_date
            # history = self.seq_len days before the forecast
            # target = forecast_horizon + future_horizon
            pbar = tqdm(forecast_init_times, desc=f"Building data for {sample} [{self.mode}]")
            for forecast_init_time in pbar:
                # init pbar
                str_time = np.datetime_as_string(forecast_init_time, unit="h")
                postfix_str = f"T: {str_time} -- nans: {NAN_COUNTER}"
                pbar.set_postfix_str(postfix_str)

                # GET HISTORICAL DATA
                history = self._get_historical_data(
                    data_h=data_h,
                    forecast_init_time=forecast_init_time,
                    seq_length_td=seq_length_history_td,
                )

                # GET FORECAST DATA
                fcast = self._get_forecast_data(
                    data_f=data_f,
                    forecast_init_time=forecast_init_time,
                )

                # GET TARGET DATA
                target = self._get_target_data(
                    data_y=data_y,
                    forecast_init_time=forecast_init_time,
                    horizon_td=target_horizon_td,
                )

                # GET FUTURE DATA
                future = self._get_future_data(
                    target=target,
                    future_horizon_td=future_horizon_td,
                )

                # FEATURE ENGINEERING
                # TODO: current assumption is that encoding_doy FOR ALL DATA (history, forecast, future)
                history, fcast, future = self._encode_times(history, fcast, future)

                # SELECT FEATURES
                #  (seq_len, len(history_variables))
                history = history[self.history_variables]
                #  (horizon, len(forecast_variables))
                fcast = fcast[self.forecast_variables]

                # SKIP NANS
                # (y only in training period)
                if self.mode == "train":
                    if np.any(target.isnull()):
                        NAN_COUNTER += 1
                        continue

                if np.any(fcast.isnull()):
                    NAN_COUNTER += 1
                    continue

                if np.any(future.isnull()):
                    NAN_COUNTER += 1
                    continue

                if np.any(history.isnull()):
                    NAN_COUNTER += 1
                    continue

                if history.shape[0] != self.seq_len:
                    # not enough history for that sample
                    NAN_COUNTER += 1
                    continue

                if target.shape[0] != self.target_horizon:  # + 1
                    NAN_COUNTER += 1
                    continue
                
                # SAVE ALL DATA to attribute
                self.all_data[COUNTER] = {
                    "x_f": fcast,
                    "x_ff": future,
                    "x_d": history,
                    "y": target,
                }
                self.sample_lookup[COUNTER] = (sample, forecast_init_time)

                COUNTER += 1

        # save for calculation of length
        self.n_samples = COUNTER
        # save metadata for each sample
        self.times = forecast_init_times

        if self.cache:
            # save metadata/check metadata (to check for match)
            # cache to disk
            self.experiment_dir 
            "sample_lookup.pkl"
            "all_data.pkl"
            assert False, "TODO: needs to implement cacheing of data?"
        pass

    def _get_historical_data(
        self,
        data_h: xr.Dataset,
        forecast_init_time: pd.Timestamp,
        seq_length_td: pd.Timedelta,
    ) -> pd.DataFrame:
        # GET HISTORICAL DATA
        history_start_time = forecast_init_time - seq_length_td
        history = data_h.sel(time=slice(history_start_time, forecast_init_time))
        history = (
            history.isel(time=slice(-self.seq_len, None))
            .drop(self.spatial_dim)
            .to_dataframe()
        )

        return history

    def _get_forecast_data(
        self,
        data_f: xr.Dataset,
        forecast_init_time: pd.Timestamp,
    ) -> pd.DataFrame:
        fcast = data_f.sel({self.forecast_initialisation_dim: forecast_init_time})
        # Get the data UP TO horizon
        fcast = fcast.isel({self.forecast_horizon_dim: slice(0, self.forecast_horizon)})
        # rename self.forecast_horizon_dim -> time
        forecast_times = (
            fcast[self.forecast_initialisation_dim] + fcast[self.forecast_horizon_dim]
        ).values
        fcast = fcast.rename({self.forecast_horizon_dim: "time"})
        fcast["time"] = forecast_times

        fcast = (
            fcast.drop_vars(
                [self.forecast_initialisation_dim, "valid_time"], errors="ignore"
            )
            .drop(self.spatial_dim)
            .to_dataframe()
        )
        return fcast

    def _get_future_data(
        self,
        target: pd.DataFrame,
        future_horizon_td: pd.Timedelta,
    ) -> pd.DataFrame:
        """[summary]

        Args:
            target (pd.DataFrame): [description]
            future_horizon_td (pd.Timedelta): [description]

        Returns:
            pd.DataFrame: DataFrame with time as index and empty columns. Populated later with encode_time
        """
        # TODO: how to pass in extra variables to future data?
        # TODO: do you want to create the future data here? that way there's never missing data
        # TODO: but the alternative is to be able to pass the future data into __init__()

        # NOTE: drop the first because that is already included in the forecast
        future_times = pd.date_range(
            target.index.max() - future_horizon_td, target.index.max()
        )[1:]
        future = pd.DataFrame({"time": future_times}).set_index("time")

        return future

    def _get_target_data(
        self,
        data_y: xr.Dataset,
        forecast_init_time: pd.Timestamp,
        horizon_td: pd.Timedelta,
    ) -> pd.DataFrame:
        # time slice from [initialisation_date: forecast_horizon + future_horizon]
        target = data_y[self.target_var].sel(
            time=slice(forecast_init_time, forecast_init_time + horizon_td)
        )
        target = (
            target.isel(time=slice(-self.target_horizon, None)) # target
            .drop(self.spatial_dim)
            .to_dataframe()
        ) 
        return target

    def _encode_times(
        self,
        history: xr.Dataset,
        fcast: xr.Dataset,
        future: xr.Dataset,
    ) -> Tuple[pd.DataFrame, ...]:
        if self.encode_doy:
            fcast["doy_sin"], fcast["doy_cos"] = create_doy(list(fcast.index.dayofyear))
            history["doy_sin"], history["doy_cos"] = create_doy(
                list(history.index.dayofyear)
            )
            future["doy_sin"], future["doy_cos"] = create_doy(
                list(future.index.dayofyear)
            )

        return history, fcast, future

    def get_meta(self, idx: int) -> Tuple[str, pd.Timestamp]:
        return self.sample_lookup[idx]

    def __getitem__(self, idx) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        data: Dict[str, pd.DataFrame] = self.all_data[idx]
        if data == {}:
            return None

        # CREATE META DICT (for recreating outputs for correct time)
        input_times = data["x_d"].index.to_numpy().astype(float)
        target_times = data["y"].index.to_numpy().astype(float)
        meta = {
            "input_times": input_times,
            "target_times": target_times,
            "index": np.array([idx]),
        }

        # CREATE NUMPY ARRAYS FOR DATA
        # (seq_len, n_historical_features)
        x_d = data["x_d"].values
        # (forecast_horizon, n_forecast_features)
        x_f = data["x_f"].values
        # (future_horizon, n_future_features)
        x_ff = data["x_ff"].values
        # (forecast_horizon + future_horizon, 1)
        y = data["y"].values

        data = {
            "meta": meta,
            "x_d": x_d,
            "y": y,
            "x_f": x_f,
            "x_ff": x_ff,
        }

        # CONVERT TO torch.Tensor OBJECTS
        for key in data.keys():
            if isinstance(data[key], dict):
                for key2 in data[key]:
                    data[key][key2] = torch.tensor(data[key][key2]).float()
            else:
                data[key] = torch.tensor(data[key]).float()

        return data


def print_instance(dd: FcastDataset, instance: int):
    print(
        f"{instance}: ",
        [
            (k, dd[instance][k].shape)
            for k in dd[instance].keys()
            if not isinstance(dd[instance][k], dict)
        ],
    )


if __name__ == "__main__":
    from h2ox.scripts.utils import load_zscore_data
    from h2ox.ai.data_utils import encode_doys, create_doy

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
    data_dir = Path(ROOT_DIR / "data")
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
        forecast=train_forecast,  # forecast,
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
    location=dd.get_meta(0)[0]; time=dd.get_meta(0)[1]
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

    assert False
