from collections import defaultdict
from datetime import timedelta
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from pandas.api.types import is_numeric_dtype
from torch import Tensor
from torch.utils.data import Dataset, Subset

from h2ox.ai.dataset.utils import assymetric_boolean_dilate, group_consecutive_nans


class FcastDataset(Dataset):
    def __init__(
        self,
        data: xr.Dataset,
        select_sites: List[str],
        historical_seq_len: int,
        forecast_horizon: int,
        future_horizon: int,
        target_var: str,
        historic_variables: List[str],  # noqa
        forecast_variables: List[str],  # noqa
        future_variables: List[str],
        max_consecutive_nan: int,
        ohe_or_multi: str,
        normalise: Optional[List[str]],
        time_dim: str = "date",
        horizon_dim: str = "steps",
        **kwargs,
    ):
        # TODO: error and var checking

        # variables
        self.target_var = target_var
        self.historic_variables = historic_variables
        self.forecast_variables = forecast_variables
        self.future_variables = future_variables
        self.time_dim = time_dim
        self.horizon_dim = horizon_dim
        self.sites = data["global_sites"].values
        self.sites_dictionary = dict(enumerate(self.sites))

        # soft data and filtering rules
        self.normalise = normalise
        self.max_consecutive_nan = max_consecutive_nan

        # self.include = include_doy

        # size of arrays
        self.historical_seq_len = historical_seq_len
        self.future_horizon = future_horizon
        self.forecast_horizon = forecast_horizon
        self.target_horizon = self.future_horizon + self.forecast_horizon

        self.ohe_or_multi = ohe_or_multi

        # turn the data into a dictionary for model input
        self.engineer_arrays(data)

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        total_str = "FcastDataset\n"
        total_str += "------------\n"
        total_str += "\n"
        total_str += f"N Samples: {self.__len__()}\n"
        total_str += f"target: {self.target_var}\n"
        total_str += f"historic_variables: {self.historic_variables}\n"
        total_str += f"forecast_variables: {self.forecast_variables}\n"
        total_str += f"future_variables: {self.future_variables}\n"
        total_str += f"historical_seq_len: {self.historical_seq_len}D\n"
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
        meta_df = self._get_meta_dataframe()
        tmin = pd.Timestamp(meta_df[self.time_dim].min())
        tmax = pd.Timestamp(meta_df[self.time_dim].max())
        total_str += f"PERIOD: {tmin}: {tmax}\n"
        total_str += f"LOCATIONS: {meta_df['location'].unique()}\n"

        return total_str

    def valid_datetimes(self, data):

        # first filter consecutive nans above a certain threshold (for corrupt data, etc.)
        dd = {}
        for var in list(data.data_vars):

            cons_nan = group_consecutive_nans(
                da=data[var],
                variable_name=var,
                outer_groupby_coords="global_sites",
                time_coord=self.time_dim,
            )
            dd[var] = cons_nan

        # find any sites/variables with more than max_consecutive_nan consecutive nans
        # TODO tommy: if you select only the first horizon why do you calculate over all?
        consecutive_nan_da = (
            (xr.merge(dd.values()) > self.max_consecutive_nan)
            .isel({self.horizon_dim: 0})
            .to_array()
            .any(dim=("variable", "global_sites"))
        )

        dates = consecutive_nan_da[self.time_dim].data

        consecutive_nan_arr = consecutive_nan_da.data
        # dialate this boolean mask to accomodate historic and future sequence length
        consecutive_nan_arr = assymetric_boolean_dilate(
            arr=consecutive_nan_arr,
            left_dilation=self.historical_seq_len,
            right_dilatation=self.target_horizon,
            target=True,
        )

        # mask the early and late data for the sequence lengths
        consecutive_nan_arr[: self.historical_seq_len] = True
        consecutive_nan_arr[-1 * self.target_horizon :] = True

        return dates[~consecutive_nan_arr]

    def _onehotencode(self, data_portion, offset_dim):
        ohe = pd.get_dummies(
            data_portion.transpose("date", "global_sites", offset_dim, "variable")
            .stack({"date-site": ("date", "global_sites")})["global_sites"]
            .to_dataframe()
        ).to_xarray()

        return xr.merge([data_portion.to_dataset(dim="variable"), ohe]).stack(
            {"date-site": ("date", "global_sites")}
        )

    def _reshape_data_multi(self, data, valid_dates):

        historic = (
            self._get_historic_data(data)
            .drop("steps")
            .stack({"sites-variable": ("global_sites", "variable")})
            .transpose("date", "historic_roll", "sites-variable")
        )

        forecast = (
            self._get_forecast_data(data)
            .stack({"sites-variable": ("global_sites", "variable")})
            .transpose("date", "steps", "sites-variable")
        )

        future = (
            self._get_future_data(data)
            .stack({"sites-variable": ("global_sites", "variable")})
            .transpose("date", "steps", "sites-variable")
        )

        targets = (
            self._get_target_data(data)
            .drop("steps")
            .stack({"sites-variable": ("global_sites", "variable")})
            .transpose("date", "target_roll", "sites-variable")
        )

        idxs = historic["date"].data[np.isin(historic["date"].data, valid_dates)]

        self.historic = historic.sel({"date": idxs}).data
        self.forecast = forecast.sel({"date": idxs}).data
        self.future = future.sel({"date": idxs}).data
        self.targets = targets.sel({"date": idxs}).data

        return idxs

    def _reshape_data_ohe(self, data, valid_dates):

        historic = self._onehotencode(
            self._get_historic_data(data).drop("steps"), "historic_roll"
        )  # DATES*sites x STEPS x var+ohe

        forecast = self._onehotencode(
            self._get_forecast_data(data), "steps"
        )  # DATES*sites x STEPS x var+ohe

        future = self._onehotencode(
            self._get_future_data(data), "steps"
        )  # DATES*sites x STEPS x var+ohe

        targets = self._onehotencode(
            self._get_target_data(data).drop("steps"), "target_roll"
        )  # DATES*sites x STEPS x var+ohe

        idxs = historic["date-site"].data[np.isin(historic["date"].data, valid_dates)]

        self.historic = (
            historic.sel({"date-site": idxs})
            .to_array()
            .transpose("date-site", "historic_roll", "variable")
            .data
        )
        self.forecast = (
            forecast.sel({"date-site": idxs})
            .to_array()
            .transpose("date-site", "steps", "variable")
            .data
        )
        self.future = (
            future.sel({"date-site": idxs})
            .to_array()
            .transpose("date-site", "steps", "variable")
            .data
        )
        self.targets = (
            targets.sel({"date-site": idxs})
            .to_array()
            .transpose("date-site", "target_roll", "variable")
            .sel({"variable": self.target_var})
            .data
        )

        return idxs

    def _interpolate_1d(self, data):
        for var in list(data.keys()):
            if is_numeric_dtype(data[var]):
                data[var] = data[var].interpolate_na(
                    dim="date", method="linear", limit=self.max_consecutive_nan
                )

        return data

    def engineer_arrays(self, data: xr.Dataset):
        """Create an `all_data` attribute which stores all the data
            DefaultDict[int, Dict[str, np.ndarray]]
        Create a `sample_lookup` attribute which stores the
            location & initialisation_date info.
            Dict[int, Tuple[str, pd.Timestamp]]
        """

        # initialise the dictionary storing all the data
        self.all_data: DefaultDict[int, Dict] = defaultdict(dict)
        self.sample_lookup: Dict[int, Tuple[str, pd.Timestamp]] = {}

        self.site_keys = data["global_sites"].values

        # TODO: but are you sure you want to do this here? you want to normalise the data
        # before so you can use the TRAIN mean/std for the TEST data
        def normalise_func(arr):
            return (arr - arr.mean()) / arr.std()

        # maybe normalise
        logger.info("soft data transforms - maybe normalise")
        if self.normalise is not None:
            for var in self.normalise:
                data[var] = data[var].groupby("global_sites").map(normalise_func)

        # mask nans by valid datetime
        logger.info("soft data transforms - validate datetimes")
        valid_dates = self.valid_datetimes(data)

        logger.info("soft data transforms - interpolate_1d")
        data = self._interpolate_1d(data)

        if self.ohe_or_multi == "multi":
            logger.info("soft data transforms - reshape data multi-target")
            idxs = self._reshape_data_multi(data, valid_dates)
            self.metadata_columns = ["date"]
        elif self.ohe_or_multi == "ohe":
            logger.info("soft data transforms - reshape with one-hot-encoding")
            idxs = self._reshape_data_ohe(data, valid_dates)
            self.metadata_columns = ["date", "site"]

        logger.info(
            f"soft data transforms - data shape - historic:{self.historic.shape}; forecast:{self.forecast.shape}; future:{self.future.shape}; targets:{self.targets.shape}"
        )

        # SAVE ALL DATA to attribute
        logger.info("soft data transforms - build data Dictionary")
        for ii, idx in enumerate(idxs):

            self.all_data[ii] = {
                "x_d": self.historic[ii, ...].astype(np.float32),
                "x_f": self.forecast[ii, ...].astype(np.float32),
                "x_ff": self.future[ii, ...].astype(np.float32),
                "y": self.targets[ii, ...].astype(np.float32),
            }

            if isinstance(idx, tuple):
                self.sample_lookup[ii] = dict(zip(self.metadata_columns, idx))
            else:
                self.sample_lookup[ii] = dict(zip(self.metadata_columns, (idx,)))

        self.n_samples = len(idxs)

    def _get_historic_data(
        self,
        data: xr.Dataset,
    ) -> xr.Dataset:

        data_h = xr.concat(
            [
                data[self.historic_variables]
                .sel({"steps": np.timedelta64(0)})
                .shift({"date": ii})
                for ii in range(self.historical_seq_len)
            ],
            pd.TimedeltaIndex(
                [
                    timedelta(days=-self.historical_seq_len + ii)
                    for ii in range(self.historical_seq_len)
                ],
                name="historic_roll",
            ),
        )

        return data_h.to_array().transpose(
            "date", "global_sites", "variable", "historic_roll"
        )

    def _get_forecast_data(
        self,
        data: xr.Dataset,
    ) -> xr.Dataset:

        forecast_period = pd.TimedeltaIndex(
            [timedelta(days=ii) for ii in range(1, self.forecast_horizon + 1)]
        )

        return (
            data[self.forecast_variables]
            .to_array()
            .sel({"steps": forecast_period})
            .transpose("date", "global_sites", "variable", "steps")
        )

    def _get_future_data(
        self,
        data: xr.Dataset,
    ) -> xr.Dataset:

        future_period = pd.TimedeltaIndex(
            [
                timedelta(days=ii)
                for ii in range(
                    self.forecast_horizon + 1,
                    self.forecast_horizon + self.future_horizon + 1,
                )
            ]
        )

        return (
            data[self.future_variables]
            .to_array()
            .sel({"steps": future_period})
            .transpose("date", "global_sites", "variable", "steps")
        )

    def _get_target_data(
        self,
        data: xr.Dataset,
    ) -> xr.Dataset:

        data_y = xr.concat(
            [
                data[self.target_var]
                .sel({"steps": np.timedelta64(0)})
                .shift({"date": -ii})
                for ii in range(1, self.forecast_horizon + self.future_horizon + 1)
            ],
            pd.TimedeltaIndex(
                [
                    timedelta(days=ii)
                    for ii in range(1, self.forecast_horizon + self.future_horizon + 1)
                ],
                name="target_roll",
            ),
        )

        return data_y.to_array().transpose(
            "date", "global_sites", "variable", "target_roll"
        )

    @staticmethod
    def _merge_dataframe_of_one_hot_encoded_data(
        array: np.ndarray,
        df: pd.DataFrame,
        new_column_names: List[str],
    ) -> pd.DataFrame:
        # repeat `array` for each row in the dataframe
        ohe = pd.DataFrame(
            np.tile(array, df.shape[0]).reshape(df.shape[0], -1),
            index=df.index,
            columns=new_column_names,
        )
        # append new ohe columns as features to dataframes
        df = df.join(ohe)

        return df

    def _encode_location(
        self,
        history: xr.Dataset,
        fcast: Optional[xr.Dataset],
        future: xr.Dataset,
        sample: str,
    ) -> Tuple[pd.DataFrame, ...]:
        array = self.categories_lookup[sample]

        # add new columns to dataframes of features
        history = self._merge_dataframe_of_one_hot_encoded_data(
            array, history, self.new_column_names
        )
        fcast = self._merge_dataframe_of_one_hot_encoded_data(
            array, fcast, self.new_column_names
        )
        future = self._merge_dataframe_of_one_hot_encoded_data(
            array, future, self.new_column_names
        )

        return history, fcast, future

    def _get_meta_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.sample_lookup.values(),
            index=self.sample_lookup.keys(),
            columns=self.metadata_columns,
        )

    def __getitem__(self, idx) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:

        data: Dict[str, pd.DataFrame] = self.all_data[idx]

        meta = self.sample_lookup[idx]

        if data == {}:
            return None

        # CREATE META DICT (for recreating outputs for correct time)
        #  NOTE: has to be in float format to play nicely with pytorch DataLoaders
        input_times = np.array(
            [
                (pd.to_datetime(meta["date"]) + timedelta(days=ii)).to_numpy()
                for ii in range(-data["x_d"].shape[0], 0)
            ]
        ).astype(float)
        target_times = np.array(
            [
                (pd.to_datetime(meta["date"]) + timedelta(days=ii)).to_numpy()
                for ii in range(1, data["y"].shape[0] + 1)
            ]
        ).astype(float)

        meta_dict = {  # noqa
            "input_times": input_times,
            "target_times": target_times,
            "index": np.array([idx]),
        }

        if "site" in meta.keys():
            meta_dict["site"] = np.array(
                [{v: k for (k, v) in self.sites_dictionary.items()}[meta["site"]]]
            )

        data["meta"] = meta_dict

        """
        # CONVERT TO torch.Tensor OBJECTS
        for key in data.keys():
            if key != "meta":
                if isinstance(data[key], dict):
                    for key2 in data[key]:
                        data[key][key2] = data[key][key2].astype(np.float32)
                else:
                    data[key] = data[key].astype(np.float32)
        """

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

    index_df = train_dataset._get_meta_dataframe()
    index_df = index_df.sort_values(time_dim)

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

    train_dd = Subset(train_dataset, train_indexes)
    validation_dd = Subset(train_dataset, val_indexes)
    test_dd = Subset(train_dataset, test_indexes)

    return train_dd, validation_dd, test_dd
