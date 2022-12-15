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
        target_difference: bool,
        variables_difference: Optional[List[str]],
        shift_target: bool,
        historic_variables: List[str],  # noqa
        forecast_variables: List[str],  # noqa
        future_variables: List[str],
        max_consecutive_nan: int,
        ohe_or_multi: str,
        drop_duplicate_vars: Optional[List[str]],
        normalise: Optional[List[str]],
        zscore: Optional[List[str]],
        std_norm: Optional[List[str]],
        norm_difference: Optional[bool],
        time_dim: str = "date",
        horizon_dim: str = "steps",
        **kwargs,
    ):
        # TODO: error and var checking

        # variables
        self.target_var = target_var
        self.target_difference = target_difference
        self.historic_variables = historic_variables
        self.forecast_variables = forecast_variables
        self.future_variables = future_variables
        self.time_dim = time_dim
        self.horizon_dim = horizon_dim
        self.sites = select_sites
        self.sites_dictionary = dict(enumerate(self.sites))
        self.norm_difference = norm_difference
        self.drop_duplicate_vars = drop_duplicate_vars
        self.shift_target = shift_target
        self.shift_variables = variables_difference

        # soft data and filtering rules
        self.normalise = normalise
        self.zscore = zscore
        self.std_norm = std_norm
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

    def _drop_duplicates(self, data):

        drop_idxs = []
        for var in self.drop_duplicate_vars:
            drop_idxs.append(np.where(data["variable"].data == var)[0][1:])

        keep_idx = ~np.isin(np.arange(data.shape[-1]), np.union1d(*drop_idxs))

        return data.isel({"sites-variable": keep_idx})

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

        if self.drop_duplicate_vars is not None:
            historic = self._drop_duplicates(historic)
            forecast = self._drop_duplicates(forecast)
            future = self._drop_duplicates(future)

        idxs = historic["date"].data[np.isin(historic["date"].data, valid_dates)]

        self.historic = historic.sel({"date": idxs})
        self.forecast = forecast.sel({"date": idxs})
        self.future = future.sel({"date": idxs})
        self.targets = targets.sel({"date": idxs})

        return idxs

    def _reshape_data_sitewise(self, data, valid_dates):

        historic = (
            self._get_historic_data(data)
            .drop("steps")
            .transpose("date", "historic_roll", "global_sites", "variable")
        )

        forecast = self._get_forecast_data(data).transpose(
            "date", "steps", "global_sites", "variable"
        )

        future = self._get_future_data(data).transpose(
            "date", "steps", "global_sites", "variable"
        )

        targets = (
            self._get_target_data(data)
            .drop("steps")
            .transpose("date", "target_roll", "global_sites", "variable")
        )

        idxs = historic["date"].data[np.isin(historic["date"].data, valid_dates)]

        self.historic = historic.sel({"date": idxs})
        self.forecast = forecast.sel({"date": idxs})
        self.future = future.sel({"date": idxs})
        self.targets = targets.sel({"date": idxs})

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
        )
        self.forecast = (
            forecast.sel({"date-site": idxs})
            .to_array()
            .transpose("date-site", "steps", "variable")
        )
        self.future = (
            future.sel({"date-site": idxs})
            .to_array()
            .transpose("date-site", "steps", "variable")
        )
        self.targets = (
            targets.sel({"date-site": idxs})
            .to_array()
            .transpose("date-site", "target_roll", "variable")
            .sel({"variable": self.target_var})
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

        data = data.sel({"global_sites": self.sites})

        self.site_keys = data["global_sites"].values

        # TODO: but are you sure you want to do this here? you want to normalise the data
        # before so you can use the TRAIN mean/std for the TEST data
        def zscore_func(arr):
            return (arr - arr.mean()) / arr.std()  # -ve to +ve std

        def std_func(arr):
            return arr / arr.std()  # -ve to +ve std

        def norm_func(arr):
            return (arr - arr.min()) / (arr.max() - arr.min())  # 0 to 1

        def norm_func_nv(arr):
            return ((arr - arr.min()) / (arr.max() - arr.min())) * 2.0 - 1  # -1 to 1

        # maybe normalise
        logger.info("soft data transforms - maybe shift, normalise, or zscore")
        # store key metrics for recovery later
        self.augment_dict = {"normalise": {}, "zscore": {}, "std_norm": {}}

        if self.shift_variables is not None:
            for var in self.shift_variables:
                data[var] = data[var] - data[var].shift(
                    {"steps": 1}
                )  # ).roll({'steps':-1})
                data[var].loc[{"steps": timedelta(days=0)}] = 0

        if self.normalise is not None:
            for var in self.normalise:
                self.augment_dict["normalise"][var] = {
                    "max": data[var].groupby("global_sites").map(lambda arr: arr.max()),
                    "min": data[var].groupby("global_sites").map(lambda arr: arr.min()),
                }
                data[var] = data[var].groupby("global_sites").map(norm_func)

        if self.std_norm is not None:
            for var in self.std_norm:
                self.augment_dict["std_norm"][var] = {
                    "std": data[var].groupby("global_sites").map(lambda arr: arr.std()),
                }
                data[var] = data[var].groupby("global_sites").map(std_func)

        if self.zscore is not None:
            for var in self.zscore:
                self.augment_dict["zscore"][var] = {
                    "mean": data[var]
                    .groupby("global_sites")
                    .map(lambda arr: arr.mean()),
                    "std": data[var].groupby("global_sites").map(lambda arr: arr.std()),
                }
                data[var] = data[var].groupby("global_sites").map(zscore_func)

        # mask nans by valid datetime
        logger.info("soft data transforms - validate datetimes")
        valid_dates = self.valid_datetimes(data)

        logger.info("soft data transforms - interpolate_1d")
        data = self._interpolate_1d(data)

        if self.target_difference:
            logger.info("soft data transforms - get target difference")
            new_target_vars = []
            for var in self.target_var:
                data[f"shift_{var}"] = data[var] - data[var].shift({"date": 1})
                new_target_vars.append(f"shift_{var}")
            self.target_var = new_target_vars
            data = data.isel({"date": slice(1, None)})

            if self.norm_difference:
                for var in self.target_var:
                    self.augment_dict["std_norm"][var] = {
                        "std": data[var]
                        .groupby("global_sites")
                        .map(lambda arr: arr.std())
                    }
                    data[var] = data[var].groupby("global_sites").map(std_func)

        # if self.shift_target:
        #    for var in self.target_var:
        #        data[var] = data[var]*2.-1.

        self.xr_ds = data.rename({"global_sites": "site", "steps": "step"})

        if self.ohe_or_multi == "multi":
            logger.info("soft data transforms - reshape data multi-target")
            idxs = self._reshape_data_multi(data, valid_dates)
            self.metadata_columns = ["date"]
        elif self.ohe_or_multi == "ohe":
            logger.info("soft data transforms - reshape with one-hot-encoding")
            idxs = self._reshape_data_ohe(data, valid_dates)
            self.metadata_columns = ["date", "site"]
        elif self.ohe_or_multi == "sitewise":
            logger.info("soft data transforms - reshape with sitewise data")
            idxs = self._reshape_data_sitewise(data, valid_dates)
            self.metadata_columns = ["date", "site"]

        logger.info(
            f"soft data transforms - data shape - historic:{self.historic.shape}; forecast:{self.forecast.shape}; future:{self.future.shape}; targets:{self.targets.shape}"
        )

        if self.ohe_or_multi == "sitewise":
            historic_site_idx = {
                site: list(self.historic["global_sites"].data).index(site)
                for site in self.sites
            }
            forecast_site_idx = {
                site: list(self.historic["global_sites"].data).index(site)
                for site in self.sites
            }
            future_site_idx = {
                site: list(self.historic["global_sites"].data).index(site)
                for site in self.sites
            }
            targets_site_idx = {
                site: list(self.historic["global_sites"].data).index(site)
                for site in self.sites
            }

            self.historic_levels = self.historic.sel(
                {"variable": ["targets_WATER_VOLUME"]}
            )
            self.historic = self.historic.sel(
                {"variable": self.historic["variable"] != "targets_WATER_VOLUME"}
            )

        # SAVE ALL DATA to attribute
        logger.info("soft data transforms - build data Dictionary")
        data_ii = 0

        for ii, idx in enumerate(idxs):

            # final check on nan (this isn't great but am getting segfaults from interpolate_na
            if not (
                np.isnan(self.historic.data[ii, ...]).any()
                + np.isnan(self.forecast.data[ii, ...]).any()
                + np.isnan(self.future.data[ii, ...]).any()
                + np.isnan(self.targets.data[ii, ...]).any()
            ):

                if not self.ohe_or_multi == "sitewise":

                    self.all_data[data_ii] = {
                        "x_d": self.historic.data[ii, ...].astype(np.float32),
                        "x_f": self.forecast.data[ii, ...].astype(np.float32),
                        "x_ff": self.future.data[ii, ...].astype(np.float32),
                        "y": self.targets.data[ii, ...].astype(np.float32),
                    }

                    if isinstance(idx, tuple):
                        self.sample_lookup[data_ii] = dict(
                            zip(self.metadata_columns, idx)
                        )
                    else:
                        self.sample_lookup[data_ii] = dict(
                            zip(self.metadata_columns, (idx,))
                        )

                    data_ii += 1

                else:

                    self.all_data[data_ii] = {
                        site: {
                            "hist_level": self.historic_levels.data[
                                ii, :, historic_site_idx[site], :
                            ].astype(np.float32),
                            "x_d": self.historic.data[
                                ii, :, historic_site_idx[site], :
                            ].astype(np.float32),
                            "x_f": self.forecast.data[
                                ii, :, forecast_site_idx[site], :
                            ].astype(np.float32),
                            "x_ff": self.future.data[
                                ii, :, future_site_idx[site], :
                            ].astype(np.float32),
                            "y": self.targets.data[
                                ii, :, targets_site_idx[site], :
                            ].astype(np.float32),
                        }
                        for site in self.sites
                    }

                    self.all_data[data_ii]["y"] = (
                        self.targets.data[
                            ii,
                            :,
                            tuple(targets_site_idx[site] for site in self.sites),
                            :,
                        ]
                        .astype(np.float32)
                        .transpose(1, 0, 2)
                    )

                    if isinstance(idx, tuple):
                        self.sample_lookup[data_ii] = dict(
                            zip(self.metadata_columns, idx)
                        )
                    else:
                        self.sample_lookup[data_ii] = dict(
                            zip(self.metadata_columns, (idx,))
                        )

                    data_ii += 1
            else:
                pass

        self.n_samples = len(idxs)

        # pickle.dump(self.all_data, open("./data/ds_data.pkl", "wb"))
        # pickle.dump(self.sample_lookup, open("./data/ds_meta.pkl", "wb"))

    def _get_historic_data(
        self,
        data: xr.Dataset,
    ) -> xr.Dataset:

        data_h = xr.concat(
            [
                data[self.historic_variables]
                .sel({"steps": np.timedelta64(0)})
                .shift({"date": self.historical_seq_len - ii - 1})
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
                for ii in range(-data[self.sites[0]]["x_d"].shape[0], 0)
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
    train_date_ranges = cfg["train_date_ranges"]
    val_date_ranges = cfg["val_date_ranges"]
    test_date_ranges = cfg["test_date_ranges"]

    index_df = train_dataset._get_meta_dataframe()
    index_df = index_df.sort_values(time_dim)

    index_df = (
        index_df.reset_index().rename(columns={"index": "pt_index"}).set_index(time_dim)
    )

    train_idx = (
        np.sum(
            np.stack(
                [
                    (index_df.index >= chunk[0]) & (index_df.index <= chunk[1])
                    for chunk in train_date_ranges
                ]
            ),
            axis=0,
        )
        > 0
    )
    val_idx = (
        np.sum(
            np.stack(
                [
                    (index_df.index >= chunk[0]) & (index_df.index <= chunk[1])
                    for chunk in val_date_ranges
                ]
            ),
            axis=0,
        )
        > 0
    )
    test_idx = (
        np.sum(
            np.stack(
                [
                    (index_df.index >= chunk[0]) & (index_df.index <= chunk[1])
                    for chunk in test_date_ranges
                ]
            ),
            axis=0,
        )
        > 0
    )

    train_indexes = index_df.loc[train_idx]["pt_index"]
    val_indexes = index_df.loc[val_idx]["pt_index"]
    test_indexes = index_df.loc[test_idx]["pt_index"]

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
