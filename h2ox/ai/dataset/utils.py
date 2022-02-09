from pathlib import Path
from typing import List, Optional, Tuple, Union

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr
import xskillscore as xs


def gcsfs_mapper():
    """Filesystem mapper for Google Cloud Storage paths."""
    fs = gcsfs.GCSFileSystem(access="read_only")
    return fs.get_mapper


def null_mapper():
    """File system mapper for local paths (i.e. no mapper required)"""

    def return_null(x):
        return x

    return return_null


def group_consecutive_nans(
    da: xr.DataArray,
    variable_name: str,
    outer_groupby_coords: str = "global_sites",
    time_coord: str = "date",
) -> xr.DataArray:
    """A method to count consecutive nans along a given dimension.
    Needed to mask consecutive nans exceeding a threshold prior to interpolation.

    E.g.:
    Input sample in da (along time_coord):     [3,5,nan,nan,7,9,10,nan,nan,nan,3,2,nan,5]
    Output sample along da (along time_coord): [0,0,2,2,0,0,0,3,3,3,0,0,1,0]

    Args:
        da (xr.DataArray): array with original data and some nan
        variable_name (str): variable name to select from array (for reverse broadcast)
        outer_groupby_coords (str): outer map dimsion
        time_coord (str): dimension along which to count consecutive nan

    Returns:
        xr.DataArray: [description]
    """
    cons_nan_da = (
        (da.isnull() != da.isnull().shift({time_coord: 1}, fill_value=np.nan))
        .cumsum(dim=time_coord)
        .groupby(outer_groupby_coords)
        .map(
            # groupby all remaning dims (date, steps[horizon])
            lambda g: g.groupby(...)
            .count()
            .sel({variable_name: g})
            .drop(variable_name)
        )
    )

    cons_nan_da = cons_nan_da.where(da.isnull()).fillna(0)

    return cons_nan_da


def assymetric_boolean_dilate(
    arr: np.ndarray,
    left_dilation: int,
    right_dilatation: int,
    target: Union[bool, int],
) -> np.ndarray:
    """A method to asymmetrically dilate a 1-D boolean array.
    Needed to mask indices of dates prior to and proceeding from nan data

    E.g.
    left_dilation: 1 step
    right_dilation: 2 steps
    target: 1
    Input arr:  [0,0,0,1,1,0,0,1,1,1,0,0,0,0,0,0,1]
    (flip:      [0,0,L,1,1,R,*,1,1,1,R,R,0,0,0,L,1], *: L & R)
    output arr: [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1]

    Args:
        arr (np.ndarray): array of boolean values of length `date`
        left_dilation (int): dilation to the left, i.e. historic sequence length
        right_dilatation (int): dilation to the right, i.e. forecast/future horizon
        target Union[bool, int]: the value in arr to dilate, i.e. dilate 1/True? or 0/False?

    Returns:
        np.ndarray: [description]
    """
    # get left and right edges where boolean mask changes sign
    # left_edge
    left_edge = np.convolve(arr == target, np.array([1, -1]), mode="same")
    # right edge
    right_edge = np.convolve(arr[::-1] == target, np.array([1, -1]), mode="same")[::-1]

    # get new indices to change bool sign
    add_left_idx = np.array(
        [np.arange(e - left_dilation, e) for e in np.where(left_edge)[0]]
    ).flatten()
    add_right_idx = np.array(
        [np.arange(e + 1, e + 1 + right_dilatation) for e in np.where(right_edge)[0]]
    ).flatten()

    # flatten, combine, and cut off out-of-bounds
    flip_idx = np.unique(np.concatenate([add_left_idx, add_right_idx]))
    flip_idx = flip_idx[(flip_idx >= 0) & (flip_idx < arr.shape[0])]

    if flip_idx.shape[0] > 0:
        # reassign values in boolean array
        arr[flip_idx] = target

    return arr


def encode_doys(
    doys: Union[int, List[int]], start_doy: int = 1, end_doy: int = 366
) -> Tuple[List[float], List[float]]:
    """
    encode date(s)/doy(s) to cyclic sine/cosine values
    returns two lists, one with sine-encoded and one with cosine-encoded doys
    """
    if not isinstance(doys, list):
        doys = [doys]

    doys_sin = []
    doys_cos = []
    for doy in doys:
        doys_sin.append(
            np.sin(2 * np.pi * (doy - start_doy) / (end_doy - start_doy + 1))
        )
        doys_cos.append(
            np.cos(2 * np.pi * (doy - start_doy) / (end_doy - start_doy + 1))
        )

    return doys_sin, doys_cos


def encode_array_of_datetimes_to_sin_cos(dates: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Return two arrays of sin, cos encoding of the dates in the array

    Args:
        dates (np.ndarray): Input array of datetimes[64]

    Returns:
        Tuple[np.ndarray, ...]: two output arrays of sin, cos encodings [float64]
    """
    shape = dates.shape
    dayofyear = np.array(pd.to_datetime(dates.flatten()).dayofyear).reshape(shape)
    doys_sin, doys_cos = encode_doys(dayofyear)
    return doys_sin[0], doys_cos[0]


def assign_sin_cos_to_forecast():
    # extract forecast horizon
    # dayofyear and then encode
    # copy forward to all location/samples
    # assign to forecast object as new variables (sin_doy, cos_doy)
    pass


def assign_sin_cos_doy_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # assert that index is datetime
    assert df.index.dtype == pd.to_datetime(["2013"]).dtype
    doys_sin, doys_cos = encode_doys(df.index.dayofyear)
    df["doy_sin"] = doys_sin[0]
    df["doy_cos"] = doys_cos[0]

    return df


def create_doy(values: List[int]) -> Tuple[List[float], ...]:
    # create day of year feature
    return encode_doys(values, start_doy=1, end_doy=366)


def calculate_errors(
    preds: xr.Dataset,
    var: str,
    time_dim: str = "initialisation_time",
    model_str: str = "s2s",
) -> xr.Dataset:
    # smp = np.unique(preds["sample"])[0]
    all_sample_errors = []
    for smp in np.unique(preds["sample"]):
        pp = preds.sel({time_dim: preds["sample"] == smp}).drop("sample")

        # TODO: use another library for these scores, xs dependency pip isntall hangs?
        rmse = (
            xs.rmse(pp["obs"], pp["sim"], dim=time_dim)
            .expand_dims(model=[model_str])
            .expand_dims(sample=[smp])
            .expand_dims(variable=[var])
            .rename("rmse")
        )
        pearson = (
            xs.pearson_r(pp["obs"], pp["sim"], dim=time_dim)
            .expand_dims(model=[model_str])
            .expand_dims(sample=[smp])
            .expand_dims(variable=[var])
            .rename("pearson-r")
        )
        mape = (
            xs.mape(pp["obs"], pp["sim"], dim=time_dim)
            .expand_dims(model=[model_str])
            .expand_dims(sample=[smp])
            .expand_dims(variable=[var])
            .rename("mape")
        )

        errors = xr.merge([rmse, pearson, mape])
        all_sample_errors.append(errors)

    all_errors = xr.merge(all_sample_errors)
    return all_errors


def normalize_data(
    ds: xr.Dataset,
    static_or_global: bool = False,
    time_dim: str = "date",
    mean_: Optional[xr.Dataset] = None,
    std_: Optional[xr.Dataset] = None,
) -> Tuple[xr.Dataset, Tuple[xr.Dataset, xr.Dataset]]:
    """
    Standardizes the data to have mean 0 and standard deviation 1.
    """
    # (Y - mean) / std
    if mean_ is None:
        assert std_ is None, "mean_ and std_ must both be None or neither None"
        if not static_or_global:
            mean_ = ds.mean(dim=time_dim)
            std_ = ds.std(dim=time_dim)
        else:
            mean_ = ds.mean()
            std_ = ds.std()

    norm_ds = (ds - mean_) / std_

    return norm_ds, (mean_, std_)


# unnormalize
def unnormalize_preds(
    preds: xr.Dataset,
    mean_: xr.Dataset,
    std_: xr.Dataset,
    target: str,
    sample: str,
    sample_dim: str = "location",
) -> xr.Dataset:
    # TODO: fix all naming conventions
    if "sample" not in preds.coords:
        preds = preds.assign_coords(sample=preds["sample"])
    # (Y * std) + mean
    preds["obs"] = (
        preds["obs"].sel(initialisation_time=preds["sample"] == sample)
        * std_[target].sel({sample_dim: sample}).values
    ) + mean_[target].sel({sample_dim: sample}).values
    preds["sim"] = (
        preds["sim"].sel(initialisation_time=preds["sample"] == sample)
        * std_[target].sel({sample_dim: sample}).values
    ) + mean_[target].sel({sample_dim: sample}).values

    return preds


def read_interim_zscore_data(data_dir: Path) -> pd.DataFrame:
    # load the data into xarray object
    data_paths = list((data_dir / "interim").glob("*.csv"))
    # date-index is the first column
    data = [pd.read_csv(f, index_col=0, parse_dates=True) for f in data_paths]
    # get location from the filename and assign date-location index
    locations = [f.name.replace("_zscore.csv", "") for f in data_paths]

    # load in each location-dataframe and concatenate
    for ix, df in enumerate(data):
        df.index.name = "date"
        df["location"] = locations[ix]
        df = df.reset_index().set_index(["date", "location"])
        # drop non-unique values
        df = df.drop_duplicates()
        data[ix] = df

    df = pd.concat(data)
    return df


def convert_dataset_to_forecast_dataset(ds: xr.Dataset) -> xr.Dataset:
    # get the forecast variables from the dataset
    forecast = ds[[v for v in ds.data_vars if "_0" not in v]].drop(
        ["set", "yrmnth", "volume_bcm"]
    )

    # convert to dataframe and change the structure (wide -> long)
    df = forecast.to_dataframe()
    forecast_long = df.stack().reset_index().rename({0: "value"}, axis=1)
    _extra = pd.DataFrame(forecast_long["level_2"].str.split("_").to_list()).rename(
        {0: "variable", 1: "forecast_horizon"}, axis=1
    )
    forecast_long = forecast_long.join(_extra)

    # convert forecast_horizon to timedelta
    forecast_long["forecast_horizon"] = pd.to_timedelta(
        forecast_long["forecast_horizon"].astype(int), unit="D"
    )
    # pivot to turn variables into columns
    forecast_long = (
        forecast_long.set_index(["date", "location", "forecast_horizon"])
        .drop("level_2", axis=1)
        .pivot(columns="variable")
    )
    forecast_long.columns = forecast_long.columns.droplevel()

    # convert to xarray object
    forecast = forecast_long.to_xarray()

    # add a "real time" coordinate
    initialisation_date = forecast["date"].values
    valid_time = (
        initialisation_date[:, np.newaxis] + forecast["forecast_horizon"].values
    )
    forecast = forecast.assign_coords(
        valid_time=(["date", "forecast_horizon"], valid_time)
    )

    return forecast


def interim_zscore_dataframes_to_xarray(ds: xr.Dataset) -> Tuple[xr.Dataset, ...]:
    # load target
    target = ds[["volume_bcm"]]

    # load history
    history = ds[[v for v in ds.data_vars if "_0" in v]]
    history = history.rename({v: v.replace("_0", "") for v in history.data_vars})

    # load forecast
    forecast = convert_dataset_to_forecast_dataset(ds)

    # convert "date" to "time"
    target = target.rename({"date": "time"})
    history = history.rename({"date": "time"})
    forecast = forecast.rename({"date": "initialisation_time"})

    return target, history, forecast


def load_zscore_data(data_dir: Path) -> Tuple[xr.Dataset, ...]:
    df = read_interim_zscore_data(data_dir)
    original_experiment_splits(df)

    ds = df.to_xarray()
    target, history, forecast = interim_zscore_dataframes_to_xarray(ds)

    return target, history, forecast


def load_samantha_dataframe_to_dataset(df: pd.DataFrame) -> xr.Dataset:
    # parse datetimes (in different formats)
    rows_slash_format = df["date"].str.contains("/")  # 4/1/2000
    rows_dash_format = df["date"].str.contains("-")  # 2022-01-01
    slash_format_datetimes = pd.to_datetime(
        df.loc[rows_slash_format, "date"], format="%m/%d/%Y"
    )
    dash_format_datetimes = pd.to_datetime(
        df.loc[rows_dash_format, "date"], format="%Y-%m-%d"
    )
    # combine the two
    datetimes = slash_format_datetimes.append(dash_format_datetimes)
    df.loc[:, "date"] = datetimes

    # convert to xarray object
    df = df.set_index(["date", "location"])
    # df.index.duplicated()
    df = df[~df.index.duplicated(keep="first")]
    ds = df.to_xarray()

    return ds


def load_PFAF_ID_metadata(data_dir: Path) -> pd.DataFrame:
    metadata_path = data_dir / "raw" / "W4P_reservoir_watersheds.csv"
    df = pd.read_csv(metadata_path).iloc[
        :,
        1:,
    ]
    return df


def load_samantha_data(data_dir: Path) -> Tuple[xr.Dataset, pd.DataFrame]:
    data_files = list((data_dir / "raw").glob("*modis.csv"))
    data = [
        pd.read_csv(f).rename(
            {"PFAF_ID": "location", "mean": f.name.split("_")[0]}, axis=1
        )
        for f in data_files
    ]

    all_ds = []
    for df in data:
        all_ds.append(load_samantha_dataframe_to_dataset(df))

    ds = xr.merge(all_ds)
    meta = load_PFAF_ID_metadata(data_dir)

    return ds, meta


def load_reservoir_metas(data_dir: Path) -> pd.DataFrame:
    meta_path = data_dir / "raw" / "h2ox_reservoirs.csv"
    df = pd.read_csv(meta_path)
    return df


def get_all_big_q_data_as_xarray(data_dir: Path) -> xr.Dataset:
    # df = pd.read_csv(data_dir / "raw" / "bigquery_24012022_original.csv", parse_dates=["DATETIME"]).rename({"DATETIME": "date", "RESERVOIR_NAME": "location"}, axis=1)
    df = pd.read_csv(
        data_dir / "raw" / "bigquery_24012022_historic.csv", parse_dates=["date"]
    ).rename({"reservoir": "location"}, axis=1)

    # convert dataframe to xr.Dataset
    df = df.sort_values(["location", "date"]).set_index(["date", "location"])
    ds = df.loc[~df.index.duplicated(keep="last")].to_xarray()
    return ds


def original_experiment_splits(df: pd.DataFrame) -> pd.DataFrame:
    train = df.reset_index().query("set == 'trn'").date.unique()
    val = df.reset_index().query("set == 'val'").date.unique()
    test = df.reset_index().query("set == 'test'").date.unique()

    return pd.DataFrame(
        {
            "set": np.concatenate(
                [["train" for _ in train], ["val" for _ in val], ["test" for _ in test]]
            ),
            "date": np.concatenate([train, val, test]),
        }
    )


def load_samantha_updated_data(data_dir: Path) -> xr.Dataset:
    data_file = list((data_dir / "raw").glob("samantha_data.csv"))[0]
    assert data_file.exists()
    df = pd.read_csv(data_file, parse_dates=True)
    df["time"] = pd.to_datetime(df["time"])
    ds = df.set_index(["time", "location"]).to_xarray()
    return ds
