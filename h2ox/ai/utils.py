from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import xskillscore as xs


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


def make_future_data_of_ones(
    min_date: pd.Timestamp, max_date: pd.Timestamp, future_horizon: int
) -> xr.Dataset:
    # create memory-efficient data for all of the forecast timesteps
    pass


def create_model_experiment_folder(
    data_dir: Path, experiment_name: str, add_datetime: bool = False
) -> Path:
    if add_datetime:
        date_str = datetime.now().strftime("%Y%M%d_%H%m")
        experiment_name += "_" + date_str

    expt_dir = data_dir / experiment_name
    if not expt_dir.exists():
        expt_dir.mkdir()
    return expt_dir


def calculate_errors(
    preds: xr.Dataset,
    var: str,
    time_dim: str = "initialisation_time",
    model_str: str = "s2s",
) -> xr.Dataset:
    smp = np.unique(preds["sample"])[0]
    pp = preds.drop("sample")

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
    return errors


def normalize_data(
    ds: xr.Dataset, static: bool = False, time_dim: str = "time"
) -> Tuple[xr.Dataset, Tuple[xr.Dataset, xr.Dataset]]:
    """
    Standardizes the data to have mean 0 and standard deviation 1.
    """
    # (Y - mean) / std
    if not static:
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


# def create_model_save_str(extra_str: str = "") -> str:
#     save_str = ""
#     date_str = datetime.now().strftime("%Y%m%d_%H%M")
#     save_str += f"TT{date_str}_{LOCATION}_{TARGET_VAR}_{TRAIN_MIN}{TRAIN_MAX}"
#     save_str += f"_H{HIDDEN_SIZE}_BS{BATCH_SIZE}"
#     save_str += "_diff" if DIFFED else ""
#     save_str += "_deseas" if DESEAS else ""
#     save_str += "_rolling" if ROLLING else ""

#     # anything else to add?
#     save_str += "_" + extra_str.replace(" ", "")

#     return save_str
