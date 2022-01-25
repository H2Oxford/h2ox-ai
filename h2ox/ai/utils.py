from typing import Union, Tuple, List
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr


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


def create_model_experiment_folder(data_dir: Path, experiment_name: str, add_datetime: bool = False) -> Path:
    if add_datetime:
        date_str = datetime.now().strftime("%Y%M%d_%H%m")
        experiment_name += "_" + date_str
    
    expt_dir = data_dir / experiment_name
    if not expt_dir.exists():
        expt_dir.mkdir()
    return expt_dir


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
