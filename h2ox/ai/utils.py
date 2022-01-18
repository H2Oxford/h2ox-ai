from typing import Union, Tuple, List
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
