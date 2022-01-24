from typing import Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from definitions import ROOT_DIR


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


if __name__ == "__main__":
    data_dir = Path(ROOT_DIR / "data")
    target, history, forecast = load_zscore_data(data_dir)
    sam_data, meta = load_samantha_data(data_dir)
    bigq_meta = load_reservoir_metas(data_dir)
    bigq_target = get_all_big_q_data_as_xarray(data_dir)

    # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/tommylees/Downloads/oxeo-main-c8ac2f9d9d36.json'
    # from google.cloud import bigquery
    # table_name = "oxeo-main.wave2web.reservoir-data"
    # client = bigquery.Client()
    # query = "SELECT * FROM `oxeo-main.wave2web.reservoir-data`"
