import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import xarray as xr
from google.cloud import bigquery
from loguru import logger
from ts.context import Context

from h2ox.ai.data_units import SynthTrigDoY
from h2ox.ai.serve import H2OxHandler

# pipe logger to sys.stdout specifically - no error in cloud console
logger.remove()
logger.add(sys.stdout, colorize=False, format="{time:YYYYMMDDHHmmss}|{level}|{message}")


def reshape_data(data_dict):
    """{'<site>':{'<var>':<data>}}"""
    res_dict = {}
    for ii_s, site in enumerate(data_dict["coords"]["site"]["data"]):
        res_dict[site] = dict(
            zip(data_dict["coords"]["variable"]["data"], data_dict["data"][ii_s])
        )

    return res_dict


def combine_data(hist, target, forecast, future):
    combined_data = {}
    for site in hist.keys():
        combined_data[site] = {
            "x_d": hist[site],
            "x_f": forecast[site],
            "x_ff": future[site],
            "y": target[site],
        }
    return combined_data


def explode_and_step(df, col, col_name):
    selection = df.explode(col).reset_index()

    selection["step"] = selection.groupby("index").cumcount()
    selection["step"] = selection["step"].apply(lambda el: timedelta(days=el))
    selection["date"] = pd.to_datetime(selection["date"], utc=True)

    return (
        selection.drop(columns=["index"])
        .rename(columns={col: col_name, "reservoir": "site"})
        .groupby(["date", "step", "site"])
        .nth(0)
    )


class BQInfClient:
    def __init__(self):

        self.client = bigquery.Client()

        self.min_dt = datetime(2010, 1, 1)

        self.tables = {
            "tracking": "oxeo-main.wave2web.tracked-reservoirs",
            "levels": "oxeo-main.wave2web.reservoir-data",
            "forecast": "oxeo-main.wave2web.forecast",
            "precipitation": "oxeo-main.wave2web.precipitation",
            "prediction": "oxeo-main.wave2web.prediction",
        }

    def check_errors(self, errors):

        if errors != []:
            raise ValueError(
                f"there where {len(errors)} error when inserting. " + str(errors),
            )

        return True

    def get_max_available_dates(self, table_key, reservoir_key, date_key):

        Q = f"""
            SELECT t.*
            FROM (SELECT t.*,
                         ROW_NUMBER() OVER (PARTITION BY {reservoir_key}
                                            ORDER BY {date_key} DESC
                                           ) as seqnum
                  FROM `{self.tables[table_key]}` as t
                 ) t
            WHERE seqnum = 1;
        """

        return self.client.query(Q).result().to_dataframe()

    def get_dates_df(self):

        # query most recent data
        df_levels = self.get_max_available_dates("levels", "RESERVOIR_NAME", "DATETIME")
        df_current = self.get_max_available_dates("prediction", "reservoir", "date")
        df_hist = self.get_max_available_dates("precipitation", "reservoir", "date")
        df_forecast = self.get_max_available_dates("forecast", "reservoir", "date")

        # concat into df
        dates_df = pd.concat(
            [
                df_levels.set_index("RESERVOIR_NAME")[["DATETIME"]].rename(
                    columns={"DATETIME": "date_levels"}
                ),
                df_current.set_index("reservoir")[["date"]].rename(
                    columns={"date": "date_current"}
                ),
                df_hist.set_index("reservoir")[["date"]].rename(
                    columns={"date": "date_hist"}
                ),
                df_forecast.set_index("reservoir")[["date"]].rename(
                    columns={"date": "date_forecast"}
                ),
            ],
            axis=1,
        ).fillna(datetime(2010, 1, 1))

        # cast the columns properly
        for cc in dates_df.columns:
            dates_df[cc] = pd.to_datetime(dates_df[cc], utc=True)

        # updateable is where levels, hist, & forcase > current
        dates_df["updateable"] = (
            (dates_df["date_levels"] > dates_df["date_current"])
            & (dates_df["date_hist"] > dates_df["date_current"])
            & (dates_df["date_forecast"] > dates_df["date_current"])
        )

        dates_df["constraint"] = dates_df[
            ["date_levels", "date_hist", "date_forecast"]
        ].min(axis=1)

        return dates_df

    def get_bq_data(
        self,
        table_key,
        min_dt,
        max_dt,
        sites,
        date_key="date",
        reservoir_key="reservoir",
        dt_cast_key="date",
    ):

        sites_str = json.dumps(sites).replace("[", "").replace("]", "")

        Q = f"""
            SELECT *
            FROM `{self.tables[table_key]}`
            WHERE {date_key}>{dt_cast_key}("{min_dt.isoformat()[0:10]}")
            AND {date_key}<={dt_cast_key}("{max_dt.isoformat()[0:10]}")
            AND {reservoir_key} in ({sites_str});
        """

        return self.client.query(Q).result().to_dataframe()

    def bq_to_xr(self, run_date, today, cfg, sites):

        precip_data = self.get_bq_data(
            "precipitation", run_date - timedelta(days=91), today, sites
        )
        forecast_data = self.get_bq_data(
            "forecast", run_date - timedelta(days=91), today, sites
        )
        levels_data = self.get_bq_data(
            "levels",
            run_date - timedelta(days=31),
            today,
            sites,
            date_key="DATETIME",
            reservoir_key="RESERVOIR_NAME",
            dt_cast_key="timestamp",
        )

        # levels
        levels_data = levels_data.rename(
            columns={"DATETIME": "date", "RESERVOIR_NAME": "site"}
        )
        levels_data["date"] = pd.to_datetime(levels_data["date"], utc=True)
        levels_data["step"] = timedelta(days=0)
        levels_arr = (
            levels_data.groupby(["date", "step", "site"])
            .nth(0)
            .drop(columns=["WATER_LEVEL", "FULL_WATER_LEVEL", "RESERVOIR_UUID"])
            .rename(columns={"WATER_VOLUME": "targets_WATER_VOLUME"})
            .to_xarray()
        )

        # precip
        precip_data = precip_data.drop(columns=["timestamp"]).rename(
            columns={"value": "chirps_precip", "reservoir": "site"}
        )
        precip_data["date"] = pd.to_datetime(precip_data["date"], utc=True)
        precip_data["step"] = timedelta(days=0)
        precip_arr = precip_data.groupby(["date", "step", "site"]).nth(0).to_xarray()

        # forcast data
        tigge_tp_arr = explode_and_step(
            forecast_data[["reservoir", "date", "values_precip"]],
            "values_precip",
            "tigge_tp",
        ).to_xarray()
        tigge_t2m_arr = explode_and_step(
            forecast_data[["reservoir", "date", "values_temp"]],
            "values_temp",
            "tigge_t2m",
        ).to_xarray()

        # TrigDoY
        doy_unit = SynthTrigDoY()

        doy_arr = doy_unit.build(
            start_datetime=pd.to_datetime(run_date - timedelta(days=91), utc=True),
            end_datetime=pd.to_datetime(today + timedelta(days=16), utc=True),
            sin_or_cos=["sin", "cos"],
            site_mapper=dict(zip(sites, sites)),
            start_step=0,
            end_step=91,
            step_size=1,
            data_unit_name="doy",
        ).rename({"global_sites": "site", "steps": "step"})

        return xr.merge([levels_arr, precip_arr, tigge_tp_arr, tigge_t2m_arr, doy_arr])

    def push_results(self, df):

        errors = self.client.insert_rows_json(
            self.tables["prediction"], df.to_dict(orient="records")
        )

        self.check_errors(errors)

        return True


def xr_to_sample(ds, cfg, ini_date, run_date, sites):

    ds["date"] = pd.to_datetime(ds["date"], utc=True)

    hist = ds[
        [
            var
            for var in cfg["dataset_parameters"]["historic_variables"]
            if var not in cfg["dataset_parameters"]["target_var"]
        ]
    ].sel(
        {
            "site": sites,
            "date": pd.to_datetime(
                pd.date_range(ini_date, run_date, freq="1D").date, utc=True
            ),
            "step": timedelta(days=0),
        }
    )

    target = ds[cfg["dataset_parameters"]["target_var"]].sel(
        {
            "site": sites,
            "date": pd.to_datetime(
                pd.date_range(ini_date, run_date, freq="1D").date, utc=True
            ),
            "step": timedelta(days=0),
        }
    )

    forecast = ds[[var for var in cfg["dataset_parameters"]["forecast_variables"]]].sel(
        {
            "site": sites,
            "date": run_date,
            "step": [timedelta(days=ii) for ii in range(0, 15)],
        }
    )

    future = ds[[var for var in cfg["dataset_parameters"]["future_variables"]]].sel(
        {
            "site": sites,
            "date": run_date,
            "step": [timedelta(days=ii) for ii in range(15, 91)],
        }
    )

    hist_dict = hist.to_array().transpose("site", "variable", "date").to_dict()
    target_dict = target.to_array().transpose("site", "variable", "date").to_dict()
    forecast_dict = forecast.to_array().transpose("site", "variable", "step").to_dict()
    future_dict = future.to_array().transpose("site", "variable", "step").to_dict()

    hist_dict = reshape_data(hist_dict)
    forecast_dict = reshape_data(forecast_dict)
    future_dict = reshape_data(future_dict)
    target_dict = reshape_data(target_dict)

    combined_dict = combine_data(hist_dict, target_dict, forecast_dict, future_dict)

    sample_data = {run_date.isoformat()[0:10]: combined_dict}

    return sample_data


def call_inference(sample, basin_network, url=None):

    if url is None:
        url = os.environ.get("inference_url")

    headers = {"Content-Type": "application/json"}

    call_sample = {"instances": [sample]}

    url = os.path.join(url, basin_network)

    r = requests.post(
        url, headers=headers, data=json.dumps(call_sample).encode("utf-8")
    )

    return r.status_code, r.text


def local_inference(today, basin_networks):

    if basin_networks is None:
        basin_networks = json.loads(os.environ.get("basin_networks"))

    model_dir = os.environ.get("model_dir")

    # 1. get most recent hist+forecast+levels data, compare to current preds

    logger.info(f"running basin networks: {basin_networks}")
    client = BQInfClient()
    dates_df = client.get_dates_df()
    logger.info("Got date df")

    # 2. for basin_network in basin_networks: if current_preds.date < all_dates, run inference, post
    msg = {}

    for basin_network in basin_networks:

        sites = json.load(open(os.path.join(model_dir, f"{basin_network}_sites.json")))
        cfg = json.load(open(os.path.join(model_dir, f"{basin_network}_cfg.json")))

        # check all are updateable
        if dates_df.loc[sites, "updateable"].all():

            # build model
            ctx = Context(
                model_name=f"{basin_network}",
                model_dir=model_dir,
                manifest={
                    "model": {
                        "serializedFile": f"{basin_network}.pt",
                    }
                },
                batch_size=8,
                gpu=None,
                mms_version=0.1,
                limit_max_image_pixels=True,
            )

            inst = H2OxHandler()
            inst.initialize(ctx)

            # inference
            # best date - data can be updated to here
            best_date = pd.to_datetime(
                dates_df.loc[sites, "constraint"].min(), utc=True
            )
            ini_date = pd.to_datetime(best_date - timedelta(days=91), utc=True)
            logger.info(f"Building sample for {basin_network} for {best_date}")

            ds = client.bq_to_xr(best_date, today, cfg, sites)
            sample = xr_to_sample(ds, cfg, ini_date, best_date, sites)

            logger.info(f"inferring sample for {basin_network} for {best_date}")
            result = inst.handle(sample, ctx)

            logger.info(f"got resultfor {basin_network} for {best_date}")

            result_df = postprocess(result, sites)

            # check nans in results
            for col in ["upper", "lower", "forecast"]:
                assert (
                    not result_df[col].apply(lambda ll: np.isnan(ll).any()).any()
                ), f"{col} has nans"

            if str(os.environ.get("PUSH")).lower() == "true":
                client.push_results(result_df)

                logger.info(
                    f"pushed {len(result_df)} records to results for for {basin_network} for {best_date}"
                )
            else:
                logger.info(
                    f"sim-pushed {len(result_df)} records to results for for {basin_network} for {best_date}"
                )

            msg[basin_network] = f"pushed local: {len(result_df)}"

        else:
            logger.info(f"{basin_network} up-to-date!")
            msg[basin_network] = "up-to-date"

    return msg


def remote_inference(today, basin_networks, url):

    if basin_networks is None:
        basin_networks = json.loads(os.environ.get("basin_networks"))

    # 1. get most recent hist+forecast+levels data, compare to current preds

    logger.info(f"running basin networks: {basin_networks}")
    client = BQInfClient()
    dates_df = client.get_dates_df()
    logger.info("Got date df")

    # 2. for basin_network in basin_networks: if current_preds.date < all_dates, run inference, post
    msg = {}

    for basin_network in basin_networks:
        sites = json.load(
            open(os.path.join(os.getcwd(), "data", f"{basin_network}.json"))
        )
        cfg = json.load(
            open(os.path.join(os.getcwd(), "data", f"{basin_network}_cfg.json"))
        )

        # check all are updateable
        if dates_df.loc[sites, "updateable"].all():

            # inference
            # best date - data can be updated to here
            best_date = pd.to_datetime(
                dates_df.loc[sites, "constraint"].min(), utc=True
            )
            ini_date = pd.to_datetime(best_date - timedelta(days=91), utc=True)
            logger.info(f"Building sample for {basin_network} for {best_date}")

            ds = client.bq_to_xr(best_date, today, cfg, sites)
            sample = xr_to_sample(ds, cfg, ini_date, best_date, sites)

            logger.info(f"inferring sample for {basin_network} for {best_date}")
            code, result = call_inference(sample, basin_network, url=url)
            result = json.loads(result)["predictions"]

            logger.info(f"got result {code} for {basin_network} for {best_date}")
            if code == 200:
                result_df = postprocess(result, sites)

                client.push_results(result_df)

                logger.info(
                    f"pushed {len(result_df)} records to results for for {basin_network} for {best_date}"
                )
                #
                msg[basin_network] = f"{code}: {len(result_df)}"

            else:
                msg[basin_network] = f"{code}"

        else:
            logger.info(f"{basin_network} up-to-date!")
            msg[basin_network] = "up-to-date"

    return msg


def postprocess(result, sites):

    result = result[0]

    records = []

    for keydate in result.keys():

        for ii_s, site in enumerate(sites):

            record = {
                "date": keydate,
                "reservoir": site,
                "upper": np.array(result[keydate]["upper"])[:, ii_s].tolist(),
                "lower": np.array(result[keydate]["lower"])[:, ii_s].tolist(),
                "forecast": np.array(result[keydate]["mean"])[:, ii_s].tolist(),
                "forecast_variable": "VOLUME",
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
            }

            records.append(record)

    return pd.DataFrame(records)
