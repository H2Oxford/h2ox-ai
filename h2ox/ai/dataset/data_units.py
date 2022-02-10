import abc
from datetime import datetime, timedelta
from pydoc import locate
from typing import Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from google.cloud import bigquery
from loguru import logger
from dask.diagnostics import ProgressBar

from h2ox.ai.dataset.xr_reducer import XRReducer


class DataUnit(abc.ABC):
    """An abstract data "unit" class for h2ox-ai."""

    @abc.abstractmethod
    def build(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
        site_mapper: Dict[str, str],
        variable_keys: List[str],
        data_unit_name: str,
        **kwargs,
    ) -> xr.DataArray:
        """Build the dataunit.

        Args:
            start_datetime (datetime): dataset start datetime
            end_datetime (datetime): dataset end datetime
            site_keys (Union[List[str],Dict[str,str]]): dataset keys, either list of global key names, or a dict mapping {data_unit:global} names
            variable_keys (List[str]): variable key list
            **kwargs

        returns:
            xr.DataArray
        """


class CSVDataUnit(DataUnit):
    """A dataclass for building dataframes from csv source files."""

    def build(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
        site_mapper: Dict[str, str],
        variable_keys: Union[List[str], Dict[str, str]],
        data_unit_name: str,
        data_path: str,
        date_col: str,
        site_col: str,
        **kwargs,
    ) -> xr.DataArray:
        """Build the dataunit. Data must be indexed by datetime, site and variable.

        Args:
            start_datetime (datetime): dataset start datetime
            end_datetime (datetime): dataset end datetime
            site_keys (Union[List[str],Dict[str,str]]): dataset keys, either list of global key names, or a dict mapping {data_unit:global} names
            variable_keys (Union[List[str],Dict[str,str]]): variable key list. If list, remap keys to values.
            data_unit_name (str): unique name for this data unit as prefix
            data_path (str): path to xarray-compatible datafile
            site_col (str): the column with the site names

        Returns:
            xr.DataArray

        """
        logger.info(f"{data_unit_name} - Building at {data_path}")

        # remap keys to dict
        if isinstance(variable_keys, list):
            remap_keys = dict(
                zip(variable_keys, [f"{data_unit_name}_{kk}" for kk in variable_keys])
            )
        else:
            remap_keys = {
                kk: f"{data_unit_name}_{vv}" for kk, vv in variable_keys.items()
            }

        # load the dataframe
        df = pd.read_csv(data_path)

        # map unique site names
        df["global_sites"] = df[site_col].map(site_mapper)
        chosen_sites = site_mapper.values()

        # cast datetime column to day
        df[date_col] = pd.to_datetime(df[date_col]).dt.floor("d")

        # filter and cast to xarray
        array = (
            df.loc[
                (df[date_col] >= start_datetime)
                & (df[date_col] <= end_datetime)
                & (df["global_sites"].isin(chosen_sites)),
                list(remap_keys.keys()) + ["global_sites", date_col],
            ]
            .set_index(["global_sites", date_col])
            .to_xarray()
        )

        # remap variable and coordinate names
        remap_keys[date_col] = "date"  # set a common date name
        array = array.rename(**remap_keys)

        # add steps dimension
        steps_idx = pd.Series([timedelta(days=ii) for ii in [0]], name="steps")
        array = array.expand_dims({"steps": steps_idx})

        return array


class ZRSpatialDataUnit(DataUnit):

    """A dataunit for building dataframes from spatial zarr archives."""

    def build(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
        site_mapper: Dict[str, str],
        variable_keys: List[str],
        data_unit_name: str,
        gdf_path: str,
        site_col: str,
        datetime_col: str,
        z_address: str,
        start_step: int,
        end_step: int,
        step_size: int,
        steps_key: Optional[str],
        zarr_mapper: Optional[str],
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        **kwargs,
    ) -> xr.DataArray:

        """Build the dataunit by reducing a zarr archive with a geopandas gdf.

        Args:
            start_datetime (datetime): dataset start datetime
            end_datetime (datetime): dataset end datetime
            site_keys (List[str]): dataset key list, where keys are local site names.
            variable_keys (List[str]): variable key list
            data_unit_name (str): unique name for this data unit as prefix
            gdf_path (str): path to a gpd.GeoDataFrame
            gdf_site_col (str): column in the gpd.GeoDataFrame containing the site names
            datetime_col (str): name of the datetime coordinate
            z_address (str): Address of the zarr archive
            lat_col (str): name of the latitude coordinate
            lon_col (str): name of the longitude coodinate
            zarr_mapper: Optional[str]

        Returns:
            xr.DataArray

        """

        if all([(ii is not None) for ii in [start_step, end_step, step_size]]):
            steps = range(start_step, end_step, step_size)
        else:
            steps = None

        logger.info(
            f"{data_unit_name} - Building; reducing {z_address} over {gdf_path}"
        )

        # load the gdf, map the site names, and set the index
        gdf = gpd.read_file(gdf_path)

        # map unique site names
        gdf["global_sites"] = gdf[site_col].map(site_mapper)
        chosen_sites = site_mapper.values()  # global name
        gdf = gdf.set_index("global_sites")

        # remap keys to dict
        # if isinstance(variable_keys, list):
        #     remap_keys = dict(
        #         zip(variable_keys, [f"{data_unit_name}_{kk}" for kk in variable_keys])
        #     )
        # else:
        #     remap_keys = {
        #         kk: f"{data_unit_name}_{vv}" for kk, vv in variable_keys.items()
        #     }

        # get the mapper
        if zarr_mapper is None:
            z_mapper = locate("h2ox.ai.dataset.utils.null_mapper")()
        else:
            z_mapper = locate(zarr_mapper)()

        # map the zxr
        zx_arr = xr.open_zarr(z_mapper(z_address))

        # reduce the xarray object for each variable - geometry
        reduced_var_arrays: Dict[str, xr.DataArray] = {}
        for variable in variable_keys:
            reduced_geom_arrays = {}
            ds = XRReducer(
                array=zx_arr[variable], lat_variable=lat_col, lon_variable=lon_col
            )

            # for each geometry in the gdf
            for idx, row in gdf.loc[gdf.index.isin(chosen_sites)].iterrows():
                reduced_geom_arrays[idx] = ds.reduce(
                    row["geometry"], start_datetime, end_datetime
                )

            reduced_var_arrays[variable] = xr.concat(
                list(reduced_geom_arrays.values()),
                pd.Index(list(reduced_geom_arrays.keys()), name="global_sites"),
            )
            reduced_var_arrays[variable].name = f"{data_unit_name}_{variable}"

        # merge back along the variable dimension
        array = xr.merge(list(reduced_var_arrays.values()))

        # force daily time dimension
        array = array.resample({datetime_col: "1D"}).mean(datetime_col)

        if steps_key is None:
            steps_key = "steps"

        # check if steps exists
        if steps_key not in [coord for coord in array.coords]:
            # create a 0th step
            steps_idx = pd.TimedeltaIndex(
                [timedelta(days=ii) for ii in [0]], name="steps"
            )
            array = array.expand_dims({steps_key: steps_idx})
        else:
            # convert integer steps into TimeDelta objects
            if steps is not None:
                array = array.sel(
                    {steps_key: pd.TimedeltaIndex([timedelta(days=ii) for ii in steps])}
                )

        # rename for consistency
        array = array.rename({datetime_col: "date", steps_key: "steps"})

        # Compute the dask objects and track progress via progress bar
        logger.info(f"{data_unit_name} - Dask --> In Memory;")
        pbar = ProgressBar()
        pbar.register()
        array = array.compute()
        pbar.unregister()

        return array


class BQDataUnit(DataUnit):

    """A dataunit for building dataframes from BigQuery archives."""

    def build(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
        site_mapper: Dict[str, str],
        variable_keys: List[str],
        data_unit_name: str,
        site_col: str,
        datetime_col: str,
        bq_address: str,
        **kwargs,
    ) -> xr.DataArray:

        """Build the dataunit by reducing a zarr archive with a geopandas gdf.

        Args:
            start_datetime (datetime): dataset start datetime
            end_datetime (datetime): dataset end datetime
            site_keys (List[str]): dataset key list, where keys are global site names.
            variable_keys (List[str]): variable key list
            data_unit_name (str): unique name for this data unit as prefix
            bq_address (str): address of the bigquery table

        Returns:
            xr.DataArray

        """

        logger.info(f"{data_unit_name} - Building; querying {bq_address}")

        client = bigquery.Client()

        query_keys = site_mapper.keys()

        sites_query = '("' + '","'.join(query_keys) + '")'

        # construct query
        Q = f"""
            SELECT {', '.join(variable_keys)}
            FROM `{bq_address}`
            WHERE
              {site_col} in {sites_query}
              AND {datetime_col} <= "{end_datetime.isoformat()[0:10]}"
              AND {datetime_col} >= "{start_datetime.isoformat()[0:10]}"
        """

        # execute query
        df = client.query(Q).result().to_dataframe()

        df[datetime_col] = pd.to_datetime(df[datetime_col]).dt.floor("D")

        df["global_sites"] = df[site_col].map(site_mapper)

        df = df.set_index(["global_sites", datetime_col])

        # drop duplicate index
        df = df[~df.index.duplicated(keep="first")]

        array = df.to_xarray()

        # remap variable and coordinate names
        if isinstance(variable_keys, list):
            remap_keys = dict(
                zip(variable_keys, [f"{data_unit_name}_{kk}" for kk in variable_keys])
            )
        else:
            remap_keys = {
                kk: f"{data_unit_name}_{vv}" for kk, vv in variable_keys.items()
            }

        remap_keys[datetime_col] = "date"  # set a common date name
        array = array.rename(**remap_keys)

        # reset datetime dtype
        array["date"] = pd.to_datetime(array["date"].data)

        # add steps dimension
        steps_idx = pd.TimedeltaIndex([timedelta(days=ii) for ii in [0]], name="steps")
        array = array.expand_dims({"steps": steps_idx})

        return array


class SynthTrigDoY(DataUnit):

    """A dataunit for building synthetic day-of-year data."""

    def build(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
        site_mapper: Dict[str, str],
        sin_or_cos: Union[str, List[str]],
        data_unit_name: str,
        start_step: int,
        end_step: int,
        step_size: int,
        **kwargs,
    ) -> xr.DataArray:

        steps = range(start_step, end_step, step_size)

        logger.info(f"{data_unit_name} - Building; synth DoY with {len(steps)} steps")

        if isinstance(sin_or_cos, str):
            sin_or_cos = [sin_or_cos]

        arrays = []
        for trig_iden in sin_or_cos:

            if trig_iden == "sin":
                trig_fn = np.sin
            elif trig_iden == "cos":
                trig_fn = np.cos
            else:
                raise ValueError("sin_or_cos must be one of (or list of) 'sin', 'cos'")

            idx = pd.date_range(start_datetime, end_datetime, freq="d")
            idx.name = "date"

            cols = pd.TimedeltaIndex([timedelta(days=ii) for ii in steps], name="steps")

            df = pd.DataFrame(index=idx, columns=cols)

            for cc in df.columns:
                df[cc] = df.index + cc
                df[cc] = trig_fn(df[cc].dt.dayofyear / 365 * 2 * np.pi)

            arr = df.unstack().to_xarray()
            arr.name = f"{data_unit_name}_{trig_iden}"
            arrays.append(arr)

        # apply to all sites

        global_sites = list(site_mapper.values())
        array = xr.merge(arrays)

        return array.expand_dims({"global_sites": global_sites})
