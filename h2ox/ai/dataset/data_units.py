import abc
import datetime
from typing import Dict, List, Optional, Union

import pandas as pd
import xarray as xr
from loguru import logger

from h2ox.ai.dataset.xr_reducer import XRReducer


class DataUnit(abc.ABC):

    """An abstract data "unit" class for h2ox-ai."""

    @abc.abstractmethod
    def build(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
        site_keys: Union[List[str], Dict[str, str]],
        variable_keys: List[str],
        data_unit_name: [str],
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
        site_keys: Union[List[str], Dict[str, str]],
        variable_keys: Union[List[str], Dict[str, str]],
        data_unit_name: [str],
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
        if isinstance(site_keys, dict):
            df["global_sites"] = df[site_col].map(site_keys)
            chosen_sites = site_keys.values()
        else:
            df["global_sites"] = df[site_col]
            chosen_sites = site_keys

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

        return array


class ZRSpatialDataUnit(DataUnit):

    """A dataunit for building dataframes from spatial zarr archives."""

    def build(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
        site_keys: Union[List[str], Dict[str, str]],
        variable_keys: List[str],
        data_unit_name: [str],
        gdf_path: str,
        gdf_site_col: str,
        z_address: str,
        step: Optional[List[int]],
        zarr_mapper: Optional[str],
        lat_col: str = "latitude",
        lon_col: str = "longitude",
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
            z_address (str): Address of the zarr archive
            lat_col (str): name of the latitude coordinate
            lon_col (str): name of the longitude coodinate
            zarr_mapper: Optional[str]

        Returns:
            xr.DataArray

        """

        # load the gdf, map the site names, and set the index
        gdf = gpd.read_file(gdf_path)

        # map unique site names
        if isinstance(site_keys, dict):
            gdf["global_sites"] = gdf[site_col].map(site_keys)
            chosen_sites = site_keys.values()  # global name
        else:
            gdf["global_sites"] = gdf[site_col]
            chosen_sites = site_keys

        gdf = gdf.set_index("global_sites")

        # remap keys to dict
        if isinstance(variable_keys, list):
            remap_keys = dict(
                zip(variable_keys, [f"{data_unit_name}_{kk}" for kk in variable_keys])
            )
        else:
            remap_keys = {
                kk: f"{data_unit_name}_{vv}" for kk, vv in variable_keys.items()
            }

        # get the mapper
        if zarr_mapp is None:
            z_mapper = locate("h2ox.ai.dataset.utils.null_mapper")
        else:
            z_mapper = locate(zarr_mapper)

        # map the zxr
        zx_arr = xr.open_zarr(z_mapper(z_address))

        reduced_var_arrays = {}

        for variable in variable_keys:

            reduced_geom_arrays = {}

            ds = XRReducer(
                array=zx_arr[variable], lat_variable=lat_col, lon_variable=lon_col
            )

            for idx, row in gdf.loc[gdf.index.isin(chosen_sites)].iterrows():

                reduced_geom_arrays[idx] = ds.reduce(
                    row["geom"], start_datetime, end_datetime
                )

            reduced_var_arrays[variable] = xr.concat(
                list(reduced_geom_arrays.values()),
                pd.Index(list(reduced_geom_arrays.keys()), name="site"),
            )
            reduced_var_arrays[variable].name = f"{data_unit_name}_{variable}""

        # merge back along the variable dimension
        return xr.merge(list(reduced_var_arrays.values()))


class BQDataUnit(DataUnit):

    """A dataunit for building dataframes from BigQuery archives."""

    def build(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
        site_keys: Union[List[str], Dict[str, str]],
        variable_keys: List[str],
        data_unit_name: [str],
        bq_address: str,
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

        client = bigquery.Client()

        # construct query
        Q = f"""
            SELECT *
            FROM `{bq_address}`
        """

        # execute query
        client.query(Q).result().to_dataframe()

        # cast to xarray df

        return array


class XRDataUnit(DataUnit):

    """A dataclass for building dataframes from xarray-compatible source files."""

    def build(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
        site_keys: Union[List[str], Dict[str, str]],
        variable_keys: List[str],
        data_unit_name: [str],
        data_path: str,
    ) -> xr.DataArray:

        """Build the dataunit. Data must be indexed by datetime, site and variable.

        Args:
            start_datetime (datetime): dataset start datetime
            end_datetime (datetime): dataset end datetime
            site_keys (List[str]): dataset key list, where keys are global site names.
            variable_keys (List[str]): variable key list
            data_unit_name (str): unique name for this data unit as prefix
            data_path (str): path to xarray-compatible datafile

        Returns:
            xr.DataArray

        """

        # load the xr data

        # check the variables etc.

        return array
