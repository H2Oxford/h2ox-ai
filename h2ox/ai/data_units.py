import abc
from datetime import datetime, timedelta
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger


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
