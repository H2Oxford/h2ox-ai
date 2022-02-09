import numpy as np
import xarray as xr
from shapely import geometry

from h2ox.ai.dataset.geoutils import get_mask


class XRReducer:
    def __init__(self, array, lat_variable="latitude", lon_variable="longitude"):

        self.lat_variable = lat_variable
        self.lon_variable = lon_variable

        self.array = array
        self.mask_geometry = None
        self.mask_array = None
        self.mask_bounds = None
        self.weighted = None
        self.mask_geom = None

    def mask(self, geom: geometry, weighted: bool = True) -> np.ndarray:

        mask, bounds, extents = get_mask(
            lons=self.array[self.lon_variable].values,
            lats=self.array[self.lat_variable].values,
            geom=geom,
            weighted=weighted,
        )

        if (
            self.array[self.lat_variable].values[-1]
            < self.array[self.lat_variable].values[0]
        ):
            # descending
            self.origin = "upper"
        else:
            self.origin = "lower"

        self.mask_array = xr.DataArray(
            mask,
            dims=(self.lat_variable, self.lon_variable),
            coords={
                self.lat_variable: self.array[self.lat_variable][
                    bounds[1] : bounds[3]
                ].values,
                self.lon_variable: self.array[self.lon_variable][
                    bounds[0] : bounds[2]
                ].values,
            },
        )
        self.mask_geom = geom
        self.mask_bounds = bounds
        self.mask_extents = extents
        self.mask_geometry = geom
        self.weighted = weighted

        return self.mask_array

    def clip(self, geom):

        if self.mask_bounds is None:
            self.mask(geom=geom)

        return self.array.isel(
            {
                self.lon_variable: slice(self.mask_bounds[0], self.mask_bounds[2]),
                self.lat_variable: slice(self.mask_bounds[1], self.mask_bounds[3]),
            }
        )

    def reduce(self, geom, start_dt, end_dt, op="mean", weighted=True):

        assert op in ["mean", "sum"], "'op' must be one of ['mean','sum']"
        assert (
            "time" in self.array.coords.keys()
        ), "'time' variable must be in coordinates"

        if self.mask_array is None or self.weighted != weighted:
            # check for no mask
            self.mask(geom, weighted=weighted)

        # check for new geom
        if self.mask_geom is not None:
            if not self.mask_geom.equals(geom):
                self.mask(geom, weighted=weighted)

        clipped_array = self.clip(geom)

        dims_order = ["time", self.lat_variable, self.lon_variable]
        dims_order = dims_order + [
            cc for cc in list(clipped_array.coords.keys()) if cc not in dims_order
        ]

        reduced_array = (
            clipped_array.sel(dict(time=slice(start_dt, end_dt)))
            .where(clipped_array > 0)
            .transpose(*dims_order)
            * self.mask_array
        ).sum(dim=(self.lat_variable, self.lon_variable))

        if op == "mean":
            return reduced_array / self.mask_array.sum()
        elif op == "sum":
            return reduced_array
