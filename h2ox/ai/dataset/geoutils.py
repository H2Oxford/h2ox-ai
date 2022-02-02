"""
These geotools were adapted from tools developed for the Wave2Web Hackathon.
Developed with permission from license holders World Resources Institute and H2Ox (Lucas Kruitwagen, Chris Arderne, Thomas Lees, and Lisa Thalheimer)
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from area import area
from shapely import geometry
from shapely.affinity import affine_transform


def get_mask(lons, lats, geom, weighted=True):

    if lats[-1] < lats[0]:
        descending = True
    else:
        descending = False

    if ((lons.min() >= 0) and (lons.max() > 180)) and (geom.bounds[0] < 0):
        # pacific-centric projection 0-360deg
        # TODO: Greenwich intersections

        geotransform = [1, 0, 0, 1, 360, 0]  # [a, b, d, e, xoff, yoff]
        affine_geom = affine_transform(geom, geotransform)
        lower_lon_idx = np.where(lons <= affine_geom.bounds[0])[0].max()
        upper_lon_idx = np.where(lons >= affine_geom.bounds[2])[0].min()

    else:
        lower_lon_idx = np.where(lons <= geom.bounds[0])[0].max()
        upper_lon_idx = np.where(lons >= geom.bounds[2])[0].min()

    if descending:
        upper_lat_idx = np.where(lats < geom.bounds[1])[0].min()
        lower_lat_idx = np.where(lats > geom.bounds[3])[0].max()

    else:

        lower_lat_idx = np.where(lats < geom.bounds[1])[0].max()
        upper_lat_idx = np.where(lats > geom.bounds[3])[0].min()

    bounding_lons = lons[lower_lon_idx : upper_lon_idx + 1]
    bounding_lats = lats[lower_lat_idx : upper_lat_idx + 1]

    if ((lons.min() >= 0) and (lons.max() > 180)) and (bounding_lons.min() > 180):
        # pacific-centric projection 0-360deg
        bounding_lons = bounding_lons - 360

    llons, llats = np.meshgrid(bounding_lons, bounding_lats)

    min_x = llons[:-1, :-1].flatten()
    max_x = llons[:-1, 1:].flatten()
    min_y = llats[:-1, :-1].flatten()
    max_y = llats[1:, :-1].flatten()

    gdf = (
        gpd.GeoDataFrame(
            pd.DataFrame(dict(minx=min_x, maxx=max_x, miny=min_y, maxy=max_y)).apply(
                lambda row: geometry.box(**row), axis=1
            )
        )
        .rename(columns={0: "geometry"})
        .set_geometry("geometry")
    )

    lon_idx, lat_idx = np.meshgrid(
        range(lower_lon_idx, upper_lon_idx), range(lower_lat_idx, upper_lat_idx)
    )

    gdf["lon_idx"] = lon_idx.flatten()
    gdf["lat_idx"] = lat_idx.flatten()

    gdf["geoarea"] = gdf.geometry.apply(lambda geom: area(geometry.mapping(geom)))

    gdf["intersection_area"] = gdf.intersection(geom).apply(
        lambda geom: area(geometry.mapping(geom))
    )

    gdf["area_weight"] = gdf["intersection_area"] / gdf["geoarea"]

    mask = np.zeros((lats.shape[0], lons.shape[0]))

    mask[
        tuple(gdf["lat_idx"].values.tolist()), tuple(gdf["lon_idx"].values.tolist())
    ] = gdf["area_weight"].values

    extents = (
        gdf.bounds.minx.min(),
        gdf.bounds.maxx.max(),
        gdf.bounds.miny.min(),
        gdf.bounds.maxy.max(),
    )

    return (
        mask[lower_lat_idx:upper_lat_idx, lower_lon_idx:upper_lon_idx],
        (lower_lon_idx, lower_lat_idx, upper_lon_idx, upper_lat_idx),
        extents,
    )
