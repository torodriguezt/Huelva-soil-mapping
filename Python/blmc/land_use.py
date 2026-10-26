# Distances to the land-use classes and assigned land use.
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

from blmc.config import CRS_UTM


def read_utm(path):
    layer = gpd.read_file(path)
    layer.geometry = layer.geometry.force_2d()
    return layer.to_crs(CRS_UTM)


def land_use_distances(points, path, tolerance_m=0.01):
    layer = read_utm(path)
    kinds = set(layer.geom_type)
    if kinds <= {"Polygon", "MultiPolygon"}:
        boundary = shapely.union_all(shapely.make_valid(layer.geometry.to_numpy()))
        return shapely.distance(shapely.points(points), boundary)
    if kinds != {"LineString"}:
        raise ValueError(f"{path}: land-use features must be polygons or connectors")

    distance = layer["distance"].to_numpy(dtype=float)
    if np.any(np.abs(distance - layer.length.to_numpy()) > np.maximum(tolerance_m, distance * 1e-6)):
        raise ValueError(f"{path}: connector distance does not match its length")
    origins = np.array([line.coords[0][:2] for line in layer.geometry])
    gap, index = cKDTree(origins).query(points, k=2)
    if np.any(gap[:, 0] > tolerance_m):
        raise ValueError(f"{path}: {np.sum(gap[:, 0] > tolerance_m)} points without a connector")
    if np.any(gap[:, 1] <= tolerance_m) or len(np.unique(index[:, 0])) < len(points):
        raise ValueError(f"{path}: ambiguous connector origins")
    return distance[index[:, 0]]


def assign_land_use(distances, ngr_use, tolerance_m=1e-5):
    classes = np.array(distances.columns)
    values = distances.to_numpy()
    categories = np.array([ngr_use[c] for c in classes])
    nearest = values.argmin(axis=1)
    minimum = values[np.arange(len(values)), nearest]
    tied = np.abs(values - minimum[:, None]) <= tolerance_m
    conflicting = [i for i in np.flatnonzero(tied.sum(axis=1) > 1) if len(set(categories[tied[i]])) > 1]
    if conflicting:
        raise ValueError(f"equally close land uses imply different NGR uses at {len(conflicting)} locations")
    zero = minimum <= tolerance_m
    return pd.DataFrame({"land_use": classes[nearest], "ngr_use": categories[nearest], "land_use_distance_m": minimum,
                         "land_use_method": np.where(zero, "zero_distance", "nearest_proxy"),
                         "land_use_n_ties": tied.sum(axis=1),
                         "ngr_use_basis": np.where(zero, "land_use_at_zero_distance", "nearest_land_use_proxy")})
