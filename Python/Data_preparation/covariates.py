# Land use of the Huelva samples and grid points.
import numpy as np
import pandas as pd

from blmc.config import DIR_DATA, DIR_LAND_USE_GRID, DIR_LAND_USE_SITES, FILE_GRID, LAND_USE
from blmc.land_use import assign_land_use, land_use_distances, read_utm


def main():
    samples = pd.read_csv(DIR_DATA / "samples.csv")
    sites = samples.loc[samples.site == "Huelva", ["id", "x", "y"]].reset_index(drop=True)
    grid = read_utm(FILE_GRID)
    points = grid.geometry if set(grid.geom_type) == {"Point"} else grid.geometry.centroid
    grid = pd.DataFrame({"x": points.x.to_numpy(), "y": points.y.to_numpy()})

    raw_sites = pd.DataFrame({cls: land_use_distances(sites[["x", "y"]].to_numpy(), DIR_LAND_USE_SITES / files[1])
                              for cls, files in LAND_USE.items()})
    raw_grid = pd.DataFrame({cls: land_use_distances(grid[["x", "y"]].to_numpy(), DIR_LAND_USE_GRID / files[2])
                             for cls, files in LAND_USE.items()})

    # log distances standardized with the sites
    columns_sites, columns_grid = {}, {}
    for cls in LAND_USE:
        log_sites, log_grid = np.log1p(raw_sites[cls]), np.log1p(raw_grid[cls])
        sd = log_sites.std(ddof=1)
        sd = sd if np.isfinite(sd) and sd > 0 else 1.0
        columns_sites[f"distance_{cls}_m"], columns_grid[f"distance_{cls}_m"] = raw_sites[cls], raw_grid[cls]
        columns_sites[f"d_{cls}"] = (log_sites - log_sites.mean()) / sd
        columns_grid[f"d_{cls}"] = (log_grid - log_sites.mean()) / sd

    ngr_use = {cls: values[0] for cls, values in LAND_USE.items()}
    out_sites = pd.concat([sites, pd.DataFrame(columns_sites), assign_land_use(raw_sites, ngr_use)], axis=1)
    out_grid = pd.concat([grid, pd.DataFrame(columns_grid), assign_land_use(raw_grid, ngr_use)], axis=1)
    assert not out_sites.isna().any().any() and not out_grid.isna().any().any()
    out_sites.to_csv(DIR_DATA / "covariates_obs.csv", index=False)
    out_grid.to_csv(DIR_DATA / "covariates_grid.csv", index=False)
    print(f"land use: {len(out_sites)} samples {out_sites.ngr_use.value_counts().to_dict()}, {len(out_grid)} grid points; "
          f"nearest-use proxies: {np.sum(out_sites.land_use_method == 'nearest_proxy')} samples, "
          f"{np.sum(out_grid.land_use_method == 'nearest_proxy')} grid points")


if __name__ == "__main__":
    main()
