import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from blmc.config import DIR_DATA, DIR_FITS, GROUPS, USE6


def to_native(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): to_native(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [to_native(v) for v in x]
    return x


def save_json(path, x):
    path.write_text(json.dumps(to_native(x), ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_folder(folder):
    with np.load(folder / "data.npz", allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    return read_json(folder / "specification.json"), data


def load_fit(name):
    return load_folder(DIR_FITS / name)


def load_chains(name, keys, stride=1, chains=4):
    out = []
    for chain in range(1, chains + 1):
        with np.load(DIR_FITS / name / f"chain_{chain}.npz") as archive:
            out.append((chain, {key: archive[key][::stride] for key in keys}))
    return out


def load_draws(name, chain, n_draws, keys):
    with np.load(DIR_FITS / name / f"chain_{chain}.npz") as archive:
        index = np.linspace(0, len(archive["beta"]) - 1, n_draws, dtype=int)
        return index, {key: archive[key][index] for key in keys}


def read_grid(columns=("x", "y", "land_use", "ngr_use")):
    return pd.read_csv(DIR_DATA / "covariates_grid.csv", usecols=list(columns))


def lattice_cells(xy, spacing, reach=60.0):
    x = np.arange(xy[:, 0].min(), xy[:, 0].max() + spacing, spacing)
    y = np.arange(xy[:, 1].min(), xy[:, 1].max() + spacing, spacing)
    nodes = np.column_stack([a.ravel() for a in np.meshgrid(x, y)])
    distance, index = cKDTree(xy).query(nodes)
    return np.unique(index[distance <= reach])


def mean_design(table, design="ngr"):
    if design == "use6":
        columns = [table.land_use.eq(use) for use in USE6]
    else:
        columns = [table.ngr_use.eq("industrial"), table.ngr_use.eq("urban")]
    return np.column_stack([np.ones(len(table))] + columns).astype(float)


def group_index(table):
    return table.land_use.map(GROUPS.index).to_numpy()


def thresholds(table, elements, regulated, data):
    ngr = pd.read_csv(DIR_DATA / "ngr.csv").set_index("element")
    columns = []
    for j in regulated:
        limit = ngr.loc[elements[j], table.ngr_use].to_numpy()
        columns.append((np.log(limit) - data["center"][j]) / data["scale"][j])
    return np.column_stack(columns)
