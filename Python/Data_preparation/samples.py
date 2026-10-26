# Samples of the three surveys in mg/kg and the NGR table.
import re

import numpy as np
import pandas as pd

from blmc.config import DIR_DATA, FILE_SITES, N_HUELVA, NGR
from blmc.land_use import read_utm

TO_MG_KG = {"ppm": lambda v: v, "ppb": lambda v: v / 1000, "pct": lambda v: v * 1e4}


def read_site(site):
    layer = read_utm(FILE_SITES[site])
    columns = [c for c in layer.columns if re.match(r"^[A-Z][a-z]?_", c)]
    table = pd.DataFrame({"site": site, "id": np.arange(1, len(layer) + 1),
                          "x": layer.geometry.x.to_numpy(), "y": layer.geometry.y.to_numpy()})
    for column in columns:
        values = pd.to_numeric(layer[column], errors="coerce").astype(float)
        unit = "ppm" if "ppm" in column else "ppb" if "ppb" in column else "pct"
        table[column.split("_")[0]] = TO_MG_KG[unit](values.where(values > 0))
    return table


def main():
    DIR_DATA.mkdir(parents=True, exist_ok=True)
    samples = pd.concat([read_site(site) for site in FILE_SITES], ignore_index=True)
    elements = [c for c in samples.columns if c not in ("site", "id", "x", "y")]
    huelva = samples[samples.site == "Huelva"]
    assert len(huelva) == N_HUELVA and set(NGR.element) <= set(elements) and not huelva[NGR.element].isna().any().any()

    # ties at the minimum flag a detection floor
    long = samples.melt(id_vars=["site"], value_vars=elements, var_name="element", value_name="mg_kg")
    quality = long.groupby(["site", "element"], sort=True).mg_kg.agg(
        n="size", n_missing=lambda v: v.isna().sum(), min_mg_kg="min", median_mg_kg="median", max_mg_kg="max",
        n_at_min=lambda v: (v == v.min()).sum(), n_at_max=lambda v: (v == v.max()).sum()).reset_index()
    quality["pct_at_min"] = 100 * quality.n_at_min / quality.n

    samples.to_csv(DIR_DATA / "samples.csv", index=False)
    quality.to_csv(DIR_DATA / "element_quality.csv", index=False)
    NGR.to_csv(DIR_DATA / "ngr.csv", index=False)
    print(f"samples: {samples.site.value_counts().to_dict()}; {len(elements)} elements")


if __name__ == "__main__":
    main()
