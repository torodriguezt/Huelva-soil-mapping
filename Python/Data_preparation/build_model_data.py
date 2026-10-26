# Standardized model input; --keep137 keeps SHS200G at row 137 instead of 93.
import argparse
import hashlib

import numpy as np
import pandas as pd

from blmc.config import CONTEXT, DIR_DATA, DIR_MODEL_DATA, ELEMENTS, GROUPS, REGULATED, USE6
from blmc.io import mean_design, save_json

MAX_FRACTION_AT_MINIMUM = 0.10   # screen against detection floors


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(keep137):
    excluded = 93 if keep137 else 137
    out = DIR_MODEL_DATA / ("keep137" if keep137 else "main")
    paths = {name: DIR_DATA / f"{name}.csv" for name in ["samples", "covariates_obs", "ngr"]}

    samples = pd.read_csv(paths["samples"])
    samples = samples[(samples.site == "Huelva") & (samples.id != excluded)].reset_index(drop=True)
    cov = pd.read_csv(paths["covariates_obs"]).set_index("id").loc[samples.id].reset_index()
    assert np.allclose(samples[["x", "y"]], cov[["x", "y"]], atol=1e-6, rtol=0)

    at_minimum = {e: float((samples[e] == samples[e].min()).mean()) for e in ELEMENTS}
    if max(at_minimum.values()) > MAX_FRACTION_AT_MINIMUM:
        raise ValueError(f"an element fails the detection screen: {at_minimum}")

    logy = np.log(samples[ELEMENTS].to_numpy())
    center, scale = logy.mean(axis=0), logy.std(axis=0, ddof=1)
    regulated = [ELEMENTS.index(e) for e in REGULATED]
    ngr = pd.read_csv(paths["ngr"]).set_index("element")
    threshold = np.full(logy.shape, np.inf)          # no NGR for the context elements
    for j in regulated:
        threshold[:, j] = ngr.loc[ELEMENTS[j], cov.ngr_use].to_numpy()
    origin = samples[["x", "y"]].min().to_numpy()

    data = dict(
        Y=(logy - center) / scale, center=center, scale=scale,
        X=mean_design(cov, "ngr"), X_labels=np.array(["intercept", "industrial", "urban"]),
        X_use6=mean_design(cov, "use6"), X_use6_labels=np.array(["intercept"] + USE6),
        coords=(samples[["x", "y"]].to_numpy() - origin) / 1000, origin=origin,
        ids=samples.id.to_numpy(), use=cov.ngr_use.to_numpy(),
        threshold_z=(np.log(threshold) - center) / scale, regulated_indices=np.array(regulated),
        variance_group_index=cov.land_use.map(GROUPS.index).to_numpy(),
        variance_group_labels=np.array(GROUPS))

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "data.npz", **data)
    save_json(out / "specification.json", dict(
        elements=ELEMENTS, regulated_elements=REGULATED, context_elements=CONTEXT,
        n=len(samples), excluded_ids=[excluded], fraction_at_minimum=at_minimum,
        sha256={path.name: sha256(path) for path in paths.values()}))
    print(f"{out.name}: {len(ELEMENTS)} elements, {len(samples)} sites, "
          f"largest fraction at the minimum {max(at_minimum.values()):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep137", action="store_true", help="keep SHS200G at row 137 and drop row 93")
    main(parser.parse_args().keep137)
