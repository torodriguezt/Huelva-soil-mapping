# Spatial level and new-sample exceedance probabilities on the prediction grid.
import argparse
import time

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, cKDTree
from threadpoolctl import threadpool_limits

from blmc.config import DIR_MAPS, FOCUS, MINOR, NGR_USES, SELECTED_MODEL
from blmc.io import group_index, load_draws, load_fit, mean_design, read_grid, save_json, thresholds
from blmc.model import local_scales
from blmc.predict import conditional_predict

KEYS = ["beta", "loading", "Sigma", "rho", "F", "group_variance", "nu"]


def predict_grid(fit, grid, draws_per_chain, rng, subsets=None):
    spec, data = load_fit(fit)
    elements, regulated = spec["elements"], list(data["regulated_indices"])
    events = {"union": regulated, **{label: [elements.index(e) for e in names] for label, names in (subsets or {}).items()}}
    xy = grid[["x", "y"]].to_numpy() / 1000 - data["origin"] / 1000
    X = mean_design(grid, spec.get("mean_design", "ngr"))
    n, q = len(grid), len(elements)
    thr = np.full((n, q), np.inf)
    thr[:, regulated] = thresholds(grid, elements, regulated, data)
    group = group_index(grid)
    element_group = spec.get("element_group")
    masks ={use: grid.ngr_use.eq(use).to_numpy() for use in NGR_USES}

    total, squares = np.zeros((n, q)), np.zeros((n, q))
    exceed = {"level": np.zeros((n, q)), "sample": np.zeros((n, q))}
    union = {(label, name): np.zeros(n) for label in events for name in ["level", "sample"]}
    areas, S = [], 0
    for chain in range(1, 5):
        _, arrays = load_draws(fit, chain, draws_per_chain, KEYS)
        for s in range(draws_per_chain):
            draw = {key: arrays[key][s] for key in ["beta", "loading", "Sigma", "rho", "F"]}
            prediction = conditional_predict(draw, data["coords"], xy, X, chunk_size=4000)
            F = prediction["mean_f"] + np.sqrt(prediction["var_f"]) * rng.standard_normal(prediction["mean_f"].shape)
            mu = X @ draw["beta"] + F @ draw["loading"]
            nu = arrays["nu"][s]
            w = rng.gamma(nu / 2, 2 / nu, size=n)
            scale = local_scales(arrays["group_variance"][s], group, element_group)
            local = (rng.standard_normal((n, q)) @ np.linalg.cholesky(draw["Sigma"]).T) * np.sqrt(scale / w[:, None])
            total += mu
            squares += mu**2
            row = {}
            for name, values in [("level", mu), ("sample", mu + local)]:
                hit = values > thr
                exceed[name] += hit
                row.update({f"{name}_{elements[j]}": hit[:, j].mean() for j in regulated})
                for label, indices in events.items():
                    any_hit = hit[:, indices].any(axis=1)
                    union[(label, name)] += any_hit
                    row[f"{name}_{label}"] = any_hit.mean()
                    for use, mask in masks.items():
                        row[f"{name}_{label}_{use}"] = any_hit[mask].mean()
            areas.append(row)
            S += 1
        print(f"chain {chain}: {S} draws", flush=True)

    mean = total / S
    columns = {}
    for j, e in enumerate(elements):
        columns[f"level_log_mean_{e}"] = mean[:, j] * data["scale"][j] + data["center"][j]
        columns[f"level_log_sd_{e}"] = np.sqrt(np.maximum(squares[:, j] / S - mean[:, j] ** 2, 0)) * data["scale"][j]
        columns[f"level_mg_kg_{e}"] = np.exp(columns[f"level_log_mean_{e}"])
    for j in regulated:
        columns[f"p_level_{elements[j]}"] = exceed["level"][:, j] / S
        columns[f"p_sample_{elements[j]}"] = exceed["sample"][:, j] / S
    for (label, name), hits in union.items():
        columns[f"p_{name}_{label}"] = hits / S
    out = pd.concat([grid[["x", "y", "land_use", "ngr_use"]].reset_index(drop=True), pd.DataFrame(columns)], axis=1)
    return out, pd.DataFrame(areas), S


def main(args):
    started = time.time()
    folder = DIR_MAPS / args.fit
    folder.mkdir(parents=True, exist_ok=True)
    _, data = load_fit(args.fit)
    grid = read_grid()
    samples = data["coords"] * 1000 + data["origin"]
    out, areas, S = predict_grid(args.fit, grid, args.draws_per_chain, np.random.default_rng(args.seed),
                                 {"focus": FOCUS, "minor": MINOR})
    out["nearest_sample_m"] = cKDTree(samples).query(grid[["x", "y"]].to_numpy())[0]
    out["outside_sample_convex_hull"] = Delaunay(samples).find_simplex(grid[["x", "y"]].to_numpy()) < 0
    out.to_csv(folder / "grid_predictions.csv", index=False)
    areas.to_csv(folder / "area_draws.csv", index=False)
    save_json(folder / "metadata.json", dict(fit=args.fit, posterior_draws=S, grid_cells=len(grid), seed=args.seed,
                                             elapsed_seconds=time.time() - started))
    print(f"saved {folder} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", default=SELECTED_MODEL)
    parser.add_argument("--draws-per-chain", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261023)
    with threadpool_limits(limits=1):
        main(parser.parse_args())
