# Credible intervals of the exceedance areas, from joint simulation on a 240 m subgrid.
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError, cholesky, solve_triangular
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from blmc.config import DIR_AREAS, DIR_MAPS, FOCUS, INTERVAL, MINOR, NGR_USES, SELECTED_MODEL
from blmc.covariance import matern32
from blmc.io import group_index, lattice_cells, load_draws, load_fit, mean_design, read_grid, save_json, thresholds
from blmc.model import local_scales

KEYS = ["beta", "loading", "Sigma", "rho", "F", "group_variance", "nu"]


def jittered_cholesky(cov, jitter):
    for factor in [1, 10, 100, 1000]:
        try:
            return cholesky(cov + jitter * factor * np.eye(len(cov)), lower=True, check_finite=False), jitter * factor
        except LinAlgError:
            continue
    raise LinAlgError("the conditional covariance is not positive definite")


def simulate_chain(task):
    with threadpool_limits(limits=task.pop("threads")):
        return _simulate_chain(**task)


def _simulate_chain(fit, chain, n_draws, local_reps, seed, sites, cells, X, regulated, thr, group, element_group, events,
                    masks, jitter):
    started = time.time()
    rng = np.random.default_rng(seed)
    selected, draws = load_draws(fit, chain, n_draws, KEYS)
    D_ss, D_gs, D_gg = cdist(sites, sites), cdist(cells, sites), cdist(cells, cells)
    n, m, q = len(sites), len(cells), draws["Sigma"].shape[1]
    rows, used_jitter = [], 0.0
    level_sum, sample_sum = np.zeros((m, len(events))), np.zeros((m, len(events)))

    for s in range(len(selected)):
        beta, loading, Sigma, phi, F = (draws[key][s] for key in ["beta", "loading", "Sigma", "rho", "F"])

        # each factor on the cells, conditional on its values at the sites
        field = np.empty((m, len(phi)))
        for k in range(len(phi)):
            L = cholesky(matern32(D_ss, phi[k]) + 1e-8 * np.eye(n), lower=True, check_finite=False)
            W = solve_triangular(L, matern32(D_gs, phi[k]).T, lower=True, check_finite=False)
            mean = W.T @ solve_triangular(L, F[:, k], lower=True, check_finite=False)
            C, added = jittered_cholesky(matern32(D_gg, phi[k]) - W.T @ W, jitter)
            used_jitter = max(used_jitter, added)
            field[:, k] = mean + C @ rng.standard_normal(m)
        mu = X @ beta + field @ loading

        # local error of a new sample
        nu, scale = draws["nu"][s], np.sqrt(local_scales(draws["group_variance"][s], group, element_group))
        w = rng.gamma(nu / 2, 2 / nu, size=(local_reps, m))
        local = (rng.standard_normal((local_reps, m, q)) @ cholesky(Sigma, lower=True).T) * (scale[None] / np.sqrt(w)[:, :, None])

        hit_level = mu[:, regulated] > thr
        hit_sample = (mu[None] + local)[:, :, regulated] > thr[None]
        level = np.column_stack([hit_level[:, idx].any(axis=1) for idx in events.values()]).astype(float)
        sample = np.column_stack([hit_sample[:, :, idx].any(axis=2).mean(axis=0) for idx in events.values()])
        level_sum += level
        sample_sum += sample

        row = {"chain": chain, "draw": int(selected[s])}
        for e, name in enumerate(events):
            for scope, mask in masks.items():
                row[f"level|{name}|{scope}"] = level[mask, e].mean()
                row[f"sample|{name}|{scope}"] = sample[mask, e].mean()
        rows.append(row)
    return rows, level_sum, sample_sum, used_jitter, time.time() - started


def summarize(draws, events, masks):
    rows = []
    for name in events:
        for scope in masks:
            level, sample = 100 * draws[f"level|{name}|{scope}"], 100 * draws[f"sample|{name}|{scope}"]
            row = {"event": name, "scope": scope}
            for label, values in [("level", level), ("sample", sample), ("difference", sample - level)]:
                row.update({f"{label}_mean": values.mean(), f"{label}_lower": values.quantile(INTERVAL[0]),
                            f"{label}_median": values.median(), f"{label}_upper": values.quantile(INTERVAL[1])})
            row["p_difference_positive"] = (sample - level > 0).mean()
            rows.append(row)
    return pd.DataFrame(rows)


def main(args):
    started = time.time()
    out = DIR_AREAS / args.fit
    out.mkdir(parents=True, exist_ok=True)
    spec, data = load_fit(args.fit)
    elements, regulated = spec["elements"], list(data["regulated_indices"])
    names = [elements[j] for j in regulated]

    grid = read_grid()
    cells = lattice_cells(grid[["x", "y"]].to_numpy(), args.spacing)
    sub = grid.iloc[cells].reset_index(drop=True)

    # events: each regulated element and 'at least one' of the union, focus and minor sets
    events = {name: [i] for i, name in enumerate(names)}
    events["union"] = list(range(len(names)))
    events["focus"] = [names.index(e) for e in FOCUS]
    events["minor"] = [names.index(e) for e in MINOR]
    masks = {"all": np.ones(len(sub), dtype=bool), **{use: sub.ngr_use.eq(use).to_numpy() for use in NGR_USES}}

    shared = dict(fit=args.fit, n_draws=args.draws_per_chain, local_reps=args.local_reps, sites=data["coords"],
                  cells=sub[["x", "y"]].to_numpy() / 1000 - data["origin"] / 1000,
                  X=mean_design(sub, spec.get("mean_design", "ngr")), regulated=regulated,
                  thr=thresholds(sub, elements, regulated, data), group=group_index(sub),
                  element_group=spec.get("element_group"), events=events, masks=masks, jitter=args.jitter,
                  threads=args.threads)
    print(f"{len(sub)} cells ({args.spacing:.0f} m), {4 * args.draws_per_chain} draws, {args.local_reps} local replicates")
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(simulate_chain, [dict(shared, chain=c, seed=args.seed + c) for c in range(1, 5)]))

    draws = pd.DataFrame([row for result in results for row in result[0]])
    S = len(draws)
    level_cell, sample_cell = sum(r[1] for r in results) / S, sum(r[2] for r in results) / S
    draws.to_csv(out / "area_draws.csv", index=False)
    summary = summarize(draws, events, masks)
    summary.to_csv(out / "area_summary.csv", index=False)

    # per-cell means must match the grid maps
    pointwise = pd.read_csv(DIR_MAPS / args.fit / "grid_predictions.csv")
    checks = []
    for e, name in enumerate(events):
        for label, joint in [("level", level_cell[:, e]), ("sample", sample_cell[:, e])]:
            reference = pointwise[f"p_{label}_{name}"].to_numpy()
            checks.append(dict(event=name, quantity=label, joint_area_pct=100 * joint.mean(),
                               pointwise_area_subsample_pct=100 * reference[cells].mean(),
                               pointwise_area_full_grid_pct=100 * reference.mean(),
                               max_abs_cell_difference=np.abs(joint - reference[cells]).max(),
                               mean_abs_cell_difference=np.abs(joint - reference[cells]).mean()))
    checks = pd.DataFrame(checks)
    checks.to_csv(out / "checks.csv", index=False)
    save_json(out / "metadata.json", dict(fit=args.fit, cells=len(sub), grid_cells=len(grid), spacing_m=args.spacing,
                                          posterior_draws=S, local_replicates=args.local_reps,
                                          max_jitter_added=max(r[3] for r in results),
                                          chain_seconds=[r[4] for r in results], elapsed_seconds=time.time() - started))

    shown = summary[summary.event.isin(["union", "focus", "minor"] + FOCUS)]
    print(shown[["event", "scope"] + [f"{q}_{s}" for q in ["level", "sample", "difference"] for s in ["mean", "lower", "upper"]]
                + ["p_difference_positive"]].round(2).to_string(index=False))
    print(checks.round(3).to_string(index=False))
    print(f"Saved {out} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", default=SELECTED_MODEL)
    parser.add_argument("--spacing", type=float, default=240.0)
    parser.add_argument("--draws-per-chain", type=int, default=500)
    parser.add_argument("--local-reps", type=int, default=32)
    parser.add_argument("--jitter", type=float, default=1e-8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261025)
    main(parser.parse_args())
