# Posterior predictive checks of the local residuals.
import argparse
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis, skew
from threadpoolctl import threadpool_limits

from blmc.config import DIR_CHECKS, DIR_FITS, GROUPS, INTERVAL, SELECTED_MODEL
from blmc.io import load_fit, save_json
from blmc.model import local_scales

STATISTICS = ["mean_log", "centered_variance_log", "skewness", "excess_kurtosis", "count_abs_z_gt_2", "moran_I"]
MINIMUM_N = {"mean_log": 2, "centered_variance_log": 2, "skewness": 8, "excess_kurtosis": 8, "count_abs_z_gt_2": 8,
             "moran_I": 12}


def neighbour_graph(coordinates, k=8):
    n = len(coordinates)
    distance = cdist(coordinates, coordinates)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argsort(distance, axis=1)[:, :k]
    directed = np.zeros((n, n), dtype=bool)
    directed[np.arange(n)[:, None], nearest] = True
    return csr_matrix((directed | directed.T).astype(float))


def moran(values, weights):
    centered = values - values.mean(axis=1, keepdims=True)
    draws, n, q = centered.shape
    lagged = (weights @ centered.transpose(1, 0, 2).reshape(n, -1)).reshape(n, draws, q).transpose(1, 0, 2)
    return n / weights.sum() * np.sum(centered * lagged, axis=1) / np.sum(centered * centered, axis=1)


def statistics(residual_log, standardized, masks, graphs):
    out = np.full((len(residual_log), len(masks), len(STATISTICS), residual_log.shape[2]), np.nan)
    for g, mask in enumerate(masks):
        n = int(mask.sum())
        raw, z = residual_log[:, mask], standardized[:, mask]
        if n >= 2:
            out[:, g, 0] = raw.mean(axis=1)
            out[:, g, 1] = raw.var(axis=1, ddof=1)
        if n >= 8:
            out[:, g, 2] = skew(raw, axis=1, bias=False)
            out[:, g, 3] = kurtosis(raw, axis=1, fisher=True, bias=False)
            out[:, g, 4] = (np.abs(z) > 2).sum(axis=1)
        if g in graphs:
            out[:, g, 5] = moran(z, graphs[g])
    return out


def compare(observed, replicated):
    if not np.isfinite(observed).all() or not np.isfinite(replicated).all():
        return {}
    levels = [INTERVAL[0], 0.5, INTERVAL[1]]
    obs, rep = np.quantile(observed, levels), np.quantile(replicated, levels)
    greater, less = np.mean(replicated >= observed), np.mean(replicated <= observed)
    return {"observed_lower": obs[0], "observed_median": obs[1], "observed_upper": obs[2],
            "replicated_lower": rep[0], "replicated_median": rep[1], "replicated_upper": rep[2],
            "p_rep_ge_observed": greater, "p_rep_le_observed": less, "two_sided_tail_area": min(1, 2 * min(greater, less)),
            "median_observed_over_replicated": obs[1] / rep[1] if rep[1] > 0 else np.nan}


def main(args):
    started = time.time()
    out = DIR_CHECKS / args.fit
    out.mkdir(parents=True, exist_ok=True)
    spec, data = load_fit(args.fit)
    elements, robust = spec["elements"], spec["robust"]
    uses = np.array(GROUPS)[data["variance_group_index"]]
    groups = ["all"] + [g for g in GROUPS if g in set(uses)]
    masks = [np.ones(len(uses), dtype=bool)] + [uses == g for g in groups[1:]]
    graphs = {g: neighbour_graph(data["coords"][mask]) for g, mask in enumerate(masks) if mask.sum() >= MINIMUM_N["moran_I"]}
    rows, cols = np.nonzero(data["censored"])
    index = data["variance_group_index"]

    observed_parts, replicated_parts = [], []
    for chain in range(1, spec["chains"] + 1):
        with np.load(DIR_FITS / args.fit / f"chain_{chain}.npz") as archive:
            selected = np.linspace(0, len(archive["beta"]) - 1, args.draws_per_chain, dtype=int)
            beta, loading, F, Sigma, a, nu, imputed = (archive[key][selected] for key in
                                                       ["beta", "loading", "F", "Sigma", "group_variance", "nu", "censored_values"])
        Y = np.repeat(data["Y"][None], len(selected), axis=0)
        Y[:, rows, cols] = imputed
        observed = Y - np.einsum("np,dpj->dnj", data["X"], beta) - np.einsum("dnk,dkj->dnj", F, loading)
        rng = np.random.default_rng(args.seed + chain)
        scale = local_scales(a, index, spec.get("element_group"))
        shape = scale.shape[:2]
        fresh = rng.gamma(nu[:, None] / 2, 2 / nu[:, None], size=shape) if robust else np.ones(shape)
        replicated = np.einsum("dnj,dkj->dnk", rng.standard_normal(observed.shape), np.linalg.cholesky(Sigma)) \
            * np.sqrt(scale / fresh[:, :, None])
        sd = np.sqrt(np.diagonal(Sigma, axis1=1, axis2=2)[:, None, :] * scale)
        observed_parts.append(statistics(observed * data["scale"], observed / sd, masks, graphs))
        replicated_parts.append(statistics(replicated * data["scale"], replicated / sd, masks, graphs))
        print(f"chain {chain}: {len(selected)} draws", flush=True)
    observed, replicated = np.concatenate(observed_parts), np.concatenate(replicated_parts)

    summary = []
    for g, group in enumerate(groups):
        n = int(masks[g].sum())
        for s, statistic in enumerate(STATISTICS):
            for j, element in enumerate(elements):
                summary.append(dict(group=group, n=n, statistic=statistic, element=element,
                                    supported=n >= MINIMUM_N[statistic],
                                    **compare(observed[:, g, s, j], replicated[:, g, s, j])))
    pd.DataFrame(summary).to_csv(out / "residual_checks_summary.csv", index=False)

    # variance ratio of each land use to agricultural soils
    ratios, agric = [], groups.index("Agric")
    for use in ["Marsh", "Urban", "Industria", "Phospho"]:
        g = groups.index(use)
        for j, element in enumerate(elements):
            ratios.append(dict(contrast=f"{use}/Agric", element=element,
                               **compare(observed[:, g, 1, j] / observed[:, agric, 1, j],
                                         replicated[:, g, 1, j] / replicated[:, agric, 1, j])))
    pd.DataFrame(ratios).to_csv(out / "residual_variance_ratio_summary.csv", index=False)
    save_json(out / "metadata.json", dict(fit=args.fit, groups=groups, draws_per_chain=args.draws_per_chain, seed=args.seed,
                                          elapsed_seconds=time.time() - started))
    print(f"saved {out} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", default=SELECTED_MODEL)
    parser.add_argument("--draws-per-chain", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261024)
    with threadpool_limits(limits=1):
        main(parser.parse_args())
