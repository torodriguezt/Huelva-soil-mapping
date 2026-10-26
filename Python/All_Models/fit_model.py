# Full-data fit of one model with four chains.
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import t as student_t

from blmc.config import DIR_FITS, DIR_MODEL_DATA, FULL_ITERATIONS, GROUPS, NGR_USES
from blmc.covariance import to_correlation
from blmc.io import load_folder, save_json
from blmc.diagnostics import diagnose
from blmc.model import censoring, parse_model
from blmc.sampler import run_chain

PAIRS = [("As", "Tl"), ("As", "Pb"), ("As", "Cu"), ("As", "V"), ("V", "Sc"), ("V", "Al"), ("V", "Ga"), ("Co", "As"), ("Co", "V")]
TRACKED_SITES = [37, 59, 61, 69, 125, 35, 5, 97, 52, 83]


def chain_task(task):
    draws, summary = run_chain(task["Y"], task["coords"], task["X"], task["groups"], task["K"], task["warmup"],
                               task["samples"], task["thin"], task["seed"], config=task["config"],
                               censored=task["censored"], censor_lower=task["lower"],
                               report_every=10000, label=f"chain {task['chain']}")
    np.savez_compressed(task["path"], **draws)
    save_json(task["path"].with_suffix(".json"), summary)
    return task["chain"]


def monitors(chain, draws, data, elements, robust):
    beta, loading, Sigma, rho, F, a, w, nu = (draws[key] for key in
                                              ["beta", "loading", "Sigma", "rho", "F", "group_variance", "weights", "nu"])
    S = len(beta)
    C0 = np.einsum("skj,skl->sjl", loading, loading)
    cor0, corE = to_correlation(C0), to_correlation(Sigma)
    var0, varE = np.diagonal(C0, axis1=1, axis2=2), np.diagonal(Sigma, axis1=1, axis2=2)
    W = np.einsum("snk,skj->snj", F, loading)
    centered = W - W.mean(axis=1, keepdims=True)
    mu = np.einsum("np,spj->snj", data["X"], beta) + W
    groups = data["variance_group_index"]
    scale_row = a[:, groups]

    cols = {"chain": np.full(S, chain), "iteration": draws["iteration"]}
    for p, label in enumerate(data["X_labels"]):
        for j, e in enumerate(elements):
            cols[f"beta_{label}_{e}"] = beta[:, p, j]
    for j, e in enumerate(elements):
        cols[f"variance_spatial_{e}"] = var0[:, j]
        cols[f"variance_local_Agric_{e}"] = varE[:, j]
    for g, label in enumerate(GROUPS[1:], start=1):
        cols[f"local_variance_ratio_{label}_vs_Agric"] = a[:, g]
    for left, right in PAIRS:
        j, k = elements.index(left), elements.index(right)
        cols[f"cor_spatial_{left}_{right}"] = cor0[:, j, k]
        cols[f"cor_local_{left}_{right}"] = corE[:, j, k]
    for h in [1.0, 3.0]:
        z = np.sqrt(3) * h / rho
        Ch = np.einsum("sk,skj,skl->sjl", (1 + z) * np.exp(-z), loading, loading)
        for left, right in PAIRS:
            cols[f"cov_h{int(h)}km_{left}_{right}"] = Ch[:, elements.index(left), elements.index(right)]
    for site in [1, 21, 41, 61, 81, 101, 121, 138]:
        found = np.flatnonzero(data["ids"] == site)
        if not len(found):
            continue
        for e in ["As", "Tl", "Cu", "Al", "V", "Co"]:
            cols[f"W_centered_site{site}_{e}"] = centered[:, int(found[0]), elements.index(e)]
    for k, values in enumerate(np.sort(rho, axis=1).T, start=1):
        cols[f"lengthscale_{k}_km"] = values
    for e in ["As", "Tl", "V", "Co"]:
        j = elements.index(e)
        z = (data["threshold_z"][None, :, j] - mu[:, :, j]) / np.sqrt(scale_row * varE[:, j, None])
        cols[f"mean_p_new_{e}"] = (student_t.sf(z, nu[:, None]) if robust else 1 - ndtr(z)).mean(axis=1)
    cols["log_likelihood"] = draws["log_likelihood"]
    if robust:
        cols["nu"], cols["log_nu"] = nu, np.log(nu)
        for site in TRACKED_SITES:
            found = np.flatnonzero(data["ids"] == site)
            if len(found):
                cols[f"weight_site{site}"] = w[:, int(found[0])]

    # replicate: a new local sample at each site
    masks = {"all": np.ones(len(groups), dtype=bool), **{g: groups == k for k, g in enumerate(GROUPS)},
             **{f"reference_{u}": data["use"] == u for u in NGR_USES}}
    regulated = data["regulated_indices"]
    rng = np.random.default_rng(20261101 + chain)
    counts = []
    for s in range(S):
        w_new = rng.gamma(nu[s] / 2, 2 / nu[s], size=len(groups)) if robust else np.ones(len(groups))
        replicate = mu[s] + np.sqrt(scale_row[s] / w_new)[:, None] * (rng.normal(size=mu[s].shape) @ np.linalg.cholesky(Sigma[s]).T)
        hit = replicate > data["threshold_z"]
        row = {"chain": chain, "iteration": int(draws["iteration"][s])}
        for scope, mask in masks.items():
            row[f"union_{scope}"] = np.any(hit[mask][:, regulated], axis=1).sum()
            row[f"multiple_{scope}"] = (hit[mask][:, regulated].sum(axis=1) >= 2).sum()
            for j in regulated:
                row[f"{elements[j]}_{scope}"] = hit[mask, j].sum()
        counts.append(row)
    return pd.DataFrame(cols), pd.DataFrame(counts), masks


def observed_counts(data, elements, masks):
    observed = data["Y"] > data["threshold_z"]
    regulated = data["regulated_indices"]
    rows = []
    for scope, mask in masks.items():
        hits = observed[mask][:, regulated]
        rows += [dict(event="union", scope=scope, n=int(mask.sum()), observed_count=int(hits.any(axis=1).sum())),
                 dict(event="multiple", scope=scope, n=int(mask.sum()), observed_count=int((hits.sum(axis=1) >= 2).sum()))]
        rows += [dict(event=elements[j], scope=scope, n=int(mask.sum()), observed_count=int(observed[mask, j].sum()))
                 for j in regulated]
    return rows


def site_weights(weights, data):
    inflation = 1 / weights
    q = np.quantile(inflation, [0.05, 0.5, 0.95], axis=0)
    xy = data["coords"] * 1000 + data["origin"]
    table = pd.DataFrame({"id": data["ids"], "land_use": np.array(GROUPS)[data["variance_group_index"]],
                          "x": xy[:, 0], "y": xy[:, 1], "weight_median": np.median(weights, axis=0),
                          "inflation_median": q[1], "inflation_q05": q[0], "inflation_q95": q[2]})
    return table.sort_values("inflation_median", ascending=False)


def main(args):
    started = time.time()
    model = parse_model(args.model)
    out = DIR_FITS / (args.name or args.model)
    if (out / "run_status.json").exists():
        print(f"{out.name}: already fitted")
        return
    spec, data = load_folder(DIR_MODEL_DATA / args.data)
    elements = spec["elements"]
    if model["design"] == "use6":
        data["X"], data["X_labels"] = data["X_use6"], data["X_use6_labels"]
    data["censored"], data["censor_lower"] = censoring(data["Y"], data["center"], data["scale"], elements)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "data.npz", **data)

    config = dict(robust=model["robust"], nu_prior_shape=args.nu_prior_shape, nu_prior_rate=args.nu_prior_rate,
                  group_log_variance_prior_sd=args.group_prior_sd, log_rho_mean=float(np.log(args.rho_prior_median_km)),
                  log_rho_sd=args.rho_prior_sd)
    censored = [dict(id=int(data["ids"][i]), element=elements[j]) for i, j in zip(*np.nonzero(data["censored"]))]
    save_json(out / "specification.json", dict(
        **spec, model=args.model, K=model["K"], robust=model["robust"], mean_design=model["design"], data=args.data,
        warmup=args.warmup, samples=args.samples, thin=args.thin, chains=args.chains, priors=config,
        censored_entries=censored))

    tasks = [dict(Y=data["Y"], coords=data["coords"], X=data["X"], groups=data["variance_group_index"],
                  K=model["K"], warmup=args.warmup, samples=args.samples, thin=args.thin, config=config,
                  censored=data["censored"], lower=data["censor_lower"], chain=c, path=out / f"chain_{c}.npz",
                  seed=20261022 + 1000 * c + 10 * model["K"] + (0 if model["robust"] else 5))
             for c in range(1, args.chains + 1)]
    print(f"{out.name}: {args.model}, {len(data['Y'])} sites, {len(elements)} elements, {len(censored)} censored entries")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(chain_task, task) for task in tasks]):
            print(f"chain {future.result()} saved after {time.time() - started:.0f}s", flush=True)

    tables, counts, weights = [], [], []
    for c in range(1, args.chains + 1):
        with np.load(out / f"chain_{c}.npz") as archive:
            selected = np.arange(0, len(archive["beta"]), args.summary_stride)
            draws = {key: archive[key][selected] for key in archive.files}
        draws["iteration"] = selected + 1
        table, count, masks = monitors(c, draws, data, elements, model["robust"])
        tables.append(table)
        counts.append(count)
        weights.append(draws["weights"])
    pd.concat(tables).to_csv(out / "monitors.csv", index=False)
    counts = pd.concat(counts, ignore_index=True)
    counts.to_csv(out / "posterior_replication_counts.csv", index=False)
    summary = [dict(row, replicated_mean=counts[f"{row['event']}_{row['scope']}"].mean(),
                    replicated_q05=counts[f"{row['event']}_{row['scope']}"].quantile(0.05),
                    replicated_q95=counts[f"{row['event']}_{row['scope']}"].quantile(0.95))
               for row in observed_counts(data, elements, masks)]
    pd.DataFrame(summary).to_csv(out / "posterior_replication_summary.csv", index=False)
    if model["robust"]:
        site_weights(np.concatenate(weights), data).to_csv(out / "site_weights.csv", index=False)

    diagnostics = diagnose(out / "monitors.csv", out / "diagnostics.csv")
    save_json(out / "run_status.json", dict(completed=True, elapsed_seconds=time.time() - started,
                                            max_rhat=diagnostics.rhat.max(), min_ess_bulk=diagnostics.ess_bulk.min(),
                                            min_ess_tail=diagnostics.ess_tail.min()))
    print(f"{out.name}: max R-hat {diagnostics.rhat.max():.3f}, min ESS bulk {diagnostics.ess_bulk.min():.0f}, "
          f"tail {diagnostics.ess_tail.min():.0f}; {time.time() - started:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="t_K3_ngr", help="t_K3_ngr, gauss_K3_ngr, t_K4_use6, ...")
    parser.add_argument("--name", default=None, help="output folder under results/fits; default the model name")
    parser.add_argument("--data", default="main", choices=["main", "keep137"])
    parser.add_argument("--warmup", type=int, default=FULL_ITERATIONS["warmup"])
    parser.add_argument("--samples", type=int, default=FULL_ITERATIONS["samples"])
    parser.add_argument("--thin", type=int, default=FULL_ITERATIONS["thin"])
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--summary-stride", type=int, default=2)
    parser.add_argument("--nu-prior-shape", type=float, default=2.0)
    parser.add_argument("--nu-prior-rate", type=float, default=0.1)
    parser.add_argument("--group-prior-sd", type=float, default=1.0, help="SD of log a_g")
    parser.add_argument("--rho-prior-median-km", type=float, default=2.0, help="median of the log-normal length-scale prior")
    parser.add_argument("--rho-prior-sd", type=float, default=0.8)
    main(parser.parse_args())
