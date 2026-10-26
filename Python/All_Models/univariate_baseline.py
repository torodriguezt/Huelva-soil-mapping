# Univariate baseline of the cross-validation: one Student-t spatial model (K = 1) per element, same folds and scores.
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, logsumexp
from scipy.stats import norm
from threadpoolctl import threadpool_limits

from blmc.config import CANDIDATES, CV_ITERATIONS, DIR_CV, DIR_MODEL_DATA, GROUPS, N_BLOCKS
from blmc.diagnostics import diagnose
from blmc.io import load_folder, read_json, save_json
from blmc.predict import conditional_predict
from blmc.sampler import run_chain
from cross_validation import fit_folder, fold_data, make_folds, site_scores, summarize

MODEL = "univariate"
ARRAYS = ["beta", "loading", "Sigma", "rho", "F", "group_variance", "nu", "log_likelihood"]


def chain_task(task):
    draws, summary = run_chain(task["Y"], task["coords"], task["X"], task["groups"], 1, task["warmup"], task["samples"],
                               task["thin"], task["seed"], censored=task["censored"], censor_lower=task["lower"])
    return task["fold"], task["element"], task["chain"], {key: draws[key] for key in ARRAYS}, summary["elapsed_seconds"]


# mean, spatial variance and local variance of a new sample of one element at the held-out sites
def predictive(chains, data, fd, args, rng):
    train, test = fd["train"], fd["test"]
    groups_test = fd["groups"][test]
    mu, spatial, local, chain_of = [], [], [], []
    for chain, draws in chains:
        for s in range(0, len(draws["beta"]), args.score_stride):
            draw = {key: draws[key][s] for key in ["beta", "loading", "rho", "F"]}
            prediction = conditional_predict(draw, data["coords"][train], data["coords"][test], data["X"][test])
            a = np.where(groups_test >= 0, draws["group_variance"][s][np.maximum(groups_test, 0)],
                         np.exp(rng.normal(size=len(groups_test))))
            nu = draws["nu"][s]
            w = rng.gamma(nu / 2, 2 / nu, size=(args.weight_draws, len(groups_test)))
            mu.append(prediction["mean_y"][:, 0])
            spatial.append(prediction["var_f"][:, 0] * draw["loading"][0, 0] ** 2)
            local.append(a / w * draws["Sigma"][s][0, 0])
            chain_of.append(chain)
    return np.array(mu), np.array(spatial), np.array(local), np.array(chain_of)


def score_fold(fold, chains, data, fd, elements, args):
    rng = np.random.default_rng(20261700 + 100 * fold)
    test = fd["test"]
    y, cens, lower, thr = fd["Y"][test], fd["censored"][test], fd["lower"][test], fd["threshold_z"][test]
    n_test, q = y.shape
    M = args.weight_draws
    parts = [predictive(chains[j], data, fd, args, rng) for j in range(q)]
    S = len(parts[0][0])
    per_draw = np.zeros((S, q))
    site_rows, element_rows = [], []
    for i in range(n_test):
        lpd, samples, level = 0.0, np.empty((S * M, q)), np.empty((S, q))
        for j, (mu, spatial, local, _) in enumerate(parts):
            mean = np.repeat(mu[:, i], M)
            sd = np.sqrt(np.repeat(spatial[:, i], M) + local[:, :, i].ravel())
            if cens[i, j]:
                logdens, jacobian = log_ndtr((mean - lower[i, j]) / sd), 0.0
            else:
                logdens, jacobian = norm.logpdf(y[i, j], mean, sd), -np.log(fd["scale"][j])
            per_draw[:, j] += logsumexp(logdens.reshape(S, M), axis=1) - np.log(M) + jacobian
            lpd += logsumexp(logdens) - np.log(S * M) + jacobian
            samples[:, j] = mean + sd * rng.standard_normal(S * M)
            level[:, j] = mu[:, i] + np.sqrt(spatial[:, i]) * rng.standard_normal(S)
        site = dict(fold=fold, model=MODEL, id=int(data["ids"][test][i]), land_use=GROUPS[data["variance_group_index"][test][i]],
                    group_in_training=bool(fd["groups"][test][i] >= 0))
        row, rows = site_scores(site, lpd, samples, level, y[i], cens[i], lower[i], thr[i], fd, data, elements, rng)
        site_rows.append(row)
        element_rows += rows
    return pd.DataFrame(site_rows), pd.DataFrame(element_rows), per_draw, parts[0][3]


def monitors(chains, per_draw, chain_of, fd, elements, args):
    rows = []
    for chain in sorted(dict(chains[0])):
        selected = np.arange(0, len(dict(chains[0])[chain]["beta"]), args.score_stride)
        cols = {"chain": np.full(len(selected), chain), "iteration": np.arange(1, len(selected) + 1)}
        for j, e in enumerate(elements):
            draws = dict(chains[j])[chain]
            for g, name in enumerate(fd["group_names"][1:], start=1):
                cols[f"{e}_local_variance_ratio_{name}_vs_Agric"] = draws["group_variance"][selected, g]
            cols[f"{e}_lengthscale_km"] = draws["rho"][selected, 0]
            cols[f"{e}_variance_spatial"] = draws["loading"][selected, 0, 0] ** 2
            cols[f"{e}_variance_local_Agric"] = draws["Sigma"][selected, 0, 0]
            cols[f"{e}_log_likelihood"] = draws["log_likelihood"][selected]
            cols[f"{e}_log_nu"] = np.log(draws["nu"][selected])
            cols[f"{e}_heldout_block_lppd"] = per_draw[chain_of == chain, j]
        rows.append(pd.DataFrame(cols))
    return pd.concat(rows, ignore_index=True)


def finish_fold(fold, chains, data, fd, elements, args, elapsed, settings):
    folder = fit_folder(fold, MODEL)
    sites, element_scores, per_draw, chain_of = score_fold(fold, chains, data, fd, elements, args)
    sites.to_csv(folder / "site_scores.csv", index=False)
    element_scores.to_csv(folder / "element_scores.csv", index=False)
    monitors(chains, per_draw, chain_of, fd, elements, args).to_csv(folder / "monitors.csv", index=False)
    d = diagnose(folder / "monitors.csv", folder / "diagnostics.csv")
    heldout = d[d.index.str.endswith("_heldout_block_lppd")]
    status = dict(completed=True, fold=fold, model=MODEL, n_train=int(fd["train"].sum()), n_test=int(fd["test"].sum()),
                  chain_seconds=elapsed, max_rhat=float(d.rhat.max()), min_ess_bulk=float(d.ess_bulk.min()),
                  min_ess_tail=float(d.ess_tail.min()), worst_rhat_variable=str(d.rhat.idxmax()),
                  heldout_lppd_rhat=float(heldout.rhat.max()), heldout_lppd_ess_bulk=float(heldout.ess_bulk.min()),
                  lpd_sum=float(sites.lpd.sum()), **settings)
    save_json(folder / "fit_status.json", status)
    print(f"fold {fold:02d} {MODEL}: lpd {status['lpd_sum']:.2f}, max R-hat {status['max_rhat']:.3f} "
          f"({status['worst_rhat_variable']})", flush=True)


def main(args):
    started = time.time()
    spec, data = load_folder(DIR_MODEL_DATA / "main")
    elements = spec["elements"]
    blocks, _ = make_folds(data)
    settings = dict(warmup=args.warmup, samples=args.samples, thin=args.thin)
    folds = [f for f in (args.folds or range(N_BLOCKS)) if not (DIR_CV / f"fold{f:02d}_{MODEL}" / "fit_status.json").exists()]
    fold_inputs = {fold: fold_data(data, blocks, fold, elements) for fold in folds}
    tasks = []
    for fold in folds:
        fd = fold_inputs[fold]
        train = fd["train"]
        for j in range(len(elements)):
            for c in range(1, args.chains + 1):
                tasks.append(dict(Y=fd["Y"][train][:, [j]], coords=data["coords"][train], X=data["X"][train],
                                  groups=fd["groups"][train], censored=fd["censored"][train][:, [j]],
                                  lower=fd["lower"][train][:, [j]], fold=fold, element=j, chain=c,
                                  seed=20261600 + 10000 * c + 100 * fold + j, **settings))
    print(f"{len(tasks)} chains for {len(folds)} folds", flush=True)

    pending, seconds = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(chain_task, t) for t in tasks]):
            fold, j, chain, arrays, elapsed = future.result()
            pending.setdefault(fold, {}).setdefault(j, []).append((chain, arrays))
            seconds.setdefault(fold, []).append(elapsed)
            if len(seconds[fold]) == len(elements) * args.chains:
                chains = {j: sorted(pending[fold][j], key=lambda item: item[0]) for j in range(len(elements))}
                finish_fold(fold, chains, data, fold_inputs[fold], elements, args, seconds.pop(fold), settings)
                del pending[fold]

    candidates = read_json(DIR_CV / "specification.json")["models"] if (DIR_CV / "specification.json").exists() else CANDIDATES
    summarize(candidates, pd.read_csv(DIR_CV / "folds.csv"))
    print(f"univariate baseline finished in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="subset of folds, for quick tests")
    parser.add_argument("--warmup", type=int, default=CV_ITERATIONS["warmup"])
    parser.add_argument("--samples", type=int, default=CV_ITERATIONS["samples"])
    parser.add_argument("--thin", type=int, default=CV_ITERATIONS["thin"])
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--score-stride", type=int, default=2)
    parser.add_argument("--weight-draws", type=int, default=4)
    with threadpool_limits(limits=1):
        main(parser.parse_args())
