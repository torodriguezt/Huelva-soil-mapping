# Spatial block cross-validation of the candidate models.
import argparse
import json
import shutil
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.linalg import cho_solve
from scipy.spatial import cKDTree
from scipy.special import log_ndtr, logsumexp
from threadpoolctl import threadpool_limits

from blmc.config import (BLOCK_SEED, CANDIDATES, CV_ITERATIONS, DIR_CV, DIR_MODEL_DATA, GROUPS, N_BLOCKS,
                         UPPER_LIMITS)
from blmc.io import load_folder, read_json, save_json
from blmc.diagnostics import diagnose
from blmc.model import design_matrix, parse_model
from blmc.predict import conditional_predict
from blmc.sampler import run_chain

ARRAYS = ["beta", "loading", "Sigma", "rho", "F", "group_variance", "weights", "nu", "log_likelihood"]


def make_folds(data):
    xy = data["coords"] * 1000 + data["origin"]
    _, blocks = kmeans2(xy / 1000, N_BLOCKS, seed=BLOCK_SEED, minit="++")
    distance = np.empty(len(xy))
    for b in range(N_BLOCKS):
        test = blocks == b
        distance[test] = cKDTree(xy[~test]).query(xy[test])[0]
    return blocks, distance


def fold_data(data, blocks, fold, elements):
    train, test = blocks != fold, blocks == fold
    logy = data["Y"] * data["scale"] + data["center"]
    log_threshold = data["threshold_z"] * data["scale"] + data["center"]
    center, scale = logy[train].mean(axis=0), logy[train].std(axis=0, ddof=1)
    censored, lower = np.zeros(logy.shape, dtype=bool), np.full(logy.shape, np.nan)
    for element, limit in UPPER_LIMITS.items():
        j = elements.index(element)
        hit = logy[:, j] >= np.log(limit) - 1e-9
        censored[hit, j], lower[hit, j] = True, (np.log(limit) - center[j]) / scale[j]
    present = [g for g in range(len(GROUPS)) if np.any(data["variance_group_index"][train] == g)]
    remap = np.full(len(GROUPS), -1)
    remap[present] = np.arange(len(present))
    return dict(train=train, test=test, Y=(logy - center) / scale, center=center, scale=scale, censored=censored,
                lower=lower, threshold_z=(log_threshold - center) / scale, groups=remap[data["variance_group_index"]],
                group_names=[GROUPS[g] for g in present])


def chain_task(task):
    draws, summary = run_chain(task["Y"], task["coords"], task["X"], task["groups"], task["K"], task["warmup"],
                               task["samples"], task["thin"], task["seed"], config={"robust": task["robust"]},
                               censored=task["censored"], censor_lower=task["lower"])
    return task["fold"], task["model"], task["chain"], {key: draws[key] for key in ARRAYS}, summary["elapsed_seconds"]


def crps_samples(samples, observation):
    x = np.sort(samples)
    m = len(x)
    return np.mean(np.abs(x - observation)) - np.sum((2 * np.arange(1, m + 1) - m - 1) * x) / m**2


def component_log_density(y, mean, cov, censored, lower):
    if not censored.any():
        chol = np.linalg.cholesky(cov)
        z = np.linalg.solve(chol, (y - mean)[..., None])[..., 0]
        return -0.5 * (np.sum(z**2, axis=-1) + 2 * np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=-1)
                       + len(y) * np.log(2 * np.pi))
    if censored.sum() > 1:
        raise NotImplementedError("only one censored entry per held-out site")
    c, o = int(np.flatnonzero(censored)[0]), ~censored
    observed = component_log_density(y[o], mean[:, o], cov[:, o][:, :, o], np.zeros(o.sum(), dtype=bool), None)
    chol = np.linalg.cholesky(cov[:, o][:, :, o])
    coefficient = np.stack([cho_solve((L, True), C) for L, C in zip(chol, cov[:, o, c])])
    m = mean[:, c] + np.einsum("sj,sj->s", coefficient, y[o] - mean[:, o])
    s = np.sqrt(cov[:, c, c] - np.einsum("sj,sj->s", coefficient, cov[:, o, c]))
    return observed + log_ndtr((m - lower[c]) / s)


def score_fit(fold, model, chains, data, fd, elements, args):
    spec = parse_model(model)
    train, test = fd["train"], fd["test"]
    rng = np.random.default_rng(20261300 + 100 * fold + zlib.crc32(model.encode()) % 97)
    level_rng = np.random.default_rng(20261400 + 100 * fold + zlib.crc32(model.encode()) % 97)
    X_test = design_matrix(data, spec["design"])[test]
    groups_test = fd["groups"][test]
    y, cens, lower, thr = fd["Y"][test], fd["censored"][test], fd["lower"][test], fd["threshold_z"][test]
    n_test, q = y.shape
    M = args.weight_draws if spec["robust"] else 1

    # a land use absent from training draws its scale from the prior
    parts = {key: [] for key in ["mu", "var_f", "loading", "Sigma", "a", "w", "chain"]}
    for chain, draws in chains:
        for s in range(0, len(draws["beta"]), args.score_stride):
            draw = {key: draws[key][s] for key in ["beta", "loading", "Sigma", "rho", "F"]}
            prediction = conditional_predict(draw, data["coords"][train], data["coords"][test], X_test)
            a = np.where(groups_test >= 0, draws["group_variance"][s][np.maximum(groups_test, 0)],
                         np.exp(rng.normal(size=n_test)))
            nu = draws["nu"][s]
            parts["mu"].append(prediction["mean_y"])
            parts["var_f"].append(prediction["var_f"])
            parts["loading"].append(draw["loading"])
            parts["Sigma"].append(draw["Sigma"])
            parts["a"].append(a)
            parts["w"].append(rng.gamma(nu / 2, 2 / nu, size=(M, n_test)) if spec["robust"] else np.ones((M, n_test)))
            parts["chain"].append(chain)
    mu, var_f, loading, Sigma, a, w, chain_of = (np.array(parts[key]) for key in parts)
    S = len(mu)
    regulated = data["regulated_indices"]
    ids = data["ids"][test]
    per_draw = np.zeros(S)
    site_rows, element_rows = [], []
    for i in range(n_test):
        V = np.einsum("sk,skq,skr->sqr", var_f[:, i], loading, loading)
        cov = (V[:, None] + (a[:, i, None] / w[:, :, i])[:, :, None, None] * Sigma[:, None]).reshape(S * M, q, q)
        mean = np.repeat(mu[:, i], M, axis=0)
        logdens = component_log_density(y[i], mean, cov, cens[i], lower[i])
        jacobian = -np.log(fd["scale"][~cens[i]]).sum()
        per_draw += logsumexp(logdens.reshape(S, M), axis=1) - np.log(M) + jacobian
        lpd = logsumexp(logdens) - np.log(S * M) + jacobian

        samples = mean + np.einsum("sqr,sr->sq", np.linalg.cholesky(cov), rng.standard_normal((S * M, q)))
        log_samples = samples * fd["scale"] + fd["center"]
        log_observed = y[i] * fd["scale"] + fd["center"]
        exceed = samples > thr[i]
        known = np.where(cens[i], np.where(lower[i] > thr[i], 1.0, np.nan), (y[i] > thr[i]).astype(float))
        if np.nansum(known[regulated]) > 0:
            union_observed = 1.0
        elif np.isnan(known[regulated]).any():
            union_observed = np.nan
        else:
            union_observed = 0.0
        union_p = np.any(exceed[:, regulated], axis=1).mean()
        level = mu[:, i] + np.einsum("sk,skq->sq", np.sqrt(var_f[:, i]) * level_rng.standard_normal(var_f[:, i].shape), loading)
        level_exceed = level > thr[i]
        energy = np.nan
        if not cens[i].any():
            z_samples = (log_samples - data["center"]) / data["scale"]
            z_observed = (log_observed - data["center"]) / data["scale"]
            energy = np.mean(np.linalg.norm(z_samples - z_observed, axis=1)) - 0.5 * np.mean(
                np.linalg.norm(z_samples - z_samples[rng.permutation(len(z_samples))], axis=1))
        site_rows.append(dict(fold=fold, model=model, id=int(ids[i]), land_use=GROUPS[data["variance_group_index"][test][i]],
                              group_in_training=bool(groups_test[i] >= 0), censored_entries=int(cens[i].sum()), lpd=lpd,
                              energy_score=energy, union_p=union_p, union_p_level=np.any(level_exceed[:, regulated], axis=1).mean(),
                              union_observed=union_observed,
                              union_brier=(union_p - union_observed) ** 2 if np.isfinite(union_observed) else np.nan))
        for j, e in enumerate(elements):
            if cens[i, j]:
                continue
            lo50, hi50, lo90, hi90 = np.quantile(log_samples[:, j], [0.25, 0.75, 0.05, 0.95])
            p = exceed[:, j].mean() if j in regulated else np.nan
            element_rows.append(dict(
                fold=fold, model=model, id=int(ids[i]), element=e, crps=crps_samples(log_samples[:, j], log_observed[j]),
                abs_error_median=abs(np.median(log_samples[:, j]) - log_observed[j]),
                squared_error_mean=(log_samples[:, j].mean() - log_observed[j]) ** 2,
                cover50=float(lo50 <= log_observed[j] <= hi50), cover90=float(lo90 <= log_observed[j] <= hi90),
                exceed_p=p, exceed_p_level=level_exceed[:, j].mean() if j in regulated else np.nan,
                exceed_observed=known[j] if j in regulated else np.nan,
                brier=(p - known[j]) ** 2 if j in regulated and np.isfinite(known[j]) else np.nan))
    iteration = np.concatenate([np.arange(1, (chain_of == c).sum() + 1) for c in np.unique(chain_of)])
    return pd.DataFrame(site_rows), pd.DataFrame(element_rows), dict(chain=chain_of, iteration=iteration,
                                                                     heldout_block_lppd=per_draw)


def fit_monitors(chains, fd, model, elements, args, block_monitor):
    rows = []
    for chain, draws in chains:
        selected = np.arange(0, len(draws["beta"]), args.score_stride)
        C0 = np.einsum("skj,skl->sjl", draws["loading"][selected], draws["loading"][selected])
        cols = {"chain": np.full(len(selected), chain), "iteration": np.arange(1, len(selected) + 1)}
        for g, name in enumerate(fd["group_names"][1:], start=1):
            cols[f"local_variance_ratio_{name}_vs_Agric"] = draws["group_variance"][selected, g]
        for k, values in enumerate(np.sort(draws["rho"][selected], axis=1).T, start=1):
            cols[f"lengthscale_{k}_km"] = values
        for j, e in enumerate(elements):
            cols[f"variance_spatial_{e}"] = C0[:, j, j]
            cols[f"variance_local_Agric_{e}"] = draws["Sigma"][selected, j, j]
        cols["log_likelihood"] = draws["log_likelihood"][selected]
        if parse_model(model)["robust"]:
            cols["log_nu"] = np.log(draws["nu"][selected])
        rows.append(pd.DataFrame(cols))
    return pd.concat(rows, ignore_index=True).merge(pd.DataFrame(block_monitor), on=["chain", "iteration"],
                                                    validate="one_to_one")


def finish_fit(fold, model, chains, data, fd, elements, args, elapsed, settings):
    folder = DIR_CV / f"fold{fold:02d}_{model}"
    if (folder / "fit_status.json").exists():   # a longer rerun replaces the fit; the earlier one moves to superseded/
        old = read_json(folder / "fit_status.json")
        tag = "-".join(str(old.get(k, args.original[k])) for k in ["warmup", "samples", "thin"])
        target = DIR_CV / "superseded" / f"{folder.name}_{tag}"
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        folder.rename(target)
    folder.mkdir(parents=True, exist_ok=True)
    sites, element_scores, block_monitor = score_fit(fold, model, chains, data, fd, elements, args)
    sites.to_csv(folder / "site_scores.csv", index=False)
    element_scores.to_csv(folder / "element_scores.csv", index=False)
    fit_monitors(chains, fd, model, elements, args, block_monitor).to_csv(folder / "monitors.csv", index=False)
    d = diagnose(folder / "monitors.csv", folder / "diagnostics.csv")
    status = dict(completed=True, fold=fold, model=model, n_train=int(fd["train"].sum()), n_test=int(fd["test"].sum()),
                  chain_seconds=elapsed, max_rhat=float(d.rhat.max()), min_ess_bulk=float(d.ess_bulk.min()),
                  min_ess_tail=float(d.ess_tail.min()), worst_rhat_variable=str(d.rhat.idxmax()),
                  heldout_lppd_rhat=float(d.loc["heldout_block_lppd", "rhat"]),
                  heldout_lppd_ess_bulk=float(d.loc["heldout_block_lppd", "ess_bulk"]), lpd_sum=float(sites.lpd.sum()),
                  **settings)
    save_json(folder / "fit_status.json", status)
    print(f"fold {fold:02d} {model}: lpd {status['lpd_sum']:.2f}, max R-hat {status['max_rhat']:.3f} "
          f"({status['worst_rhat_variable']}), held-out lpd R-hat {status['heldout_lppd_rhat']:.3f}", flush=True)


def summarize(models, folds):
    statuses = [read_json(p) for p in sorted(DIR_CV.glob("fold*/fit_status.json"))]
    folders = [DIR_CV / f"fold{s['fold']:02d}_{s['model']}" for s in statuses]
    sites = pd.concat([pd.read_csv(f / "site_scores.csv") for f in folders], ignore_index=True)
    elements = pd.concat([pd.read_csv(f / "element_scores.csv") for f in folders], ignore_index=True)
    convergence = pd.DataFrame(statuses)
    convergence.to_csv(DIR_CV / "cv_convergence.csv", index=False)
    sites.to_csv(DIR_CV / "cv_site_scores.csv", index=False)
    elements.to_csv(DIR_CV / "cv_element_scores.csv", index=False)

    complete = sites.groupby("model").id.nunique()
    models = [m for m in models if m in complete.index and complete[m] == complete.max()]
    sites, elements = sites[sites.model.isin(models)], elements[elements.model.isin(models)]
    rows = []
    for model in models:
        s, e, c = sites[sites.model == model], elements[elements.model == model], convergence[convergence.model == model]
        rows.append(dict(model=model, n_sites=len(s), lpd_mean=s.lpd.mean(), lpd_sum=s.lpd.sum(),
                         energy_score_mean=s.energy_score.mean(), crps_mean=e.crps.mean(), mae_median=e.abs_error_median.mean(),
                         rmse_mean=np.sqrt(e.squared_error_mean.mean()), cover50=e.cover50.mean(), cover90=e.cover90.mean(),
                         brier_elements=e.brier.mean(), brier_union=s.union_brier.mean(), fits=len(c),
                         fits_rhat_above_1p01=int((c.max_rhat > 1.01).sum()),
                         fits_heldout_lppd_rhat_above_1p01=int((c.heldout_lppd_rhat > 1.01).sum()),
                         worst_max_rhat=c.max_rhat.max(), worst_heldout_lppd_rhat=c.heldout_lppd_rhat.max()))
    pd.DataFrame(rows).to_csv(DIR_CV / "cv_model_summary.csv", index=False)

    # paired differences with block-level standard errors
    block = folds.set_index("id").block
    ranked = sorted(models, key=lambda m: -sites[sites.model == m].lpd.sum())
    pair_rows = []
    for i, better in enumerate(ranked):
        for reference in ranked[i + 1:]:
            for score, frame, sign in [("lpd", sites, 1), ("energy_score", sites, -1), ("union_brier", sites, -1),
                                       ("crps", elements, -1), ("brier", elements, -1)]:
                keys = ["id"] if frame is sites else ["id", "element"]
                merged = frame[frame.model == better][keys + [score]].merge(
                    frame[frame.model == reference][keys + [score]], on=keys, suffixes=("_a", "_b")).dropna()
                diff = sign * (merged[f"{score}_a"] - merged[f"{score}_b"])
                by_site = diff.groupby(merged.id).sum()
                by_block = by_site.groupby(block.loc[by_site.index].to_numpy()).sum()
                pair_rows.append(dict(comparison=f"{better} - {reference}", score=score, n_units=len(diff),
                                      n_sites=len(by_site), total_difference=diff.sum(), mean_per_site=by_site.mean(),
                                      se_total_site=np.sqrt(len(by_site)) * by_site.std(ddof=1),
                                      se_total_block=np.sqrt(len(by_block)) * by_block.std(ddof=1),
                                      blocks_favouring_first=int((by_block > 0).sum()), n_blocks=len(by_block)))
    pairwise = pd.DataFrame(pair_rows)
    pairwise.to_csv(DIR_CV / "cv_pairwise.csv", index=False)
    elements.groupby(["model", "element"]).agg(
        crps=("crps", "mean"), mae_median=("abs_error_median", "mean"), rmse=("squared_error_mean", lambda v: np.sqrt(v.mean())),
        cover50=("cover50", "mean"), cover90=("cover90", "mean"), brier=("brier", "mean")).reset_index().to_csv(
        DIR_CV / "cv_by_element.csv", index=False)
    select_model(pairwise, ranked, convergence)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def select_model(pairwise, ranked, convergence):
    lpd = pairwise[pairwise.score == "lpd"].set_index("comparison")
    rhat = convergence.groupby("model").heldout_lppd_rhat.max()
    eligible = [m for m in ranked if rhat.get(m, np.inf) <= 1.01] or list(ranked)
    best = eligible[0]
    within = [best] + [m for m in eligible[1:] if lpd.loc[f"{best} - {m}"].total_difference <= lpd.loc[f"{best} - {m}"].se_total_block]

    def complexity(m):
        spec = parse_model(m)
        return spec["K"], spec["design"] == "use6", spec["robust"]

    chosen = min(within, key=complexity)
    save_json(DIR_CV / "selected_model.json", dict(selected=chosen, **parse_model(chosen), best_lpd_model=best,
                                                   within_one_se=within, excluded_for_convergence=[m for m in ranked if m not in eligible]))
    print(f"selected model: {chosen} (best lpd {best}; within one SE: {within})", flush=True)


def main(args):
    started = time.time()
    DIR_CV.mkdir(parents=True, exist_ok=True)
    spec, data = load_folder(DIR_MODEL_DATA / "main")
    elements = spec["elements"]
    blocks, distance = make_folds(data)
    folds = pd.DataFrame({"id": data["ids"], "block": blocks, "heldout_nearest_training_m": distance})
    folds.to_csv(DIR_CV / "folds.csv", index=False)
    previous = read_json(DIR_CV / "specification.json") if (DIR_CV / "specification.json").exists() else {}
    models = args.models or previous.get("models") or CANDIDATES
    if args.summarize_only:
        summarize(models, folds)
        return
    args.original = {k: previous.get(k, getattr(args, k)) for k in ["warmup", "samples", "thin"]}

    if args.plan:
        plan = read_json(Path(args.plan))
        (DIR_CV / "reruns").mkdir(exist_ok=True)
        save_json(DIR_CV / "reruns" / f"plan_{time.strftime('%Y%m%d_%H%M')}.json", dict(plan=plan, chains=args.chains))
        entries = [dict(model=e["model"], fold=f, warmup=e["warmup"], samples=e["samples"], thin=e["thin"])
                   for e in plan for f in e["folds"]]
    else:
        save_json(DIR_CV / "specification.json", dict(
            models=models, n_blocks=N_BLOCKS, kmeans_seed=BLOCK_SEED, block_sizes=np.bincount(blocks).tolist(),
            heldout_nearest_training_m=dict(zip(["q25", "q50", "q75", "q90"], np.quantile(distance, [0.25, 0.5, 0.75, 0.9]).round(1))),
            warmup=args.warmup, samples=args.samples, thin=args.thin, chains=args.chains, score_stride=args.score_stride,
            weight_draws=args.weight_draws))
        entries = [dict(model=m, fold=f, warmup=args.warmup, samples=args.samples, thin=args.thin)
                   for f in (args.folds or range(N_BLOCKS)) for m in models]

    fold_inputs = {fold: fold_data(data, blocks, fold, elements) for fold in sorted({e["fold"] for e in entries})}
    tasks, settings = [], {}
    for e in entries:
        fold, model = e["fold"], e["model"]
        length = {k: e[k] for k in ["warmup", "samples", "thin"]}
        status = DIR_CV / f"fold{fold:02d}_{model}" / "fit_status.json"
        if status.exists() and (not args.plan or all(read_json(status).get(k) == v for k, v in length.items())):
            continue
        fd, spec_model = fold_inputs[fold], parse_model(model)
        train = fd["train"]
        settings[(fold, model)] = length
        for c in range(1, args.chains + 1):
            tasks.append(dict(Y=fd["Y"][train], coords=data["coords"][train], X=design_matrix(data, spec_model["design"])[train],
                              groups=fd["groups"][train], K=spec_model["K"], robust=spec_model["robust"],
                              censored=fd["censored"][train], lower=fd["lower"][train], fold=fold, model=model, chain=c,
                              seed=20261200 + 10000 * c + 100 * fold + 10 * models.index(model), **length))
    print(f"{len(tasks)} chains for {len(tasks) // args.chains} fits", flush=True)

    pending, seconds, failures = {}, {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(chain_task, t): (t["fold"], t["model"], t["chain"]) for t in tasks}
        for future in as_completed(futures):
            try:
                fold, model, chain, arrays, elapsed = future.result()
            except Exception as error:   # the fit stays unfinished; a rerun redoes only that fit
                failures.append(dict(zip(["fold", "model", "chain"], futures[future]), error=repr(error)))
                print(f"FAILED chain {futures[future]}: {error!r}", flush=True)
                continue
            key = (fold, model)
            pending.setdefault(key, []).append((chain, arrays))
            seconds.setdefault(key, []).append(elapsed)
            if len(pending[key]) == args.chains:
                try:
                    finish_fit(fold, model, sorted(pending.pop(key), key=lambda item: item[0]), data, fold_inputs[fold],
                               elements, args, seconds[key], settings[key])
                except Exception as error:
                    failures.append(dict(fold=fold, model=model, chain=None, error=repr(error)))
                    print(f"FAILED scoring fold {fold} {model}: {error!r}", flush=True)
    if failures:
        pd.DataFrame(failures).to_csv(DIR_CV / "failures.csv", index=False)
    summarize(models, folds)
    print(f"cross-validation finished in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None, help="default: the candidates in blmc/config.py")
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="subset of folds, for quick tests")
    parser.add_argument("--plan", default=None, help="JSON list of {model, folds, warmup, samples, thin}")
    parser.add_argument("--warmup", type=int, default=CV_ITERATIONS["warmup"])
    parser.add_argument("--samples", type=int, default=CV_ITERATIONS["samples"])
    parser.add_argument("--thin", type=int, default=CV_ITERATIONS["thin"])
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--score-stride", type=int, default=2)
    parser.add_argument("--weight-draws", type=int, default=4)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.plan and args.models:
        parser.error("--plan reruns fits of the existing specification; do not pass --models")
    with threadpool_limits(limits=1):
        main(args)
