# Numbers of the article's tables, one CSV per table.
import numpy as np
import pandas as pd

from blmc.config import (CREDIBLE, DIR_AREAS, DIR_CHECKS, DIR_CV, DIR_FITS, DIR_MAPS, DIR_SENSITIVITY, DIR_TABLES, FOCUS,
                         GROUPS, MINOR, NGR_USES, SELECTED_MODEL, SENSITIVITIES)
from blmc.io import read_json
from blmc.model import parse_model

CALIBRATION_BINS = [0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0001]
COVER = f"cover{round(100 * CREDIBLE)}"
REFITS = [f"sensitivity/{name}" for name in SENSITIVITIES]


def params(fit):
    d = pd.read_csv(DIR_FITS / fit / "diagnostics.csv").set_index("variable")
    names = ["nu"] + [f"local_variance_ratio_{g}_vs_Agric" for g in GROUPS[1:]] + [f"lengthscale_{k}_km" for k in (1, 2, 3)]
    return d.loc[names, ["median", "lower", "upper"]].reset_index(names="parameter")


def areas(fit):
    out = pd.read_csv(DIR_MAPS / fit / "grid_predictions.csv")
    rows = []
    for event in FOCUS + ["union"] + MINOR + ["minor"]:
        row = {"event": event}
        for name in ["level", "sample"]:
            row[f"{name}_all"] = 100 * out[f"p_{name}_{event}"].mean()
            for use in NGR_USES:
                row[f"{name}_{use}"] = 100 * out.loc[out.ngr_use == use, f"p_{name}_{event}"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def area_intervals(fit):
    summary = pd.read_csv(DIR_AREAS / fit / "area_summary.csv")
    summary = summary[summary.scope.eq("all") & summary.event.isin(FOCUS + ["union"])]
    return summary[["event"] + [f"{q}_{s}" for q in ["level", "sample", "difference"] for s in ["mean", "lower", "upper"]]]


def cv_scores(model):
    elements = pd.read_csv(DIR_CV / "cv_element_scores.csv")
    sites = pd.read_csv(DIR_CV / "cv_site_scores.csv")
    return elements[elements.model.eq(model)].dropna(subset=["exceed_observed"]), \
        sites[sites.model.eq(model)].dropna(subset=["union_observed"])


def posterior_checks(fit, model):
    replicated = pd.read_csv(DIR_FITS / fit / "posterior_replication_summary.csv")
    replicated = replicated[replicated.scope.eq("all")].set_index("event")
    elements, sites = cv_scores(model)
    predicted = elements.groupby("element").exceed_p.sum()
    predicted["union"] = sites.union_p.sum()
    out = pd.read_csv(DIR_MAPS / fit / "grid_predictions.csv")
    rows = []
    for event in FOCUS + ["union", "multiple"] + MINOR:
        row = dict(event=event, **replicated.loc[event, ["observed_count", "replicated_mean", "replicated_lower",
                                                         "replicated_upper"]].to_dict(),
                   cross_validated=predicted.get(event, np.nan))
        if event in MINOR:
            row.update(level_pct=100 * out[f"p_level_{event}"].mean(), sample_pct=100 * out[f"p_sample_{event}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def calibration(model):
    elements, sites = cv_scores(model)
    elements = elements[elements.element.isin(FOCUS)]
    rows = []
    for label, frame, p, o in [("elements", elements, "exceed_p", "exceed_observed"), ("union", sites, "union_p", "union_observed")]:
        bins = pd.cut(frame[p], CALIBRATION_BINS, right=False)
        grouped = frame.groupby(bins, observed=False).agg(n=(p, "size"), expected=(p, "sum"), observed=(o, "sum"))
        grouped.loc["total"] = [len(frame), frame[p].sum(), frame[o].sum()]
        brier = ((frame[p] - frame[o]) ** 2).mean()
        rate = frame[o].mean()
        grouped["brier_total"] = np.where(grouped.index == "total", brier, np.nan)
        grouped["brier_skill_total"] = np.where(grouped.index == "total", 1 - brier / (rate * (1 - rate)), np.nan)
        rows.append(grouped.reset_index(names="predicted").assign(scope=label))
    return pd.concat(rows, ignore_index=True)


def cv_models():
    summary = pd.read_csv(DIR_CV / "cv_model_summary.csv").set_index("model")
    pairwise = pd.read_csv(DIR_CV / "cv_pairwise.csv")
    pairwise = pairwise[pairwise.score.eq("lpd")].set_index("comparison")
    selected = read_json(DIR_CV / "selected_model.json")
    best = selected["best_lpd_model"]
    rows = []
    for model, row in summary.iterrows():
        spec = parse_model(model) if model != "univariate" else dict(robust=True, design="ngr", K=1)
        comparison = pairwise.loc[f"{best} - {model}"] if model != best else None
        rows.append(dict(model=model, errors="Student-t" if spec["robust"] else "Gaussian", mean=spec["design"],
                         K=spec["K"], p=3 if spec["design"] == "ngr" else 5, lpd=row.lpd_sum,
                         delta_lpd=np.nan if comparison is None else -comparison.total_difference,
                         se_block=np.nan if comparison is None else comparison.se_total_block,
                         blocks_favouring_best=np.nan if comparison is None else comparison.blocks_favouring_first,
                         crps=row.crps_mean, coverage=row[COVER], brier_union=row.brier_union,
                         selected=model == selected["selected"]))
    return pd.DataFrame(rows)


def cv_statuses():
    specification = read_json(DIR_CV / "specification.json")
    statuses = pd.DataFrame([read_json(p) for p in sorted(DIR_CV.glob("fold*/fit_status.json"))])
    statuses["iterations"] = statuses.warmup + statuses.samples * statuses.thin
    order = [m for m in specification["models"] if m in set(statuses.model)]
    return statuses, order + sorted(set(statuses.model) - set(order))


def convergence(fit):
    rows = []
    for name in [fit] + REFITS:
        d = pd.read_csv(DIR_FITS / name / "diagnostics.csv")
        spec = read_json(DIR_FITS / name / "specification.json")
        rows.append(dict(fit=name, draws=f"{spec['chains']}x{spec['samples']}", max_rhat=d.rhat.max(),
                         min_ess_bulk=d.ess_bulk.min(), min_ess_tail=d.ess_tail.min()))
    statuses, order = cv_statuses()
    for model in order:
        part = statuses[statuses.model == model]
        rows.append(dict(fit=f"cv/{model}", draws=f"4x{int(part.samples.min())}-{int(part.samples.max())}",
                         max_rhat=part.max_rhat.max(), min_ess_bulk=part.min_ess_bulk.min(),
                         min_ess_tail=part.min_ess_tail.min(), lpd_rhat=part.heldout_lppd_rhat.max(),
                         lpd_ess_bulk=part.heldout_lppd_ess_bulk.min(), lengthened_fits=int((part.samples > part.samples.min()).sum())))
    return pd.DataFrame(rows)


def timing(fit):
    rows = []
    statuses, order = cv_statuses()
    for model in order:
        part = statuses[statuses.model == model]
        seconds = np.concatenate(part.chain_seconds.to_list())
        iterations = np.repeat(part.iterations.to_numpy(), [len(s) for s in part.chain_seconds])
        rows.append(dict(fit=f"cv/{model}", chains=len(seconds), iterations_median=int(np.median(iterations)),
                         seconds_median=np.median(seconds), seconds_min=seconds.min(), seconds_max=seconds.max(),
                         seconds_per_1000=1000 * seconds.sum() / iterations.sum(), chain_hours=seconds.sum() / 3600))
    for name in [fit] + REFITS:
        spec = read_json(DIR_FITS / name / "specification.json")
        seconds = np.array([read_json(DIR_FITS / name / f"chain_{c}.json")["elapsed_seconds"] for c in range(1, spec["chains"] + 1)])
        iterations = spec["warmup"] + spec["samples"] * spec["thin"]
        rows.append(dict(fit=name, chains=len(seconds), iterations_median=iterations, seconds_median=seconds.mean(),
                         seconds_min=seconds.min(), seconds_max=seconds.max(),
                         seconds_per_1000=1000 * seconds.mean() / iterations, chain_hours=seconds.sum() / 3600))
    return pd.DataFrame(rows)


def area_sensitivity():
    review = pd.read_csv(DIR_SENSITIVITY / "review" / "sensitivity_map_areas.csv").set_index("fit")
    refits = review.drop(index=SELECTED_MODEL)
    return pd.DataFrame([dict(column=c, selected=review.loc[SELECTED_MODEL, c], minimum=refits[c].min(),
                              maximum=refits[c].max(), **refits[c].to_dict()) for c in review.columns])


# local scales by element group, and Tl in marsh and agricultural cells, against the selected fit
def element_groups():
    parameters = pd.read_csv(DIR_SENSITIVITY / "review" / "sensitivity_parameters.csv")
    parameters = parameters[parameters.fit.isin([SELECTED_MODEL, "sensitivity/element_groups"])
                            & parameters.parameter.str.startswith("local_variance_ratio_")]
    rows = [dict(quantity=r.parameter, fit=r.fit, median=r.median, lower=r.lower, upper=r.upper) for r in parameters.itertuples()]
    for fit in [SELECTED_MODEL, "sensitivity/element_groups"]:
        out = pd.read_csv(DIR_MAPS / fit / "grid_predictions.csv", usecols=["land_use", "p_level_Tl", "p_sample_Tl"])
        for use in ["Marsh", "Agric"]:
            cells = out[out.land_use == use]
            rows += [dict(quantity=f"Tl_{name}_pct_{use}", fit=fit, median=100 * cells[f"p_{name}_Tl"].mean())
                     for name in ["level", "sample"]]
    return pd.DataFrame(rows)


def residual_ratios(fit):
    ratios = pd.read_csv(DIR_CHECKS / fit / "residual_variance_ratio_summary.csv")
    return ratios[ratios.element.isin(FOCUS)]


def main():
    DIR_TABLES.mkdir(parents=True, exist_ok=True)
    model = read_json(DIR_CV / "selected_model.json")["selected"]
    fit = SELECTED_MODEL
    tables = {"tab_params": lambda: params(fit), "tab_areas": lambda: areas(fit),
              "tab_area_intervals": lambda: area_intervals(fit), "tab_ppc_other_elements": lambda: posterior_checks(fit, model),
              "tab_calibration": lambda: calibration(model), "tab_cv_appendix": cv_models,
              "tab_convergence_all": lambda: convergence(fit), "tab_timing": lambda: timing(fit),
              "tab_area_sensitivity": area_sensitivity, "element_group_scales": element_groups,
              "residual_variance_ratios": lambda: residual_ratios(fit)}
    for name, build in tables.items():
        table = build()
        table.to_csv(DIR_TABLES / f"{name}.csv", index=False)
        print(f"\n{name}\n{table.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
