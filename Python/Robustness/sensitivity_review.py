# Comparison of the selected fit with its sensitivity refits.
import numpy as np
import pandas as pd

from blmc.config import DIR_FITS, DIR_MAPS, DIR_SENSITIVITY, INTERVAL, SELECTED_MODEL, SENSITIVITIES
from blmc.covariance import to_correlation
from blmc.io import load_chains, load_fit

PAIRS = [("Pb", "Al"), ("Cu", "V"), ("As", "V"), ("As", "Tl"), ("As", "Hg"), ("Hg", "Pb"), ("Co", "V")]
EVENTS = ["union", "multiple", "As", "Tl", "V", "Pb", "Cu"]
LEVELS = [INTERVAL[0], 0.5, INTERVAL[1]]


def spatial_correlations(fit, elements):
    loading = np.concatenate([draws["loading"] for _, draws in load_chains(fit, ["loading"], stride=8)])
    R = to_correlation(np.einsum("skj,skl->sjl", loading, loading))
    eigenvalues = np.sort(np.linalg.eigvalsh(np.median(R, axis=0)))[::-1][:3]
    pairs = {f"{a}-{b}": np.quantile(R[:, elements.index(a), elements.index(b)], LEVELS) for a, b in PAIRS}
    return pairs, eigenvalues


def main():
    out = DIR_SENSITIVITY / "review"
    out.mkdir(parents=True, exist_ok=True)
    fits = [SELECTED_MODEL] + [f"sensitivity/{name}" for name in SENSITIVITIES]
    elements = load_fit(SELECTED_MODEL)[0]["elements"]
    reference = pd.read_csv(DIR_FITS / SELECTED_MODEL / "site_weights.csv").set_index("id").inflation_median
    summary, parameters, correlations, replication, areas = [], [], [], [], []
    for fit in fits:
        d = pd.read_csv(DIR_FITS / fit / "diagnostics.csv").set_index("variable")
        row = dict(fit=fit, max_rhat=d.rhat.max(), min_ess_bulk=d.ess_bulk.min(), min_ess_tail=d.ess_tail.min())
        weights = pd.read_csv(DIR_FITS / fit / "site_weights.csv").set_index("id")
        shared = weights.index.intersection(reference.index)
        row["spearman_inflation_vs_selected"] = weights.inflation_median.loc[shared].corr(reference.loc[shared], method="spearman")
        row["top6_sites"] = " ".join(str(i) for i in weights.index[:6])
        pairs, eigenvalues = spatial_correlations(fit, elements)
        row.update({f"eigenvalue_{k + 1}": value for k, value in enumerate(eigenvalues)})
        summary.append(row)
        names = [v for v in d.index if v == "nu" or v.startswith(("local_variance_ratio_", "lengthscale_"))]
        parameters += [dict(fit=fit, parameter=v, median=d.loc[v, "median"], lower=d.loc[v, "lower"], upper=d.loc[v, "upper"])
                       for v in names]
        correlations += [dict(fit=fit, pair=pair, lower=q[0], median=q[1], upper=q[2]) for pair, q in pairs.items()]

        replicated = pd.read_csv(DIR_FITS / fit / "posterior_replication_summary.csv")
        replicated = replicated[replicated.scope == "all"].set_index("event")
        replication += [dict(fit=fit, event=e, observed=replicated.loc[e, "observed_count"],
                             replicated_mean=replicated.loc[e, "replicated_mean"],
                             replicated_lower=replicated.loc[e, "replicated_lower"],
                             replicated_upper=replicated.loc[e, "replicated_upper"]) for e in EVENTS]

        area = pd.read_csv(DIR_MAPS / fit / "area_draws.csv")
        areas.append({"fit": fit, **{f"{name}_{event}_pct": 100 * area[f"{name}_{event}"].mean()
                                     for name in ["level", "sample"] for event in ["union", "As", "V", "Tl"]}})

    pd.DataFrame(summary).to_csv(out / "sensitivity_summary.csv", index=False)
    pd.DataFrame(parameters).to_csv(out / "sensitivity_parameters.csv", index=False)
    pd.DataFrame(correlations).to_csv(out / "sensitivity_correlations.csv", index=False)
    pd.DataFrame(replication).to_csv(out / "sensitivity_replication.csv", index=False)
    areas = pd.DataFrame(areas)
    areas.to_csv(out / "sensitivity_map_areas.csv", index=False)
    print(areas.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
