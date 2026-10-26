# Comparison of the selected fit with its sensitivity refits.
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from blmc.config import DIR_FITS, DIR_SENSITIVITY, SELECTED_MODEL, SENSITIVITIES
from blmc.covariance import to_correlation
from blmc.io import load_chains, load_fit, read_grid
from Results.grid_predictions import predict_grid

PARAMETERS = ["nu", "local_variance_ratio_Phospho_vs_Agric", "local_variance_ratio_Marsh_vs_Agric",
              "local_variance_ratio_Industria_vs_Agric", "local_variance_ratio_Urban_vs_Agric",
              "local_variance_ratio_Bare_vs_Agric", "lengthscale_1_km", "lengthscale_2_km", "lengthscale_3_km"]
PAIRS = [("Pb", "Al"), ("Cu", "V"), ("As", "V"), ("As", "Tl"), ("As", "Hg"), ("Hg", "Pb"), ("Co", "V")]
EVENTS = ["union", "multiple", "As", "Tl", "V", "Pb", "Cu"]


def spatial_correlations(fit, elements):
    loading = np.concatenate([draws["loading"] for _, draws in load_chains(fit, ["loading"], stride=8)])
    R = to_correlation(np.einsum("skj,skl->sjl", loading, loading))
    eigenvalues = np.sort(np.linalg.eigvalsh(np.median(R, axis=0)))[::-1][:3]
    pairs = {f"{a}-{b}": np.quantile(R[:, elements.index(a), elements.index(b)], [0.05, 0.5, 0.95]) for a, b in PAIRS}
    return pairs, eigenvalues


def main():
    out = DIR_SENSITIVITY / "review"
    out.mkdir(parents=True, exist_ok=True)
    fits = [SELECTED_MODEL] + [f"sensitivity/{name}" for name in SENSITIVITIES]
    elements = load_fit(SELECTED_MODEL)[0]["elements"]
    grid = read_grid()
    reference = pd.read_csv(DIR_FITS / SELECTED_MODEL / "site_weights.csv").set_index("id").inflation_median
    summary, correlations, replication, areas = [], [], [], []
    for fit in fits:
        d = pd.read_csv(DIR_FITS / fit / "diagnostics.csv").set_index("variable")
        row = dict(fit=fit, max_rhat=d.rhat.max(), min_ess_bulk=d.ess_bulk.min(), min_ess_tail=d.ess_tail.min())
        for variable in PARAMETERS:
            row.update({f"{variable}_median": d.loc[variable, "median"], f"{variable}_q05": d.loc[variable, "q5"],
                        f"{variable}_q95": d.loc[variable, "q95"]})
        weights = pd.read_csv(DIR_FITS / fit / "site_weights.csv").set_index("id")
        shared = weights.index.intersection(reference.index)
        row["spearman_inflation_vs_selected"] = weights.inflation_median.loc[shared].corr(reference.loc[shared], method="spearman")
        row["top6_sites"] = " ".join(str(i) for i in weights.index[:6])
        pairs, eigenvalues = spatial_correlations(fit, elements)
        row.update({f"eigenvalue_{k + 1}": value for k, value in enumerate(eigenvalues)})
        summary.append(row)
        correlations += [dict(fit=fit, pair=pair, q05=q[0], median=q[1], q95=q[2]) for pair, q in pairs.items()]

        replicated = pd.read_csv(DIR_FITS / fit / "posterior_replication_summary.csv")
        replicated = replicated[replicated.scope == "all"].set_index("event")
        replication += [dict(fit=fit, event=e, observed=replicated.loc[e, "observed_count"],
                             replicated_mean=replicated.loc[e, "replicated_mean"],
                             replicated_q05=replicated.loc[e, "replicated_q05"],
                             replicated_q95=replicated.loc[e, "replicated_q95"]) for e in EVENTS]

        _, area, _ = predict_grid(fit, grid, 50, np.random.default_rng(20261024))
        areas.append({"fit": fit, **{f"{name}_{event}_pct": 100 * area[f"{name}_{event}"].mean()
                                     for name in ["level", "sample"] for event in ["union", "As", "V", "Tl"]}})
        print(f"{fit}: done", flush=True)

    pd.DataFrame(summary).to_csv(out / "sensitivity_summary.csv", index=False)
    pd.DataFrame(correlations).to_csv(out / "sensitivity_correlations.csv", index=False)
    pd.DataFrame(replication).to_csv(out / "sensitivity_replication.csv", index=False)
    areas = pd.DataFrame(areas)
    areas.to_csv(out / "sensitivity_map_areas.csv", index=False)
    print(areas.round(1).to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
