# R-hat and bulk/tail ESS of the monitored quantities.
import warnings

import arviz as az
import numpy as np
import pandas as pd


def diagnose(monitors_path, out_path):
    monitors = pd.read_csv(monitors_path).sort_values(["chain", "iteration"])
    chains = monitors.chain.unique()
    rows = []
    for variable in monitors.columns.drop(["chain", "iteration"]):
        draws = np.stack([monitors.loc[monitors.chain == c, variable].to_numpy(dtype=float) for c in chains])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rhat, bulk, tail = az.rhat(draws), az.ess(draws, method="bulk"), az.ess(draws, method="tail")
            mcse = az.mcse(draws, method="mean")
        rows.append(dict(variable=variable, mean=draws.mean(), sd=draws.std(ddof=1), median=np.median(draws),
                         q5=np.quantile(draws, 0.05), q95=np.quantile(draws, 0.95), mcse_mean=float(np.squeeze(mcse)),
                         rhat=float(rhat), ess_bulk=float(bulk), ess_tail=float(tail)))
    out = pd.DataFrame(rows)
    out["n_chains"], out["draws_per_chain"] = len(chains), draws.shape[1]
    out["flag"] = np.where(~np.isfinite(out[["rhat", "ess_bulk", "ess_tail"]]).all(axis=1), "undefined",
                           np.where(out.rhat > 1.01, "rhat_above_1.01",
                                    np.where((out.ess_bulk < 400) | (out.ess_tail < 400), "ess_below_400", "ok")))
    out.to_csv(out_path, index=False)
    return out.set_index("variable")
