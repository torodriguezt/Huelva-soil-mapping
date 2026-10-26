# Helpers shared by the full-data fits and the cross-validation.
import numpy as np

from blmc.config import UPPER_LIMITS


def parse_model(name):
    errors, factors, design = name.split("_")
    if errors not in ("t", "gauss") or not factors.startswith("K") or design not in ("ngr", "use6"):
        raise ValueError(f"model names look like t_K3_ngr or gauss_K4_use6, not {name}")
    return dict(K=int(factors[1:]), robust=errors == "t", design=design)


def design_matrix(data, design):
    return data["X_use6"] if design == "use6" else data["X"]


def censoring(Y, center, scale, elements):
    censored, lower = np.zeros(Y.shape, dtype=bool), np.full(Y.shape, np.nan)
    for element, limit in UPPER_LIMITS.items():
        j = elements.index(element)
        bound = (np.log(limit) - center[j]) / scale[j]
        hit = Y[:, j] >= bound - 1e-9
        censored[hit, j], lower[hit, j] = True, bound
    return censored, lower

