# Helpers shared by the full-data fits and the cross-validation.
import numpy as np

from blmc.config import ELEMENT_GROUPS, UPPER_LIMITS


def element_group_index(elements):
    names = list(ELEMENT_GROUPS)
    return np.array([next(names.index(g) for g, members in ELEMENT_GROUPS.items() if e in members) for e in elements])


def local_scales(group_variance, groups, element_group=None):
    """a for every site and element, (..., n, q); (..., n, 1) when one scale is shared by all elements."""
    a = np.asarray(group_variance)
    if element_group is None:
        return a[..., groups][..., None]
    return a[..., groups, :][..., np.asarray(element_group)]


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

