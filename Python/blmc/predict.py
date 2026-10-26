import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.spatial.distance import cdist

from blmc.covariance import matern32


def conditional_predict(draw, coords_obs, coords_new, X_new, chunk_size=1000, jitter=1e-8):
    obs, new = np.asarray(coords_obs, dtype=float), np.asarray(coords_new, dtype=float)
    beta, loading, rho, F = (np.asarray(draw[key], dtype=float) for key in ["beta", "loading", "rho", "F"])
    K, m = loading.shape[0], len(new)

    distance = cdist(obs, obs)
    chols, weights = [], []
    for k in range(K):
        chol = cholesky(matern32(distance, rho[k]) + jitter * np.eye(len(obs)), lower=True, check_finite=False)
        chols.append(chol)
        weights.append(cho_solve((chol, True), F[:, k], check_finite=False))

    mean_f, var_f = np.empty((m, K)), np.empty((m, K))
    for start in range(0, m, chunk_size):
        end = min(m, start + chunk_size)
        cross_distance = cdist(new[start:end], obs)
        for k in range(K):
            cross = matern32(cross_distance, rho[k])
            mean_f[start:end, k] = cross @ weights[k]
            whitened = solve_triangular(chols[k], cross.T, lower=True, check_finite=False)
            var_f[start:end, k] = np.maximum(1.0 - np.einsum("ij,ij->j", whitened, whitened), 0.0)

    return {"mean_y": np.asarray(X_new, dtype=float) @ beta + mean_f @ loading, "mean_f": mean_f, "var_f": var_f}
