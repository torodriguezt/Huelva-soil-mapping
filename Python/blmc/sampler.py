t# Gibbs sampler of the robust spatial factor model (Zhang and Banerjee 2022, extended).
import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.spatial.distance import cdist
from scipy.special import gammaln
from scipy.stats import invwishart, truncnorm
from threadpoolctl import threadpool_limits

from blmc.covariance import matern32

KEYS = ["beta", "loading", "Sigma", "rho", "F", "group_variance", "weights", "nu", "censored_values", "log_likelihood"]


@dataclass(frozen=True)
class Config:
    robust: bool = True
    beta_prior_variance: float = 10.0
    loading_prior_variance: float = 1.0
    iw_df_offset: float = 2.0
    residual_prior_mean: float = 0.3
    log_rho_mean: float = float(np.log(2.0))
    log_rho_sd: float = 0.8
    group_log_variance_prior_sd: float = 1.0
    nu_prior_shape: float = 2.0
    nu_prior_rate: float = 0.1
    nu_initial_median: float = 10.0
    rho_proposal_sd: float = 0.3
    group_proposal_sd: float = 0.2
    amplitude_proposal_sd: float = 0.2
    nu_proposal_sd: float = 0.3
    adapt_interval: int = 50
    adapt_target: float = 0.44
    jitter: float = 1e-8


class Proposal:

    def __init__(self, dimension, sd, lower=0.005):
        self.sd = np.full(dimension, sd, dtype=float)
        self.warm = np.zeros(dimension, dtype=int)
        self.sample = np.zeros(dimension, dtype=int)
        self.batch = np.zeros(dimension, dtype=int)
        self.lower = lower

    def record(self, accepted, warmup):
        if warmup:
            self.warm += accepted
            self.batch += accepted
        else:
            self.sample += accepted

    def adapt(self, iteration, config, active=None):
        if active is None:
            active = np.ones(len(self.sd), dtype=bool)
        gain = min(0.5, 1.0 / np.sqrt(iteration // config.adapt_interval))
        rate = self.batch[active] / config.adapt_interval
        self.sd[active] = np.clip(self.sd[active] * np.exp(gain * (rate - config.adapt_target)), self.lower, 1.5)
        self.batch[:] = 0


def correlation_state(distance, rho, jitter):
    matrix = matern32(distance, rho)
    matrix.flat[:: matrix.shape[0] + 1] += jitter
    chol = cholesky(matrix, lower=True, check_finite=False)
    return chol, float(2.0 * np.log(np.diag(chol)).sum())


def row_precision_without_beta(X, row_variance, beta_prior_variance):
    inverse_variance = 1.0 / np.asarray(row_variance)
    weighted_X = inverse_variance[:, None] * X
    small = cholesky(X.T @ weighted_X + np.eye(X.shape[1]) / beta_prior_variance, lower=True, check_finite=False)
    precision = np.diag(inverse_variance) - weighted_X @ cho_solve((small, True), weighted_X.T, check_finite=False)
    return (precision + precision.T) * 0.5


def factor_precision(Y, loading, Sigma, R_inverse, row_precision):
    n, K = Y.shape[0], loading.shape[0]
    Sigma_inv_loading_t = cho_solve((cholesky(Sigma, lower=True, check_finite=False), True), loading.T, check_finite=False)
    precision = np.kron(loading @ Sigma_inv_loading_t, row_precision)
    for k in range(K):
        block = slice(k * n, (k + 1) * n)
        precision[block, block] += R_inverse[k]
    precision = (precision + precision.T) * 0.5
    rhs = ((row_precision @ Y) @ Sigma_inv_loading_t).ravel(order="F")
    return precision, rhs


def collapsed_term(precision, rhs, log_det_R):
    chol = cholesky(precision, lower=True, check_finite=False)
    quadratic = rhs @ cho_solve((chol, True), rhs, check_finite=False)
    return float(-0.5 * (log_det_R + 2.0 * np.log(np.diag(chol)).sum() - quadratic))


def sample_factors(Y, loading, Sigma, R_inverse, row_precision, rng):
    precision, rhs = factor_precision(Y, loading, Sigma, R_inverse, row_precision)
    chol = cholesky(precision, lower=True, check_finite=False)
    mean = cho_solve((chol, True), rhs, check_finite=False)
    noise = solve_triangular(chol.T, rng.normal(size=rhs.size), lower=False, check_finite=False)
    return (mean + noise).reshape((Y.shape[0], loading.shape[0]), order="F")


def sample_mniw(Y, H, prior_precision, nu0, Psi0, rng):
    precision_chol = cholesky(H.T @ H + np.diag(prior_precision), lower=True, check_finite=False)
    mean = cho_solve((precision_chol, True), H.T @ Y, check_finite=False)
    residual = Y - H @ mean
    scale = Psi0 + residual.T @ residual + mean.T @ (prior_precision[:, None] * mean)
    scale = (scale + scale.T) * 0.5
    Sigma = np.atleast_2d(invwishart.rvs(df=nu0 + Y.shape[0], scale=scale, random_state=rng))
    Sigma = (Sigma + Sigma.T) * 0.5
    sigma_chol = cholesky(Sigma, lower=True, check_finite=False)
    noise = solve_triangular(precision_chol.T, rng.normal(size=mean.shape), lower=False, check_finite=False)
    return mean + noise @ sigma_chol.T, Sigma


def update_ranges(Y, loading, Sigma, distance, log_rho, chols, log_dets, R_inverse, sd, config, rng, row_precision):
    K, n, _ = R_inverse.shape
    identity = np.eye(n)
    accepted = np.zeros(K, dtype=bool)
    precision, rhs = factor_precision(Y, loading, Sigma, R_inverse, row_precision)
    current_ll = collapsed_term(precision, rhs, np.sum(log_dets))
    for k in range(K):
        proposed = log_rho[k] + rng.normal(scale=sd[k])
        current = current_ll - 0.5 * ((log_rho[k] - config.log_rho_mean) / config.log_rho_sd) ** 2
        try:
            with np.errstate(over="ignore", under="ignore"):
                candidate_rho = np.exp(proposed)
            if not np.isfinite(candidate_rho) or candidate_rho <= 0:
                continue
            chol, log_det = correlation_state(distance, candidate_rho, config.jitter)
            inverse = cho_solve((chol, True), identity, check_finite=False)
            candidate_precision = precision.copy()
            block = slice(k * n, (k + 1) * n)
            candidate_precision[block, block] += inverse - R_inverse[k]
            candidate_precision = (candidate_precision + candidate_precision.T) * 0.5
            candidate_ll = collapsed_term(candidate_precision, rhs, np.sum(log_dets) - log_dets[k] + log_det)
            target = candidate_ll - 0.5 * ((proposed - config.log_rho_mean) / config.log_rho_sd) ** 2
        except np.linalg.LinAlgError:
            continue
        if np.log(rng.uniform()) < target - current:
            accepted[k] = True
            log_rho[k] = proposed
            chols[k], log_dets[k], R_inverse[k] = chol, log_det, inverse
            precision, current_ll = candidate_precision, candidate_ll
    return accepted


def update_amplitudes(F, loading, R_inverse, Sigma, sd, config, rng):
    # same likelihood; the proposal has Jacobian exp((n - q) u)
    accepted = np.zeros(loading.shape[0], dtype=bool)
    sigma_chol = cholesky(Sigma, lower=True, check_finite=False)
    for k in range(loading.shape[0]):
        u = rng.normal(scale=sd[k])
        factor_quad = F[:, k] @ R_inverse[k] @ F[:, k]
        loading_quad = loading[k] @ cho_solve((sigma_chol, True), loading[k], check_finite=False) / config.loading_prior_variance
        with np.errstate(over="ignore", invalid="ignore"):
            log_ratio = -0.5 * (np.expm1(2.0 * u) * factor_quad + np.expm1(-2.0 * u) * loading_quad) \
                + (F.shape[0] - loading.shape[1]) * u
        if np.isfinite(log_ratio) and np.log(rng.uniform()) < log_ratio:
            F[:, k] *= np.exp(u)
            loading[k] /= np.exp(u)
            accepted[k] = True
    return accepted


def site_quadratics(residual, Sigma):
    chol = cholesky(Sigma, lower=True, check_finite=False)
    return np.sum(residual * cho_solve((chol, True), residual.T, check_finite=False).T, axis=1)


def update_groups(weighted_quadratics, groups, eta, counts, q, sd, config, rng):
    group_quadratics = np.bincount(groups, weights=weighted_quadratics, minlength=len(eta))

    def target(value, g):
        with np.errstate(over="ignore", invalid="ignore"):
            return float(-0.5 * (counts[g] * q * value + group_quadratics[g] * np.exp(-value)
                                 + (value / config.group_log_variance_prior_sd) ** 2))

    accepted = np.zeros(len(eta), dtype=bool)
    for g in range(1, len(eta)):
        candidate = eta[g] + rng.normal(scale=sd[g])
        with np.errstate(over="ignore", under="ignore"):
            variance = np.exp(candidate)
        if not np.isfinite(variance) or variance <= 0:
            continue
        proposed = target(candidate, g)
        if np.isfinite(proposed) and np.log(rng.uniform()) < proposed - target(eta[g], g):
            eta[g] = candidate
            accepted[g] = True
    return accepted


def log_nu_target(log_nu, mahalanobis, q, config):
    with np.errstate(over="ignore", under="ignore"):
        nu = np.exp(log_nu)
    if not np.isfinite(nu) or nu <= 0:
        return -np.inf
    n = len(mahalanobis)
    loglik = n * (gammaln(0.5 * (nu + q)) - gammaln(0.5 * nu) - 0.5 * q * np.log(nu)) \
        - 0.5 * (nu + q) * np.sum(np.log1p(mahalanobis / nu))
    return float(loglik) + config.nu_prior_shape * log_nu - config.nu_prior_rate * nu


def update_nu(log_nu, mahalanobis, q, sd, config, rng):
    candidate = log_nu + rng.normal(scale=sd)
    current = log_nu_target(log_nu, mahalanobis, q, config)
    proposed = log_nu_target(candidate, mahalanobis, q, config)
    if np.isfinite(proposed) and np.log(rng.uniform()) < proposed - current:
        return candidate, True
    return log_nu, False


def impute_censored(Y, mean, Sigma, row_variance, rows, cols, lower, rng):
    q = Y.shape[1]
    for r, c, bound in zip(rows, cols, lower):
        others = np.arange(q) != c
        coefficient = np.linalg.solve(Sigma[np.ix_(others, others)], Sigma[c, others])
        m = mean[r, c] + coefficient @ (Y[r, others] - mean[r, others])
        s = np.sqrt(row_variance[r] * (Sigma[c, c] - coefficient @ Sigma[c, others]))
        Y[r, c] = m + s * truncnorm.rvs((bound - m) / s, np.inf, random_state=rng)


def log_likelihood(residual, Sigma, row_scale, nu=None):
    q = residual.shape[1]
    chol = cholesky(Sigma, lower=True, check_finite=False)
    d = np.sum(residual * cho_solve((chol, True), residual.T, check_finite=False).T, axis=1) / row_scale
    log_det = 2.0 * np.log(np.diag(chol)).sum() + q * np.log(row_scale)
    if nu is None:
        return float(np.sum(-0.5 * (q * np.log(2 * np.pi) + log_det + d)))
    return float(np.sum(gammaln(0.5 * (nu + q)) - gammaln(0.5 * nu) - 0.5 * q * np.log(nu * np.pi)
                        - 0.5 * log_det - 0.5 * (nu + q) * np.log1p(d / nu)))


def run_chain(Y, coords, X, groups, K, n_warmup, n_samples, thin, seed, config=None,
              censored=None, censor_lower=None, report_every=0, label=""):
    config = Config(**(config or {}))
    Y, coords, X = (np.array(a, dtype=float) for a in (Y, coords, X))
    groups = np.asarray(groups).astype(int)
    if censored is None:
        censored = np.zeros(Y.shape, dtype=bool)
    rows, cols = np.nonzero(censored)
    lower = np.asarray(censor_lower, dtype=float)[rows, cols] if rows.size else np.empty(0)
    if rows.size:
        Y[rows, cols] = np.maximum(Y[rows, cols], lower)
    with threadpool_limits(limits=1):
        return _run_chain(Y, coords, X, groups, K, n_warmup, n_samples, thin, seed, config, rows, cols, lower,
                          report_every, label)


def _run_chain(Y, coords, X, groups, K, n_warmup, n_samples, thin, seed, config, rows, cols, lower, report_every, label):
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    n, q = Y.shape
    p, G = X.shape[1], groups.max() + 1
    counts = np.bincount(groups, minlength=G)
    nu0 = q + config.iw_df_offset
    Psi0 = (nu0 - q - 1) * config.residual_prior_mean * np.eye(q)
    prior_precision = np.r_[np.full(p, 1 / config.beta_prior_variance), np.full(K, 1 / config.loading_prior_variance)]

    # start from least squares and the SVD of the residuals
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    left, singular, right = np.linalg.svd(Y - X @ beta, full_matrices=False)
    start_scale = rng.choice([-1.0, 1.0], K) * np.exp(rng.normal(0, 0.25, K))
    F = left[:, :K] * (np.sqrt(n) * start_scale)
    loading = singular[:K, None] * right[:K] / (np.sqrt(n) * start_scale[:, None])
    residual = Y - X @ beta - F @ loading
    Sigma = residual.T @ residual / n + 0.15 * np.eye(q)
    eta = np.zeros(G)
    log_rho = rng.normal(config.log_rho_mean, 0.35, K)
    weights = np.ones(n)
    log_nu = np.log(config.nu_initial_median) + rng.normal(0, 0.5) if config.robust else np.nan

    distance, identity = cdist(coords, coords), np.eye(n)
    chols, log_dets, R_inverse = [], np.empty(K), np.empty((K, n, n))
    for k in range(K):
        chol, log_dets[k] = correlation_state(distance, np.exp(log_rho[k]), config.jitter)
        chols.append(chol)
        R_inverse[k] = cho_solve((chol, True), identity, check_finite=False)
    ranges = Proposal(K, config.rho_proposal_sd, lower=0.01)
    scales = Proposal(G, config.group_proposal_sd)
    amplitudes = Proposal(K, config.amplitude_proposal_sd)
    tail = Proposal(1, config.nu_proposal_sd)
    scales.sd[0] = 0.0
    free_groups = np.arange(G) != 0

    shapes = {"beta": (p, q), "loading": (K, q), "Sigma": (q, q), "rho": (K,), "F": (n, K), "group_variance": (G,),
              "weights": (n,), "nu": (), "censored_values": (len(rows),), "log_likelihood": ()}
    draws = {key: np.empty((n_samples,) + shape) for key, shape in shapes.items()}
    H = np.empty((n, p + K))
    H[:, :p] = X
    saved = 0
    total = n_warmup + n_samples * thin
    for iteration in range(1, total + 1):
        warmup = iteration <= n_warmup
        if rows.size:
            impute_censored(Y, X @ beta + F @ loading, Sigma, np.exp(eta[groups]) / weights, rows, cols, lower, rng)
        quadratics = site_quadratics(Y - X @ beta - F @ loading, Sigma)
        if config.robust:
            # nu with the weights integrated out, then the weights
            mahalanobis = quadratics / np.exp(eta[groups])
            log_nu, accepted = update_nu(log_nu, mahalanobis, q, tail.sd[0], config, rng)
            tail.record(np.array([accepted]), warmup)
            weights = rng.gamma(0.5 * (np.exp(log_nu) + q), 2.0 / (np.exp(log_nu) + mahalanobis))
        scales.record(update_groups(quadratics * weights, groups, eta, counts, q, scales.sd, config, rng), warmup)

        row_variance = np.exp(eta[groups]) / weights
        row_precision = row_precision_without_beta(X, row_variance, config.beta_prior_variance)
        ranges.record(update_ranges(Y, loading, Sigma, distance, log_rho, chols, log_dets, R_inverse, ranges.sd,
                                    config, rng, row_precision), warmup)
        F = sample_factors(Y, loading, Sigma, R_inverse, row_precision, rng)
        H[:, p:] = F
        root_precision = 1.0 / np.sqrt(row_variance)
        gamma, Sigma = sample_mniw(Y * root_precision[:, None], H * root_precision[:, None], prior_precision, nu0, Psi0, rng)
        beta, loading = gamma[:p], gamma[p:]
        amplitudes.record(update_amplitudes(F, loading, R_inverse, Sigma, amplitudes.sd, config, rng), warmup)

        if warmup and iteration % config.adapt_interval == 0:
            ranges.adapt(iteration, config)
            scales.adapt(iteration, config, active=free_groups)
            amplitudes.adapt(iteration, config)
            if config.robust:
                tail.adapt(iteration, config)
        if not warmup and (iteration - n_warmup) % thin == 0:
            nu = np.exp(log_nu) if config.robust else None
            for key, value in [("beta", beta), ("loading", loading), ("Sigma", Sigma), ("rho", np.exp(log_rho)),
                               ("F", F), ("group_variance", np.exp(eta)), ("weights", weights),
                               ("nu", np.nan if nu is None else nu), ("censored_values", Y[rows, cols])]:
                draws[key][saved] = value
            draws["log_likelihood"][saved] = log_likelihood(Y - X @ beta - F @ loading, Sigma, np.exp(eta[groups]), nu)
            saved += 1
        if report_every and iteration % report_every == 0:
            print(f"{label} {iteration}/{total} iterations", flush=True)

    elapsed = time.perf_counter() - start
    sampling = n_samples * thin
    summary = dict(seed=seed, K=K, n_warmup=n_warmup, n_samples=n_samples, thin=thin, elapsed_seconds=elapsed,
                   censored_entries=len(rows), config=asdict(config),
                   acceptance=dict(range=ranges.sample / sampling, group=scales.sample / sampling,
                                   amplitude=amplitudes.sample / sampling, nu=tail.sample / sampling),
                   proposal_sd=dict(range=ranges.sd, group=scales.sd, amplitude=amplitudes.sd, nu=tail.sd))
    return draws, summary
