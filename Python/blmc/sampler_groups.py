# Sampler with local scales by land use and element group: e_i | w_i ~ N(0, D_g Sigma D_g / w_i).
import time
from dataclasses import asdict

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.spatial.distance import cdist
from scipy.special import gammaln
from scipy.stats import invwishart, truncnorm
from threadpoolctl import threadpool_limits

from blmc.sampler import Config, Proposal, collapsed_term, correlation_state, update_amplitudes, update_nu

KEYS = ["beta", "loading", "Sigma", "rho", "F", "group_variance", "weights", "nu", "censored_values", "log_likelihood"]


def site_precisions(eta, element_group, Sigma_inv):
    """C_g = D_g^-1 Sigma^-1 D_g^-1 for every land use, and 1/d_g."""
    inv_d = np.exp(-0.5 * eta[:, element_group])
    return inv_d[:, :, None] * Sigma_inv[None] * inv_d[:, None, :], inv_d


def latent_system(Y, X, loading, C, groups, weights, R_inverse, Sigma_inv, beta_prior_variance):
    """Precision and rhs of z = (vec F, vec beta), both column-major, given loading, Sigma and the scales."""
    n, q = Y.shape
    p, K = X.shape[1], loading.shape[0]
    nK = n * K
    Q = C[groups] * weights[:, None, None]                      # precision of each site's error
    LQ = np.einsum("kj,njl->nkl", loading, Q)
    P = np.zeros((nK + p * q, nK + p * q))
    sites = np.arange(n)
    LQL = np.einsum("nkj,lj->nkl", LQ, loading)
    for k in range(K):
        for l in range(K):
            P[k * n + sites, l * n + sites] = LQL[:, k, l]
        P[k * n:(k + 1) * n, k * n:(k + 1) * n] += R_inverse[k]
    cross = np.einsum("nkj,nm->knjm", LQ, X).reshape(nK, q * p)
    P[:nK, nK:], P[nK:, :nK] = cross, cross.T
    P[nK:, nK:] = np.kron(Sigma_inv, np.eye(p) / beta_prior_variance)
    for g in range(len(C)):
        rows = groups == g
        P[nK:, nK:] += np.kron(C[g], X[rows].T @ (X[rows] * weights[rows, None]))
    Qy = np.einsum("njl,nl->nj", Q, Y)
    rhs = np.r_[(Qy @ loading.T).ravel(order="F"), (X.T @ Qy).ravel(order="F")]
    return (P + P.T) * 0.5, rhs


def update_ranges(P, rhs, distance, log_rho, chols, log_dets, R_inverse, sd, config, rng):
    """Metropolis on log phi_k with F and beta integrated out; returns the precision of the final state."""
    K, n, _ = R_inverse.shape
    identity = np.eye(n)
    accepted = np.zeros(K, dtype=bool)
    current_ll = collapsed_term(P, rhs, np.sum(log_dets))
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
            candidate = P.copy()
            block = slice(k * n, (k + 1) * n)
            candidate[block, block] += inverse - R_inverse[k]
            candidate = (candidate + candidate.T) * 0.5
            candidate_ll = collapsed_term(candidate, rhs, np.sum(log_dets) - log_dets[k] + log_det)
            target = candidate_ll - 0.5 * ((proposed - config.log_rho_mean) / config.log_rho_sd) ** 2
        except np.linalg.LinAlgError:
            continue
        if np.log(rng.uniform()) < target - current:
            accepted[k] = True
            log_rho[k] = proposed
            chols[k], log_dets[k], R_inverse[k] = chol, log_det, inverse
            P, current_ll = candidate, candidate_ll
    return accepted, P


def sample_gaussian(P, rhs, rng):
    chol = cholesky(P, lower=True, check_finite=False)
    mean = cho_solve((chol, True), rhs, check_finite=False)
    return mean + solve_triangular(chol.T, rng.normal(size=rhs.size), lower=False, check_finite=False)


def sample_coefficients(Y, H, C, groups, weights, Sigma_inv, prior_precision, rng):
    """(beta, Lambda) given F and Sigma; prior MN(0, V0, Sigma)."""
    q, d = Y.shape[1], H.shape[1]
    P = np.kron(Sigma_inv, np.diag(prior_precision))
    rhs = np.zeros((d, q))
    for g in range(len(C)):
        rows = groups == g
        weighted = H[rows] * weights[rows, None]
        P += np.kron(C[g], H[rows].T @ weighted)
        rhs += weighted.T @ Y[rows] @ C[g]
    return sample_gaussian((P + P.T) * 0.5, rhs.ravel(order="F"), rng).reshape((d, q), order="F")


def sample_sigma(scaled_residual, weights, Gamma, prior_precision, nu0, Psi0, rng):
    """Sigma given (beta, Lambda): inverse-Wishart with n + p + K extra degrees of freedom."""
    scale = Psi0 + (scaled_residual * weights[:, None]).T @ scaled_residual + Gamma.T @ (prior_precision[:, None] * Gamma)
    df = nu0 + len(weights) + Gamma.shape[0]
    Sigma = np.atleast_2d(invwishart.rvs(df=df, scale=(scale + scale.T) * 0.5, random_state=rng))
    return (Sigma + Sigma.T) * 0.5


def update_scales(residual, groups, weights, eta, element_group, Sigma_inv, sd, config, rng):
    """Metropolis on log a_{g,c} for every land use g > 0 and element group c; Agric stays at zero."""
    sizes = np.bincount(element_group, minlength=eta.shape[1])
    accepted = np.zeros(eta.shape, dtype=bool)
    for g in range(1, eta.shape[0]):
        rows = groups == g
        E, w = residual[rows], weights[rows]

        def target(values):
            r = E * np.exp(-0.5 * values[element_group])
            quad = np.sum(w * np.sum((r @ Sigma_inv) * r, axis=1))
            return -0.5 * (rows.sum() * np.sum(sizes * values) + quad + np.sum((values / config.group_log_variance_prior_sd) ** 2))

        current = target(eta[g])
        for c in range(eta.shape[1]):
            candidate = eta[g].copy()
            candidate[c] += rng.normal(scale=sd[g, c])
            proposed = target(candidate)
            if np.isfinite(proposed) and np.log(rng.uniform()) < proposed - current:
                eta[g], current, accepted[g, c] = candidate, proposed, True
    return accepted


def impute_censored(Y, mean, Sigma, d, weights, rows, cols, lower, rng):
    """Redraw each censored entry (true value >= lower) given the rest of its site, in place."""
    q = Y.shape[1]
    for r, c, bound in zip(rows, cols, lower):
        others = np.arange(q) != c
        coefficient = np.linalg.solve(Sigma[np.ix_(others, others)], Sigma[c, others])
        m = mean[r, c] + d[r, c] * coefficient @ ((Y[r, others] - mean[r, others]) / d[r, others])
        s = d[r, c] * np.sqrt((Sigma[c, c] - coefficient @ Sigma[c, others]) / weights[r])
        Y[r, c] = m + s * truncnorm.rvs((bound - m) / s, np.inf, random_state=rng)


def log_likelihood(residual, Sigma, log_scale, nu=None):
    """log density of the residuals, with log_scale[i, j] = log a for site i and element j."""
    q = residual.shape[1]
    chol = cholesky(Sigma, lower=True, check_finite=False)
    r = residual * np.exp(-0.5 * log_scale)
    d = np.sum(r * cho_solve((chol, True), r.T, check_finite=False).T, axis=1)
    log_det = 2.0 * np.log(np.diag(chol)).sum() + log_scale.sum(axis=1)
    if nu is None:
        return float(np.sum(-0.5 * (q * np.log(2 * np.pi) + log_det + d)))
    return float(np.sum(gammaln(0.5 * (nu + q)) - gammaln(0.5 * nu) - 0.5 * q * np.log(nu * np.pi)
                        - 0.5 * log_det - 0.5 * (nu + q) * np.log1p(d / nu)))


def run_chain(Y, coords, X, groups, element_group, K, n_warmup, n_samples, thin, seed, config=None,
              censored=None, censor_lower=None, report_every=0, label=""):
    config = Config(**(config or {}))
    Y, coords, X = (np.array(a, dtype=float) for a in (Y, coords, X))
    groups, element_group = np.asarray(groups).astype(int), np.asarray(element_group).astype(int)
    if censored is None:
        censored = np.zeros(Y.shape, dtype=bool)
    rows, cols = np.nonzero(censored)
    lower = np.asarray(censor_lower, dtype=float)[rows, cols] if rows.size else np.empty(0)
    if rows.size:
        Y[rows, cols] = np.maximum(Y[rows, cols], lower)
    with threadpool_limits(limits=1):
        return _run_chain(Y, coords, X, groups, element_group, K, n_warmup, n_samples, thin, seed, config,
                          rows, cols, lower, report_every, label)


def _run_chain(Y, coords, X, groups, element_group, K, n_warmup, n_samples, thin, seed, config, rows, cols, lower,
               report_every, label):
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    n, q = Y.shape
    p, G, n_classes = X.shape[1], groups.max() + 1, element_group.max() + 1
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
    eta = np.zeros((G, n_classes))
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
    scales = Proposal(G * n_classes, config.group_proposal_sd)
    amplitudes = Proposal(K, config.amplitude_proposal_sd)
    tail = Proposal(1, config.nu_proposal_sd)
    free_scales = np.repeat(np.arange(G) != 0, n_classes)
    scales.sd[~free_scales] = 0.0

    shapes = {"beta": (p, q), "loading": (K, q), "Sigma": (q, q), "rho": (K,), "F": (n, K),
              "group_variance": (G, n_classes), "weights": (n,), "nu": (), "censored_values": (len(rows),),
              "log_likelihood": ()}
    draws = {key: np.empty((n_samples,) + shape) for key, shape in shapes.items()}
    H = np.empty((n, p + K))
    H[:, :p] = X
    nK = n * K
    saved = 0
    total = n_warmup + n_samples * thin
    for iteration in range(1, total + 1):
        warmup = iteration <= n_warmup
        Sigma_inv = cho_solve((cholesky(Sigma, lower=True, check_finite=False), True), np.eye(q), check_finite=False)
        mean = X @ beta + F @ loading
        if rows.size:
            impute_censored(Y, mean, Sigma, np.exp(0.5 * eta[groups][:, element_group]), weights, rows, cols, lower, rng)
        residual = Y - mean
        if config.robust:
            # nu with the weights integrated out, then the weights
            r = residual * np.exp(-0.5 * eta[groups][:, element_group])
            mahalanobis = np.sum((r @ Sigma_inv) * r, axis=1)
            log_nu, accepted = update_nu(log_nu, mahalanobis, q, tail.sd[0], config, rng)
            tail.record(np.array([accepted]), warmup)
            weights = rng.gamma(0.5 * (np.exp(log_nu) + q), 2.0 / (np.exp(log_nu) + mahalanobis))
        scales.record(update_scales(residual, groups, weights, eta, element_group, Sigma_inv,
                                    scales.sd.reshape(G, n_classes), config, rng).ravel(), warmup)

        C, inv_d = site_precisions(eta, element_group, Sigma_inv)
        P, rhs = latent_system(Y, X, loading, C, groups, weights, R_inverse, Sigma_inv, config.beta_prior_variance)
        accepted, P = update_ranges(P, rhs, distance, log_rho, chols, log_dets, R_inverse, ranges.sd, config, rng)
        ranges.record(accepted, warmup)
        F = sample_gaussian(P, rhs, rng)[:nK].reshape((n, K), order="F")
        H[:, p:] = F
        Gamma = sample_coefficients(Y, H, C, groups, weights, Sigma_inv, prior_precision, rng)
        beta, loading = Gamma[:p], Gamma[p:]
        Sigma = sample_sigma((Y - H @ Gamma) * inv_d[groups], weights, Gamma, prior_precision, nu0, Psi0, rng)
        amplitudes.record(update_amplitudes(F, loading, R_inverse, Sigma, amplitudes.sd, config, rng), warmup)

        if warmup and iteration % config.adapt_interval == 0:
            ranges.adapt(iteration, config)
            scales.adapt(iteration, config, active=free_scales)
            amplitudes.adapt(iteration, config)
            if config.robust:
                tail.adapt(iteration, config)
        if not warmup and (iteration - n_warmup) % thin == 0:
            nu = np.exp(log_nu) if config.robust else None
            for key, value in [("beta", beta), ("loading", loading), ("Sigma", Sigma), ("rho", np.exp(log_rho)),
                               ("F", F), ("group_variance", np.exp(eta)), ("weights", weights),
                               ("nu", np.nan if nu is None else nu), ("censored_values", Y[rows, cols])]:
                draws[key][saved] = value
            draws["log_likelihood"][saved] = log_likelihood(Y - X @ beta - F @ loading, Sigma,
                                                            eta[groups][:, element_group], nu)
            saved += 1
        if report_every and iteration % report_every == 0:
            print(f"{label} {iteration}/{total} iterations", flush=True)

    elapsed = time.perf_counter() - start
    sampling = n_samples * thin
    summary = dict(seed=seed, K=K, n_warmup=n_warmup, n_samples=n_samples, thin=thin, elapsed_seconds=elapsed,
                   censored_entries=len(rows), element_group=element_group, config=asdict(config),
                   acceptance=dict(range=ranges.sample / sampling, group=scales.sample / sampling,
                                   amplitude=amplitudes.sample / sampling, nu=tail.sample / sampling),
                   proposal_sd=dict(range=ranges.sd, group=scales.sd, amplitude=amplitudes.sd, nu=tail.sd))
    return draws, summary
