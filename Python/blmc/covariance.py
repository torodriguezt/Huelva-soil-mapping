import numpy as np


def matern32(distance, phi):
    if not np.isfinite(phi) or phi <= 0:
        raise ValueError("phi must be finite and positive")
    scaled = np.sqrt(3.0) * np.asarray(distance, dtype=float) / phi
    return (1.0 + scaled) * np.exp(-scaled)


def to_correlation(covariance):
    sd = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    return covariance / (sd[..., :, None] * sd[..., None, :])
