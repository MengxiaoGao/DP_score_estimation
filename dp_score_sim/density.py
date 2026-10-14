"""Density utilities for 1D score-to-density bridge experiments."""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp

from .core import true_score


def log_unnormalized_density(
    x: np.ndarray | float,
    gamma: float,
    target: str = "multiscale",
    amplitude: float = 0.25,
    K: int = 10,
    lam: float = 0.25,
    mixture_mu: float = 2.0,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if target == "multiscale":
        out = -0.5 * x**2
        for k in range(K + 1):
            out -= amplitude * 2.0 ** (-k * (gamma + 1.0)) * np.cos((2.0**k) * x)
        return out
    if target == "cusp":
        return -0.5 * x**2 - lam * np.abs(x) ** (1.0 + gamma)
    if target == "mixture":
        # log of 0.5*N(-mu,1)+0.5*N(mu,1), up to the Gaussian normalizer.
        a = -0.5 * (x - mixture_mu) ** 2
        b = -0.5 * (x + mixture_mu) ** 2
        return np.log(0.5) + np.logaddexp(a, b)
    raise ValueError(f"unknown target: {target}")


@lru_cache(maxsize=64)
def _normalizer_cached(
    gamma: float,
    target: str,
    amplitude: float,
    K: int,
    lam: float,
    mixture_mu: float,
) -> float:
    if target == "mixture":
        # log_unnormalized_density omits 1/sqrt(2*pi).
        return math.sqrt(2.0 * math.pi)

    f = lambda z: math.exp(float(log_unnormalized_density(
        z, gamma, target, amplitude, K, lam, mixture_mu
    )))
    val, _ = quad(f, -np.inf, np.inf, epsabs=1e-10, epsrel=1e-9, limit=300)
    return float(val)


def target_density(
    x: np.ndarray,
    gamma: float,
    target: str = "multiscale",
    amplitude: float = 0.25,
    K: int = 10,
    lam: float = 0.25,
    mixture_mu: float = 2.0,
) -> np.ndarray:
    z = _normalizer_cached(gamma, target, amplitude, K, lam, mixture_mu)
    return np.exp(log_unnormalized_density(
        x, gamma, target, amplitude, K, lam, mixture_mu
    )) / z


@dataclass
class Histogram1D:
    edges: np.ndarray
    centers: np.ndarray
    widths: np.ndarray
    masses: np.ndarray


def make_uniform_edges(R: float, h_target: float) -> np.ndarray:
    m = max(4, int(math.ceil(2.0 * R / h_target)))
    return np.linspace(-R, R, m + 1)


def histogram_masses(
    x: np.ndarray,
    edges: np.ndarray,
    epsilon: float | None,
    rng: np.random.Generator,
    replacement_adjacency: bool = True,
) -> Histogram1D:
    """Histogram masses with optional Laplace release.

    For replacement adjacency, the count-vector L1 sensitivity is 2;
    for add/remove adjacency it is 1.  The released counts are clipped
    to nonnegative values and renormalized, both by post-processing.
    """
    x = np.asarray(x, dtype=float)
    eps = np.finfo(float).eps
    xc = np.clip(x, edges[0] + eps, edges[-1] - eps)
    counts, _ = np.histogram(xc, bins=edges)
    counts = counts.astype(float)

    if epsilon is not None and np.isfinite(epsilon):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        sensitivity = 2.0 if replacement_adjacency else 1.0
        counts = counts + rng.laplace(
            loc=0.0, scale=sensitivity / epsilon, size=counts.size
        )
        counts = np.maximum(counts, 0.0)

    if counts.sum() <= 0:
        masses = np.full(counts.size, 1.0 / counts.size)
    else:
        masses = counts / counts.sum()

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    return Histogram1D(edges=edges, centers=centers, widths=widths, masses=masses)


def _safe_slopes(slopes: np.ndarray, widths: np.ndarray, safety: float = 0.98) -> np.ndarray:
    """Clip slopes so 1+s(x-c) remains nonnegative throughout each cell."""
    max_abs = safety * 2.0 / widths
    return np.clip(slopes, -max_abs, max_abs)


def evaluate_piecewise_constant(hist: Histogram1D, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    j = np.searchsorted(hist.edges, x, side="right") - 1
    valid = (j >= 0) & (j < hist.masses.size)
    jv = j[valid]
    out[valid] = hist.masses[jv] / hist.widths[jv]
    return out


def evaluate_local_affine(
    hist: Histogram1D,
    slopes: np.ndarray,
    x: np.ndarray,
    enforce_nonnegative: bool = True,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    slopes = np.asarray(slopes, dtype=float)
    if slopes.shape != hist.centers.shape:
        raise ValueError("slopes must have one value per cell")
    if enforce_nonnegative:
        slopes = _safe_slopes(slopes, hist.widths)

    out = np.zeros_like(x)
    j = np.searchsorted(hist.edges, x, side="right") - 1
    valid = (j >= 0) & (j < hist.masses.size)
    jv = j[valid]
    base = hist.masses[jv] / hist.widths[jv]
    out[valid] = base * (1.0 + slopes[jv] * (x[valid] - hist.centers[jv]))
    # Numerical safety only; with safe slopes this should not be needed.
    out = np.maximum(out, 0.0)
    return out


def grid_tv(p: np.ndarray, q: np.ndarray, x: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(0.5 * np.trapezoid(np.abs(p - q), x))


def exact_score_at_centers(
    centers: np.ndarray,
    gamma: float,
    target: str,
    amplitude: float,
    K: int,
    lam: float,
    mixture_mu: float,
) -> np.ndarray:
    return true_score(
        centers,
        gamma,
        target=target,
        amplitude=amplitude,
        K=K,
        lam=lam,
        mixture_mu=mixture_mu,
    )
