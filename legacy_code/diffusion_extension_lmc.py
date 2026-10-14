#!/usr/bin/env python3
"""Diffusion-extension simulation for the DP score-estimation paper.

Pipeline (1D, CPU friendly):

    data -> projected LMC score draw -> DP histogram -> local-affine density
         -> exact OU-smoothed score of that density -> reverse OU sampler.

The parameter draw uses projected Langevin Monte Carlo, matching the practical
mechanism in the paper.  Projection onto the 1D coefficient polytope is done by
Dykstra's algorithm, which converges to the Euclidean projection onto

    |theta_j| <= B,  |theta_{j+1}-theta_j| <= D.

The forward OU score of the cellwise-affine surrogate is evaluated from exact
truncated-Gaussian moments; no score network is trained.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import ndtr
from scipy.stats import wasserstein_distance

# Import the core score/density implementation from the simulation suite.
HERE = Path(__file__).resolve().parent
SUITE = HERE / "suite"
if not SUITE.exists():
    # Compatibility fallback when this module is copied next to the suite.
    SUITE = HERE.parent / "dp_score_simulation_suite"
if str(SUITE) not in sys.path:
    sys.path.insert(0, str(SUITE))

from dp_score_sim.core import (  # noqa: E402
    SimConfig, sample_target, build_model, score_from_theta,
    true_score, theta_risk, gibbs_draws, conditional_mechanism_risk,
)
from dp_score_sim.density import (  # noqa: E402
    target_density, make_uniform_edges, histogram_masses,
    evaluate_local_affine, grid_tv, _safe_slopes,
)


# ---------------------------------------------------------------------------
# Exact Euclidean projection (to numerical tolerance) by Dykstra cycles
# ---------------------------------------------------------------------------

def _project_pair_strip(z0: float, z1: float, D: float) -> tuple[float, float]:
    """Euclidean projection of (z0,z1) onto |z1-z0| <= D."""
    diff = z1 - z0
    if diff > D:
        e = 0.5 * (diff - D)
        return z0 + e, z1 - e
    if diff < -D:
        e = 0.5 * (-D - diff)
        return z0 - e, z1 + e
    return z0, z1


def dykstra_project(
    y: np.ndarray,
    B: float,
    D: float,
    max_cycles: int = 80,
    tol: float = 1e-10,
) -> tuple[np.ndarray, int, float]:
    """Project y onto box + adjacent-difference constraints.

    Cyclic Dykstra projection onto one box and m-1 pairwise strips.  For this
    1D chain polytope the elementary set projections are analytic.
    """
    x = np.asarray(y, dtype=float).copy()
    m = x.size
    p_box = np.zeros(m)
    p_pair = np.zeros((max(m - 1, 0), 2))

    for cyc in range(1, max_cycles + 1):
        x_old = x.copy()

        # Box set.
        z = x + p_box
        xn = np.clip(z, -B, B)
        p_box = z - xn
        x = xn

        # Pairwise strips.
        for j in range(m - 1):
            z0 = x[j] + p_pair[j, 0]
            z1 = x[j + 1] + p_pair[j, 1]
            q0, q1 = _project_pair_strip(z0, z1, D)
            p_pair[j, 0] = z0 - q0
            p_pair[j, 1] = z1 - q1
            x[j], x[j + 1] = q0, q1

        change = float(np.max(np.abs(x - x_old)))
        box_v = max(0.0, float(np.max(np.abs(x)) - B))
        diff_v = 0.0 if m <= 1 else max(0.0, float(np.max(np.abs(np.diff(x))) - D))
        violation = max(box_v, diff_v)
        if change <= tol and violation <= 10 * tol:
            return x, cyc, violation

    box_v = max(0.0, float(np.max(np.abs(x)) - B))
    diff_v = 0.0 if m <= 1 else max(0.0, float(np.max(np.abs(np.diff(x))) - D))
    return x, max_cycles, max(box_v, diff_v)


@dataclass
class LMCInfo:
    step_size: float
    lipschitz: float
    ridge: float
    n_steps: int
    mean_projection_cycles: float
    max_projection_violation: float
    risk_trace_steps: list[int]
    risk_trace: list[float]


def projected_lmc_draw(
    model,
    rng: np.random.Generator,
    n_steps: int,
    ridge: float,
    step_factor: float,
    x_test: np.ndarray | None = None,
    cfg: SimConfig | None = None,
    trace_every: int = 0,
    projection_cycles: int = 80,
    projection_tol: float = 1e-10,
) -> tuple[np.ndarray, LMCInfo]:
    """One projected-LMC draw from the ridge-regularized target.

    eta = step_factor / L, where L is the smooth-potential gradient Lipschitz
    constant.  The starting point theta=0 is data independent.
    """
    lam_max = float(np.linalg.eigvalsh(model.A).max())
    Lgrad = model.c * lam_max + ridge
    eta = step_factor / max(Lgrad, 1e-14)

    theta = np.zeros(model.h.size)
    proj_cycles_used = []
    max_v = 0.0
    trace_steps, trace_risks = [], []

    for k in range(1, n_steps + 1):
        grad = model.c * (model.A @ theta + model.h) + ridge * theta
        proposal = theta - eta * grad + math.sqrt(2.0 * eta) * rng.normal(size=theta.size)
        theta, cyc, v = dykstra_project(
            proposal,
            model.coef_bound,
            model.diff_bound,
            max_cycles=projection_cycles,
            tol=projection_tol,
        )
        proj_cycles_used.append(cyc)
        max_v = max(max_v, v)

        if trace_every and (k == 1 or k % trace_every == 0 or k == n_steps):
            trace_steps.append(k)
            if x_test is not None and cfg is not None:
                r = theta_risk(
                    model, theta, x_test, cfg.gamma, cfg.target,
                    cfg.amplitude, cfg.K, cfg.lam, cfg.mixture_mu,
                )
                trace_risks.append(float(r))

    info = LMCInfo(
        step_size=eta,
        lipschitz=Lgrad,
        ridge=ridge,
        n_steps=n_steps,
        mean_projection_cycles=float(np.mean(proj_cycles_used)),
        max_projection_violation=float(max_v),
        risk_trace_steps=trace_steps,
        risk_trace=trace_risks,
    )
    return theta, info


# ---------------------------------------------------------------------------
# Cellwise-affine surrogate and exact OU moments
# ---------------------------------------------------------------------------

@dataclass
class AffineCells:
    left: np.ndarray
    right: np.ndarray
    centers: np.ndarray
    widths: np.ndarray
    masses: np.ndarray
    slopes: np.ndarray
    intercept: np.ndarray  # density = intercept + slopecoef * y
    slopecoef: np.ndarray


def affine_cells_from_hist(hist, slopes: np.ndarray) -> AffineCells:
    safe = _safe_slopes(np.asarray(slopes, dtype=float), hist.widths)
    base = hist.masses / hist.widths
    # base * [1 + safe*(y-c)] = base*(1-safe*c) + base*safe*y
    intercept = base * (1.0 - safe * hist.centers)
    slopecoef = base * safe
    return AffineCells(
        left=hist.edges[:-1].copy(),
        right=hist.edges[1:].copy(),
        centers=hist.centers.copy(),
        widths=hist.widths.copy(),
        masses=hist.masses.copy(),
        slopes=safe,
        intercept=intercept,
        slopecoef=slopecoef,
    )


def _phi(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def ou_affine_density_score(
    x: np.ndarray,
    cells: AffineCells,
    t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact P_t rho and score for a 1D cellwise-affine density.

    Forward OU: X_t = a Y + sqrt(v) Z, a=e^{-t}, v=1-e^{-2t}.
    Each cell integral is reduced to truncated Normal moments I0,I1,I2.
    The score uses
        grad log p_t(x) = -(x-a E[Y|X_t=x])/v.
    """
    if t <= 0:
        raise ValueError("t must be positive")
    x = np.atleast_1d(np.asarray(x, dtype=float))
    a = math.exp(-t)
    v = max(1.0 - math.exp(-2.0 * t), 1e-14)
    sigma = math.sqrt(v) / a
    mu = x[:, None] / a

    zl = (cells.left[None, :] - mu) / sigma
    zu = (cells.right[None, :] - mu) / sigma
    cdf = ndtr(zu) - ndtr(zl)
    pl = _phi(zl)
    pu = _phi(zu)

    inva = 1.0 / a
    I0 = inva * cdf
    I1 = inva * (mu * cdf + sigma * (pl - pu))
    I2 = inva * (
        (mu * mu + sigma * sigma) * cdf
        + 2.0 * mu * sigma * (pl - pu)
        + sigma * sigma * (zl * pl - zu * pu)
    )

    A = cells.intercept[None, :]
    B = cells.slopecoef[None, :]
    p = np.sum(A * I0 + B * I1, axis=1)
    ymoment = np.sum(A * I1 + B * I2, axis=1)
    p_safe = np.maximum(p, 1e-300)
    post_mean = ymoment / p_safe
    score = -(x - a * post_mean) / v
    return p_safe, score


def sample_affine_density(cells: AffineCells, n: int, rng: np.random.Generator) -> np.ndarray:
    """Exact rejection sampling cell-by-cell from the piecewise affine density."""
    idx = rng.choice(cells.masses.size, size=n, p=cells.masses)
    out = np.empty(n)
    for j in np.unique(idx):
        positions = np.flatnonzero(idx == j)
        need = positions.size
        l, r, c, s = cells.left[j], cells.right[j], cells.centers[j], cells.slopes[j]
        M = 1.0 + abs(s) * 0.5 * (r - l)
        vals = []
        got = 0
        while got < need:
            m = max(64, int(1.7 * (need - got)))
            y = rng.uniform(l, r, size=m)
            f = 1.0 + s * (y - c)
            keep = rng.uniform(size=m) < (f / M)
            acc = y[keep]
            take = min(need - got, acc.size)
            if take:
                vals.append(acc[:take]); got += take
        out[positions] = np.concatenate(vals)
    return out


def sample_ptau(cells: AffineCells, tau: float, n: int, rng: np.random.Generator) -> np.ndarray:
    y0 = sample_affine_density(cells, n, rng)
    a = math.exp(-tau)
    sd = math.sqrt(max(1.0 - math.exp(-2.0 * tau), 0.0))
    return a * y0 + sd * rng.normal(size=n)


# ---------------------------------------------------------------------------
# Adaptive reverse OU grid and sampler
# ---------------------------------------------------------------------------

def adaptive_forward_times(T: float, tau: float, kappa: float) -> np.ndarray:
    """Descending forward-time grid t_0=T > ... > t_N=tau.

    Uses h_k <= kappa * min(1,t_k), the standard shape needed near the
    early-stopping time: constant step away from zero and geometric refinement
    when t<1.  This gives N = O(kappa^{-1}(T+log(1/tau))).
    """
    if not (0 < tau < T):
        raise ValueError("need 0 < tau < T")
    if not (0 < kappa < 1):
        raise ValueError("kappa must lie in (0,1)")
    times = [float(T)]
    t = float(T)
    while t > tau * (1 + 1e-14):
        h = kappa * min(1.0, t)
        h = min(h, t - tau)
        if h <= 0:
            break
        t -= h
        times.append(float(t))
        if len(times) > 2_000_000:
            raise RuntimeError("adaptive grid unexpectedly large")
    times[-1] = float(tau)
    return np.asarray(times)


def reverse_ou_adaptive_samples(
    cells: AffineCells,
    T: float,
    tau: float,
    kappa: float,
    n_particles: int,
    rng: np.random.Generator,
    score_grid: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Euler-Maruyama reverse OU on an adaptive forward-time grid."""
    times = adaptive_forward_times(T, tau, kappa)
    y = rng.normal(size=n_particles)
    for k in range(len(times) - 1):
        t = times[k]
        h = times[k] - times[k + 1]
        _, sg = ou_affine_density_score(score_grid, cells, t)
        sval = np.interp(y, score_grid, sg, left=sg[0], right=sg[-1])
        y = y + h * (y + 2.0 * sval) + math.sqrt(2.0 * h) * rng.normal(size=n_particles)
    return y, len(times) - 1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def binned_tv_samples_vs_density(
    samples: np.ndarray,
    density_fn,
    xmin: float,
    xmax: float,
    bins: int,
    quad_per_bin: int = 7,
) -> float:
    """Histogram TV proxy using per-bin quadrature for the target mass."""
    edges = np.linspace(xmin, xmax, bins + 1)
    counts, _ = np.histogram(samples, bins=edges)
    emp = counts / len(samples)

    target_mass = np.zeros(bins)
    for j in range(bins):
        xs = np.linspace(edges[j], edges[j + 1], quad_per_bin)
        vals = density_fn(xs)
        target_mass[j] = np.trapezoid(vals, xs)
    inside = float(target_mass.sum())
    tail = max(0.0, 1.0 - inside)
    return float(0.5 * np.sum(np.abs(emp - target_mass)) + 0.5 * tail)


def empirical_tv_between_samples(a: np.ndarray, b: np.ndarray, xmin: float, xmax: float, bins: int) -> float:
    edges = np.linspace(xmin, xmax, bins + 1)
    ca, _ = np.histogram(a, bins=edges)
    cb, _ = np.histogram(b, bins=edges)
    pa = ca / max(len(a), 1)
    pb = cb / max(len(b), 1)
    return float(0.5 * np.sum(np.abs(pa - pb)))


def delta_theory(n: int, alpha: float, gamma: float, d: int = 1) -> float:
    a = n ** (-(gamma + 1.0) / (2.0 * gamma + d + 2.0))
    b = (n * alpha) ** (-(gamma + 1.0) / (d + gamma + 1.0))
    return float(max(a, b))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(args):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = SimConfig(
        gamma=args.gamma, target=args.target, amplitude=args.amplitude, K=args.K,
        lam=args.lam, mixture_mu=args.mixture_mu,
    )
    rng = np.random.default_rng(args.seed)

    x = sample_target(
        args.n, cfg.gamma, rng, cfg.target, cfg.amplitude, cfg.K, cfg.lam, cfg.mixture_mu
    )
    x_test = sample_target(
        args.n_test, cfg.gamma, rng, cfg.target, cfg.amplitude, cfg.K, cfg.lam, cfg.mixture_mu
    )

    alpha_score = args.privacy_split * args.alpha
    alpha_hist = (1.0 - args.privacy_split) * args.alpha
    model = build_model(
        x, cfg.gamma, alpha_score, cfg.target, cfg.amplitude, cfg.K, cfg.lam,
        cfg.mixture_mu, cfg.bandwidth_factor, cfg.R_factor, cfg.coef_bound_factor,
    )

    theta, lmc_info = projected_lmc_draw(
        model, rng, n_steps=args.lmc_steps, ridge=args.ridge,
        step_factor=args.lmc_step_factor, x_test=x_test, cfg=cfg,
        trace_every=args.lmc_trace_every, projection_cycles=args.projection_cycles,
    )
    lmc_risk = theta_risk(
        model, theta, x_test, cfg.gamma, cfg.target, cfg.amplitude,
        cfg.K, cfg.lam, cfg.mixture_mu,
    )

    # Optional Gibbs diagnostic only; the actual estimator used below is LMC.
    gibbs_risk = np.nan
    if args.gibbs_diagnostic:
        grng = np.random.default_rng(args.seed + 991)
        gdraws = gibbs_draws(model, grng, 500, 200, 10, 20)
        gibbs_risk = conditional_mechanism_risk(
            model, gdraws, x_test, cfg.gamma, cfg.target,
            cfg.amplitude, cfg.K, cfg.lam, cfg.mixture_mu,
        )

    edges = make_uniform_edges(model.R, args.h_factor * model.b)
    hist = histogram_masses(x, edges, alpha_hist, rng, replacement_adjacency=True)
    slopes = score_from_theta(model, theta, hist.centers)
    cells = affine_cells_from_hist(hist, slopes)

    xmax = max(args.xmax, model.R + 4.0, abs(cfg.mixture_mu) + 6.0 if cfg.target == "mixture" else args.xmax)
    x_grid = np.linspace(-xmax, xmax, args.metric_grid)
    truth = target_density(
        x_grid, cfg.gamma, cfg.target, cfg.amplitude, cfg.K, cfg.lam, cfg.mixture_mu
    )
    sur = evaluate_local_affine(hist, slopes, x_grid)
    tv_sur = grid_tv(truth, sur, x_grid)

    T = math.log(args.n) if args.T <= 0 else args.T
    tau = (1.0 / args.n) if args.tau <= 0 else args.tau
    p_tau, _ = ou_affine_density_score(x_grid, cells, tau)
    p_tau = p_tau / max(np.trapezoid(p_tau, x_grid), 1e-15)
    tv_ptau = grid_tv(truth, p_tau, x_grid)
    tv_early = grid_tv(sur, p_tau, x_grid)

    # Reference iid sample from P_tau surrogate gives the Monte-Carlo/binning floor.
    ref_ptau = sample_ptau(cells, tau, args.reference_particles, rng)
    ref_true = sample_target(
        args.reference_particles, cfg.gamma, rng, cfg.target,
        cfg.amplitude, cfg.K, cfg.lam, cfg.mixture_mu,
    )

    score_grid = np.linspace(-xmax, xmax, args.score_grid)

    if args.kappa:
        kappas = [float(k) for k in args.kappa]
    else:
        delta = delta_theory(args.n, args.alpha, cfg.gamma)
        kth = delta**2
        kappas = [min(0.25, m * kth) for m in args.kappa_multipliers]

    rows = []
    samples_for_plot = None
    for kk in kappas:
        for rep in range(args.reverse_reps):
            rrng = np.random.default_rng(args.seed + 10000 + rep + int(round(1e8 * kk)))
            y, N = reverse_ou_adaptive_samples(
                cells, T, tau, kk, args.particles, rrng, score_grid
            )

            true_fn = lambda z: target_density(
                np.asarray(z), cfg.gamma, cfg.target, cfg.amplitude,
                cfg.K, cfg.lam, cfg.mixture_mu,
            )
            ptau_fn = lambda z: ou_affine_density_score(np.asarray(z), cells, tau)[0]

            tv_true_proxy = binned_tv_samples_vs_density(
                y, true_fn, -xmax, xmax, args.tv_bins
            )
            tv_ptau_proxy = binned_tv_samples_vs_density(
                y, ptau_fn, -xmax, xmax, args.tv_bins
            )
            w_true = float(wasserstein_distance(y, ref_true))
            w_ptau = float(wasserstein_distance(y, ref_ptau))
            rows.append({
                "kappa": kk, "N": N, "rep": rep,
                "tv_proxy_to_true": tv_true_proxy,
                "tv_proxy_to_Ptau": tv_ptau_proxy,
                "W1_to_true": w_true, "W1_to_Ptau": w_ptau,
            })
            if kk == min(kappas) and rep == 0:
                samples_for_plot = y.copy()
            print(
                f"kappa={kk:.4g}, N={N:5d}, rep={rep}: "
                f"TVproxy(true)={tv_true_proxy:.4f}, "
                f"TVproxy(Ptau)={tv_ptau_proxy:.4f}, "
                f"W1(Ptau)={w_ptau:.4f}"
            )

    raw = pd.DataFrame(rows)
    raw.to_csv(out / "reverse_raw.csv", index=False)
    summ = raw.groupby(["kappa", "N"], as_index=False).agg(
        mean_tv_true=("tv_proxy_to_true", "mean"),
        sd_tv_true=("tv_proxy_to_true", "std"),
        mean_tv_ptau=("tv_proxy_to_Ptau", "mean"),
        sd_tv_ptau=("tv_proxy_to_Ptau", "std"),
        mean_W1_true=("W1_to_true", "mean"),
        mean_W1_ptau=("W1_to_Ptau", "mean"),
        n_rep=("rep", "count"),
    )
    summ["se_tv_true"] = summ.sd_tv_true / np.sqrt(summ.n_rep)
    summ["se_tv_ptau"] = summ.sd_tv_ptau / np.sqrt(summ.n_rep)
    summ.to_csv(out / "reverse_summary.csv", index=False)

    # Monte-Carlo TV floor: iid samples from the actual P_tau surrogate.
    ptau_fn = lambda z: ou_affine_density_score(np.asarray(z), cells, tau)[0]
    true_fn = lambda z: target_density(
        np.asarray(z), cfg.gamma, cfg.target, cfg.amplitude,
        cfg.K, cfg.lam, cfg.mixture_mu,
    )
    floor_ptau = binned_tv_samples_vs_density(ref_ptau[:args.particles], ptau_fn, -xmax, xmax, args.tv_bins)
    floor_true = binned_tv_samples_vs_density(ref_true[:args.particles], true_fn, -xmax, xmax, args.tv_bins)

    delta = delta_theory(args.n, args.alpha, cfg.gamma)
    kappa_theory = delta**2
    meta = pd.DataFrame([{
        "n": args.n, "alpha": args.alpha,
        "alpha_score": alpha_score, "alpha_hist": alpha_hist,
        "target": cfg.target, "gamma": cfg.gamma,
        "b": model.b, "R": model.R, "basis_dim": model.idx.size,
        "lmc_steps": args.lmc_steps, "ridge": args.ridge,
        "lmc_step_factor": args.lmc_step_factor,
        "lmc_step_size": lmc_info.step_size,
        "lmc_lipschitz": lmc_info.lipschitz,
        "lmc_risk": lmc_risk, "gibbs_reference_risk": gibbs_risk,
        "mean_projection_cycles": lmc_info.mean_projection_cycles,
        "max_projection_violation": lmc_info.max_projection_violation,
        "T": T, "tau": tau,
        "delta_polynomial": delta, "kappa_theory_polynomial": kappa_theory,
        "TV_surrogate_true": tv_sur,
        "TV_Ptau_true": tv_ptau,
        "TV_surrogate_Ptau": tv_early,
        "TV_proxy_MC_floor_Ptau": floor_ptau,
        "TV_proxy_MC_floor_true": floor_true,
    }])
    meta.to_csv(out / "meta.csv", index=False)

    if lmc_info.risk_trace_steps:
        pd.DataFrame({"step": lmc_info.risk_trace_steps, "score_risk": lmc_info.risk_trace}).to_csv(
            out / "lmc_trace.csv", index=False
        )
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        ax.plot(lmc_info.risk_trace_steps, lmc_info.risk_trace, marker="o")
        if np.isfinite(gibbs_risk):
            ax.axhline(gibbs_risk, linestyle="--", label="Gibbs target reference")
            ax.legend()
        ax.set_xscale("log")
        ax.set_xlabel("projected-LMC iterations")
        ax.set_ylabel("score risk")
        ax.set_title("Projected-LMC diagnostic")
        fig.tight_layout()
        fig.savefig(out / "lmc_trace.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Figure: reverse discretization.
    s = summ.sort_values("N")
    fig, ax = plt.subplots(figsize=(7.3, 5.2))
    ax.errorbar(s.N, s.mean_tv_ptau, yerr=s.se_tv_ptau, marker="o", capsize=3,
                label=r"reverse samples vs $P_\tau\tilde\rho_D$ (binned TV)")
    ax.errorbar(s.N, s.mean_tv_true, yerr=s.se_tv_true, marker="s", capsize=3,
                label=r"reverse samples vs $\rho_0$ (binned TV)")
    ax.axhline(floor_ptau, linestyle=":", label="Monte-Carlo/binning floor")
    ax.axhline(tv_ptau, linestyle="--", label=r"exact TV$(P_\tau\tilde\rho_D,\rho_0)$")
    ax.set_xscale("log")
    ax.set_xlabel("reverse steps N")
    ax.set_ylabel("TV / binned-TV proxy")
    ax.set_title("Diffusion post-processing: discretization approaches the statistical floor")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "diffusion_discretization.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "diffusion_discretization.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure: density/samples.
    if samples_for_plot is not None:
        fig, ax = plt.subplots(figsize=(7.8, 5.0))
        ax.plot(x_grid, truth, linewidth=2, label=r"true $\rho_0$")
        ax.plot(x_grid, sur, label=r"private $\tilde\rho_D$")
        ax.plot(x_grid, p_tau, linestyle="--", label=r"$P_\tau\tilde\rho_D$")
        ax.hist(samples_for_plot, bins=args.tv_bins, range=(-xmax, xmax), density=True,
                histtype="step", label=f"reverse samples (N={int(s.N.max())})")
        ax.set_xlim(-min(6.0, xmax), min(6.0, xmax))
        ax.set_xlabel("x"); ax.set_ylabel("density")
        ax.set_title("Private density -> exact OU scores -> reverse samples")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "diffusion_qualitative.png", dpi=200, bbox_inches="tight")
        fig.savefig(out / "diffusion_qualitative.pdf", bbox_inches="tight")
        plt.close(fig)

    print("\n--- summary ---")
    print(meta.to_string(index=False))
    print("\nReverse summary:")
    print(summ.sort_values("N").to_string(index=False))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="outputs/diffusion_lmc")
    p.add_argument("--target", choices=["multiscale", "cusp", "mixture"], default="multiscale")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--amplitude", type=float, default=0.25)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--lam", type=float, default=0.25)
    p.add_argument("--mixture-mu", type=float, default=2.0)
    p.add_argument("--n", type=int, default=3200)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--privacy-split", type=float, default=0.5)
    p.add_argument("--h-factor", type=float, default=1.0)

    p.add_argument("--lmc-steps", type=int, default=5000)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--lmc-step-factor", type=float, default=0.02)
    p.add_argument("--lmc-trace-every", type=int, default=250)
    p.add_argument("--projection-cycles", type=int, default=80)
    p.add_argument("--gibbs-diagnostic", action="store_true")

    p.add_argument("--T", type=float, default=0.0, help="<=0 uses log(n)")
    p.add_argument("--tau", type=float, default=0.0, help="<=0 uses 1/n")
    p.add_argument("--kappa", nargs="+", type=float, default=None,
                   help="explicit adaptive-grid kappa values")
    p.add_argument("--kappa-multipliers", nargs="+", type=float,
                   default=[8.0, 4.0, 2.0, 1.0],
                   help="multiples of polynomial-theory delta_n^2")
    p.add_argument("--reverse-reps", type=int, default=3)
    p.add_argument("--particles", type=int, default=4000)
    p.add_argument("--reference-particles", type=int, default=20000)
    p.add_argument("--score-grid", type=int, default=301)
    p.add_argument("--metric-grid", type=int, default=2001)
    p.add_argument("--tv-bins", type=int, default=80)
    p.add_argument("--xmax", type=float, default=7.0)
    p.add_argument("--n-test", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261012)
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
