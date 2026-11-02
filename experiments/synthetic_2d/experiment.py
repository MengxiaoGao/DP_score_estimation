#!/usr/bin/env python3
"""Standalone 2D score / diffusion / replace-one sensitivity experiments.

Dependencies: numpy scipy numba pandas. No local module imports.

Examples (all paths and seeds are recorded in the output manifest):
  python two_dimensional_simulation.py --task validate --output checks_2d
  python two_dimensional_simulation.py --profile smoke --task all --output smoke_2d
  python two_dimensional_simulation.py --profile pilot --task alpha --output pilot_2d
  python two_dimensional_simulation.py --profile paper --task all --workers 2 --output paper_2d
  python two_dimensional_simulation.py --task summarize --output paper_2d

The paper profile is a configuration, NOT a convergence certificate. Review
pilot diagnostics before a large run. Re-running the SAME command resumes
atomic row and numerical-stage checkpoints. Sleep suspends computation; wake
can resume it. An exited process requires re-running the command.

Target: X=B Z, independent multiscale Z coordinates, K=40, amplitude=.25.
B is the ORIGINAL public correlation-.5 transform normalized at gamma=.5;
it stays fixed in both panels. It is used only for target generation and
evaluation, never to demix observations before fitting a joint tensor spline.

Alpha labels are SCORE budgets. Diffusion total budget is twice that value.
Private score: final zero-start projected-LMC iterate. Nonprivate score:
constrained, unregularized score-matching ERM in the SAME spline class.
The histogram uses simplex-projected Laplace frequencies (old 2D convention).
Reverse diffusion uses exact product truncated-normal moments, no interpolation.
2D binned TV is TWO-SAMPLE TV with separate overflow bins, unlike the 1D
integrated-bin convention. OU TV has a separate inner Monte Carlo SE.

Sensitivity changes exactly one 2D observation. The basis, constraints, and
histogram grid are frozen at the reference score budget for every method.
Common-random-number effects depend on a coupling and are not DP guarantees.
Independent-neighbor and same-data-repeat effects are also saved. The ideal
exponential mechanism has a calibrated loss sensitivity; finite LMC and
numerical projection do NOT constitute a certified finite-run DP mechanism.

Origin: multidimensional_2d_joint_20261014/{joint.py,vendor/target1d.py};
replace-one design: diffusion_extension_github_20261014/stability.py.
This version fixes the gamma=1 Holder bound and out-of-domain tensor indexing,
adds score risk, a gamma panel, nonprivate ERM, and full diffusion sensitivity.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
from multiprocessing import get_context
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from numpy.polynomial.legendre import leggauss
from numba import njit
from scipy import sparse
from scipy.stats import t as student_t, wasserstein_distance


@dataclass(frozen=True)
class Config:
    schema: int = 1
    seed: int = 20261023
    profile: str = "pilot"
    n_values: tuple = (1000, 2000, 4000, 8000, 16000)
    score_alphas: tuple = (.5, 1., 2., 4., 8.)
    gammas: tuple = (.2, .4, .6, .8, 1.)
    alpha_panel_gamma: float = .5
    gamma_panel_score_alpha: float = 2.
    K: int = 40
    amplitude: float = .25
    normalization_gamma: float = .5
    repetitions: int = 5
    lmc_steps: int = 100000
    # The earlier 2D pilot used .01. A representative n=1000,
    # alpha_score=.5, gamma=.5 check at .1 reached the 1,000-cycle Dykstra cap.
    lmc_step_factor: float = .01
    lmc_chunk: int = 1000
    lmc_checkpoint_every: int = 5000
    projection_cycles: int = 1000
    projection_tol: float = 1e-9
    ridge_mode: str = "theoretical"
    legacy_ridge: float = .001
    privacy_beta: float = .01
    bandwidth_factor: float = .5
    bandwidth_cap: float = .3
    R_factor: float = 1.25
    affine_safety: float = .98
    particles: int = 2000
    reference_particles: int = 50000
    test_particles: int = 10000
    ou_metric_particles: int = 8192
    bins_per_axis: int = 16
    metric_bound: float = 6.
    quadrature_order: int = 16
    quadrature_tolerance: float = .002
    kappa_multiplier: float = 1.
    reverse_checkpoint_every: int = 100
    erm_max_steps: int = 20000
    erm_tol: float = 2e-6
    sensitivity_gamma: float = .5
    sensitivity_reference_score_alpha: float = 2.
    neighbor_mode: str = "replace"
    perturbation: tuple = (.25, -.25)
    sliced_directions: int = 32


def profile(name):
    c = Config(profile=name)
    if name == "paper":
        return replace(c, repetitions=100, lmc_steps=500000,
                       particles=20000, ou_metric_particles=32768)
    if name == "smoke":
        return replace(c, n_values=(200, 400), score_alphas=(.5, 2.),
                       gammas=(.4, 1.), repetitions=2, lmc_steps=100,
                       particles=128, reference_particles=1024,
                       test_particles=512, ou_metric_particles=512,
                       lmc_chunk=50, lmc_checkpoint_every=50,
                       reverse_checkpoint_every=100)
    return c


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def rng(c, *keys):
    return np.random.default_rng(np.random.SeedSequence([c.seed] + [int(k) for k in keys]))


def stream_seed(c, *keys):
    return int(rng(c, *keys).integers(1, 2**31 - 1))


def hash_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def log_latent(x, gamma, amplitude, K):
    out = -.5 * np.asarray(x)**2
    for k in range(K + 1):
        out -= amplitude * 2.**(-k * (1 + gamma)) * np.cos(2.**k * x)
    return out


def latent_score(x, gamma, amplitude, K):
    out = -np.asarray(x).copy()
    for k in range(K + 1):
        out += amplitude * 2.**(-k * gamma) * np.sin(2.**k * x)
    return out


@lru_cache(None)
def latent_constants(gamma, amplitude, K):
    vals = []
    for order in (16, 32):
        u, w = leggauss(order)
        e = np.linspace(-12, 12, 1537)
        x = ((e[:-1] + e[1:])[:, None]/2 + np.diff(e)[:, None]*u/2).ravel()
        ww = (np.diff(e)[:, None]*w/2).ravel()
        p = np.exp(log_latent(x, gamma, amplitude, K))
        Z = ww @ p
        vals.append((float(Z), float(ww @ (p*x*x)/Z)))
    err = max(abs(vals[0][0]-vals[1][0]), abs(vals[0][1]-vals[1][1]))
    if err > 1e-7:
        raise RuntimeError(f"Target normalizer needs refinement: delta={err}")
    return vals[1][0], vals[1][1], err


@lru_cache(None)
def target_constants(gamma, amplitude, K, normalization_gamma):
    Z, var, error = latent_constants(gamma, amplitude, K)
    _, refvar, _ = latent_constants(normalization_gamma, amplitude, K)
    off = 2 - np.sqrt(3.)
    A = np.array([[1., off], [off, 1.]])
    B = A * np.sqrt(2 / (refvar*np.sum(A*A)))
    H = np.linalg.inv(B)
    # Finite-series Holder bound is valid including gamma=1. For gamma<1,
    # the old dyadic bound is also valid; choose the smaller proven bound.
    osc = amplitude*(K+1)*2**(1-gamma)
    if gamma < 1:
        dyadic = amplitude*(2**(1-gamma)/(2**(1-gamma)-1) + 2/(1-2**(-gamma)))
        osc = min(osc, dyadic)
    op = np.linalg.norm(H, 2)
    L = op**2 + osc*2**((1-gamma)/2)*op**(1+gamma)
    return dict(Z=Z, latent_variance=var, B=B, H=H, holder_L=float(L),
                covariance=var*B@B.T, normalizer_delta=error)


def tc(c, gamma):
    return target_constants(gamma, c.amplitude, c.K, c.normalization_gamma)


def draw(count, random, c, gamma):
    need = 2*count
    chunks = []
    envelope = sum(2.**(-k*(1+gamma)) for k in range(c.K+1))
    while need:
        x = random.normal(size=max(1024, int(2.4*need)))
        potential = np.zeros_like(x)
        for k in range(c.K+1):
            potential += 2.**(-k*(1+gamma))*np.cos(2.**k*x)
        keep = x[random.random(len(x)) < np.exp(-c.amplitude*(potential+envelope))]
        take = min(len(keep), need)
        chunks.append(keep[:take])
        need -= take
    return np.concatenate(chunks).reshape(count, 2) @ tc(c, gamma)["B"].T


def truth(x, c, gamma):
    t = tc(c, gamma)
    z = np.asarray(x) @ t["H"].T
    return np.exp(log_latent(z[:, 0], gamma, c.amplitude, c.K) +
                  log_latent(z[:, 1], gamma, c.amplitude, c.K))*abs(np.linalg.det(t["H"]))/t["Z"]**2


def true_score(x, c, gamma):
    H = tc(c, gamma)["H"]
    return latent_score(np.asarray(x) @ H.T, gamma, c.amplitude, c.K) @ H


def basis(x, mod, derivative=None):
    u = np.asarray(x)/mod["b"]
    fl = np.floor(u).astype(np.int64)
    v = u-fl
    ids = fl[:, :, None]-np.arange(3)[None, None, :]-mod["jmin"]
    val = np.stack([.5*v*v, -v*v+v+.5, .5*(1-v)**2], axis=2)
    der = np.stack([v, 1-2*v, v-1], axis=2)/mod["b"]
    a = der[:, 0] if derivative == 0 else val[:, 0]
    b = der[:, 1] if derivative == 1 else val[:, 1]
    i = np.broadcast_to(ids[:, 0, :, None], (len(x), 3, 3)).ravel()
    j = np.broadcast_to(ids[:, 1, None, :], (len(x), 3, 3)).ravel()
    m = mod["m"]
    valid = (i >= 0) & (i < m) & (j >= 0) & (j < m)
    rows = np.repeat(np.arange(len(x)), 9)
    data = (a[:, :, None]*b[:, None, :]).ravel()
    return sparse.csr_matrix((data[valid], (rows[valid], i[valid]*m+j[valid])), shape=(len(x), m*m))


def make_class(n, score_alpha, gamma, c):
    R = max(2.5, c.R_factor*np.sqrt(np.log(n+1)))
    raw = c.bandwidth_factor*max(n**(-1/(4+2*gamma)),
                               (n*score_alpha)**(-1/(3+gamma)))*np.sqrt(np.log(n+1))
    b = min(raw, c.bandwidth_cap)
    jmin = math.floor(-2*R/b)-3
    m = math.ceil(2*R/b)-jmin+1
    target = tc(c, gamma)
    H = target["H"]
    osc = c.amplitude*sum(2.**(-k*gamma) for k in range(c.K+1))
    bound = 1.1*max(np.abs(H.T@H).sum(axis=1)*(2*R+3*b) + np.abs(H.T).sum(axis=1)*osc)
    D = target["holder_L"]*b**gamma
    vertices = np.arange(m*m).reshape(m, m)
    e0 = np.r_[vertices[:-1, :].ravel(), vertices[:, :-1].ravel()]
    e1 = np.r_[vertices[1:, :].ravel(), vertices[:, 1:].ravel()]
    return dict(n=n, gamma=gamma, R=float(R), b=float(b), b_raw=float(raw),
                jmin=jmin, m=m, B=float(bound), D=float(D), Q=float(bound**2+2*D/b),
                holder_L=target["holder_L"], e0=e0, e1=e1)


def fit_statistics(x, mod):
    z = x.copy()
    outside = np.any(abs(z) > mod["R"], axis=1)
    z[outside] = 0
    psi = basis(z, mod)
    gram = (psi.T@psi).tocsr()
    gram.eliminate_zeros()
    gram.sort_indices()
    h = np.column_stack([np.asarray(basis(z, mod, a).sum(axis=0)).ravel() for a in (0, 1)])
    return dict(mod, gram=gram, h=h, truncated_fraction=float(outside.mean()))


def calibrate(mod, score_alpha, c):
    temp = score_alpha/(4*mod["Q"])
    dim = 2*mod["m"]**2
    if c.ridge_mode == "legacy":
        ridge = c.legacy_ridge
    else:
        g = mod["gamma"]
        rn = max(mod["n"]**(-2*g/(4+2*g)), (mod["n"]*score_alpha)**(-2*g/(3+g)))
        upsilon = c.privacy_beta/(1+math.exp(score_alpha))
        ridge = min(rn, 2*math.log1p(upsilon/2)/mod["B"]**2)/dim
    return float(temp), float(ridge)


@njit(cache=True)
def project(y, B, D, e0, e1, max_cycles, tol):
    x = y.copy()
    box = np.zeros_like(x)
    p0 = np.zeros((len(e0), 2))
    p1 = np.zeros_like(p0)
    violation = 0.
    for it in range(max_cycles):
        old = x.copy()
        v = x+box
        xn = np.minimum(B, np.maximum(-B, v))
        box = v-xn
        x = xn
        for e in range(len(e0)):
            i, j = e0[e], e1[e]
            for k in range(2):
                a, b = x[i, k]+p0[e, k], x[j, k]+p1[e, k]
                diff = b-a
                shift = .5*(diff-min(D, max(-D, diff)))
                x[i, k], x[j, k] = a+shift, b-shift
                p0[e, k], p1[e, k] = -shift, shift
        violation = max(0., np.max(np.abs(x))-B)
        for e in range(len(e0)):
            for k in range(2):
                violation = max(violation, abs(x[e0[e], k]-x[e1[e], k])-D)
        if np.max(np.abs(x-old)) < tol and violation < 10*tol:
            return x, it+1, violation, True
    return x, max_cycles, violation, False


@njit(cache=True)
def lmc_chunk(theta, data, indices, indptr, h, temp, ridge, eta,
              B, D, e0, e1, steps, seed, maxcy, tol):
    np.random.seed(seed)
    maxv, cycs = 0., 0
    for _ in range(steps):
        grad = h.copy()
        for i in range(len(theta)):
            for z in range(indptr[i], indptr[i+1]):
                j, v = indices[z], data[z]
                grad[i, 0] += v*theta[j, 0]
                grad[i, 1] += v*theta[j, 1]
        proposal = theta-eta*(temp*grad+ridge*theta)+np.sqrt(2*eta)*np.random.randn(*theta.shape)
        theta, cy, vio, ok = project(proposal, B, D, e0, e1, maxcy, tol)
        maxv, cycs = max(maxv, vio), cycs+cy
        if not ok:
            return theta, maxv, cycs, False
    return theta, maxv, cycs, True


def lmc(mod, score_alpha, seed, c, checkpoint):
    temp, ridge = calibrate(mod, score_alpha, c)
    A = mod["gram"]
    lip = temp*float(np.asarray(abs(A).sum(axis=1)).max())+ridge
    eta = c.lmc_step_factor/lip
    random = np.random.default_rng(seed)
    theta = np.zeros_like(mod["h"])
    done, maxv, cycles, elapsed = 0, 0., 0, 0.
    trace = []
    probes = basis(np.array([[0., 0.], [1., 1.], [-1., .5]]), mod)
    signature = fingerprint(dict(seed=seed, steps=c.lmc_steps, eta=eta,
                                 ridge=ridge, gram=hash_array(A.data), h=hash_array(mod["h"])))
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        with np.load(checkpoint) as z:
            state = json.loads(str(z["state"]))
            if state["signature"] != signature:
                raise RuntimeError("LMC checkpoint mismatch")
            theta = z["theta"]
        done, maxv, cycles, elapsed = state["done"], state["maxv"], state["cycles"], state["elapsed"]
        random.bit_generator.state = state["rng"]
        trace = state["trace"]
    start = time.perf_counter()
    for step in range(done, c.lmc_steps, c.lmc_chunk):
        count = min(c.lmc_chunk, c.lmc_steps-step)
        theta, vio, cy, ok = lmc_chunk(theta, A.data, A.indices, A.indptr, mod["h"], temp, ridge, eta,
            mod["B"], mod["D"], mod["e0"], mod["e1"], count,
            int(random.integers(1, 2**31-1)), c.projection_cycles, c.projection_tol)
        if not ok:
            raise RuntimeError(f"Projection did not converge at LMC chunk {step}; violation={vio}")
        cycles += cy
        maxv = max(maxv, vio)
        objective = temp*(.5*np.sum(theta*(A@theta))+np.sum(mod["h"]*theta))+.5*ridge*np.sum(theta**2)
        trace.append([step+count, float(theta.mean()), float(np.mean(theta**2)),
                      float(objective)]+np.asarray(probes@theta).ravel().tolist())
        if (step+count) % c.lmc_checkpoint_every == 0 or step+count == c.lmc_steps:
            atomic_npz(checkpoint, theta=theta, state=json.dumps(dict(signature=signature, done=step+count,
                maxv=maxv, cycles=cycles, elapsed=elapsed+time.perf_counter()-start,
                rng=random.bit_generator.state, trace=trace)))
    return theta, dict(lmc_eta=eta, lmc_lipschitz=lip, ridge=ridge, lmc_steps=c.lmc_steps,
        lmc_max_violation=maxv, lmc_mean_projection_cycles=cycles/c.lmc_steps,
        lmc_seconds=elapsed+time.perf_counter()-start, lmc_seed=seed)


def constrained_erm(mod, c):
    A, h = mod["gram"]/mod["n"], mod["h"]/mod["n"]
    lip = float(np.asarray(abs(A).sum(axis=1)).max())
    theta = np.zeros_like(h)
    z, momentum = theta.copy(), 1.
    start = time.perf_counter()
    residual = np.inf
    for step in range(c.erm_max_steps):
        new, _, vio, ok = project(z-(A@z+h)/lip, mod["B"], mod["D"], mod["e0"], mod["e1"],
                                   c.projection_cycles, c.projection_tol)
        if not ok:
            raise RuntimeError("ERM projection did not converge")
        next_momentum = (1+np.sqrt(1+4*momentum**2))/2
        next_z = new+(momentum-1)/next_momentum*(new-theta)
        # Adaptive restart of accelerated projected gradient.
        if np.sum((z-new)*(new-theta)) > 0:
            next_momentum, next_z = 1., new.copy()
        theta, z, momentum = new, next_z, next_momentum
        if step % 25 == 0 or step+1 == c.erm_max_steps:
            projected, _, _, ok = project(theta-(A@theta+h)/lip, mod["B"], mod["D"],
                                           mod["e0"], mod["e1"], c.projection_cycles, c.projection_tol)
            residual = float(lip*np.max(abs(theta-projected)))
            if ok and residual <= c.erm_tol:
                return theta, dict(erm_steps=step+1, erm_gradient_mapping=residual,
                                   erm_seconds=time.perf_counter()-start)
    raise RuntimeError(f"ERM did not meet stationarity tolerance: {residual}")


def simplex(v):
    u = np.sort(v.ravel())[::-1]
    s = np.cumsum(u)-1
    i = np.arange(1, len(u)+1)
    rho = np.flatnonzero(u-s/i > 0)[-1]
    return np.maximum(v-s[rho]/(rho+1), 0.)


def reconstruct(x, mod, theta, hist_alpha, seed, c, private=True):
    R, b = mod["R"], mod["b"]
    edges = np.unique(np.r_[-R, np.arange(math.floor(-R/b)+1, math.ceil(R/b))*b, R])
    centers, w = (edges[:-1]+edges[1:])/2, np.diff(edges)
    z = x.copy()
    z[np.any(abs(z) > R, axis=1)] = 0
    counts = np.histogram2d(z[:, 0], z[:, 1], bins=(edges, edges))[0]
    mass = counts/len(x)
    if private:
        mass = simplex(mass+np.random.default_rng(seed).laplace(scale=2/(len(x)*hist_alpha), size=mass.shape))
    xx, yy = np.meshgrid(centers, centers, indexing="ij")
    slope = np.asarray(basis(np.c_[xx.ravel(), yy.ravel()], mod)@theta).reshape(len(w), len(w), 2)
    radius = (abs(slope[:, :, 0])*w[:, None]+abs(slope[:, :, 1])*w[None, :])/2
    scale = np.minimum(1., c.affine_safety/np.maximum(radius, 1e-300))
    slope *= scale[:, :, None]
    if abs(mass.sum()-1) > 1e-10 or mass.min() < 0:
        raise RuntimeError("Invalid reconstructed mass")
    return dict(edges=edges, mass=mass, slope=slope), dict(cell_count=mass.size,
        slope_protected_fraction=float(np.mean(scale < 1)), slope_protected_mass=float(mass[scale < 1].sum()),
        mass_sum=float(mass.sum()), raw_affine_radius_max=float(radius.max()))


@njit(cache=True)
def moments1(x, t, edges):
    a, sv = np.exp(-t), np.sqrt(-np.expm1(-2*t))
    sig, mu = sv/a, x/a
    out = np.zeros((len(edges)-1, 4))
    norm = 1/np.sqrt(2*np.pi)
    for j in range(len(edges)-1):
        l, r = edges[j], edges[j+1]
        center = (l+r)/2
        zl, zu = (a*l-x)/sv, (a*r-x)/sv
        if zl > 38 or zu < -38:
            continue
        if zl > 0:
            cdf = .5*(math.erfc(zl/np.sqrt(2))-math.erfc(zu/np.sqrt(2)))
        else:
            cdf = .5*(math.erfc(-zu/np.sqrt(2))-math.erfc(-zl/np.sqrt(2)))
        pl, pu = norm*np.exp(-.5*zl*zl), norm*np.exp(-.5*zu*zu)
        out[j, 0] = cdf/a
        out[j, 1] = (mu-center)*cdf/a+sig/a*(pl-pu)
        out[j, 2] = (pl-pu)/(a*sv)
        out[j, 3] = ((l-center)*pl-(r-center)*pu)/(a*sv)+cdf/(a*a)
    return out


@njit(cache=True)
def ou_eval(points, t, edges, mass, slope):
    out, score = np.zeros(len(points)), np.zeros((len(points), 2))
    m = len(edges)-1
    a = np.exp(-t)
    sig = np.sqrt(-np.expm1(-2*t))/a
    for k in range(len(points)):
        x, y = points[k, 0], points[k, 1]
        u, v = moments1(x, t, edges), moments1(y, t, edges)
        ilo = max(0, np.searchsorted(edges, x/a-38*sig)-1)
        ihi = min(m, np.searchsorted(edges, x/a+38*sig))
        jlo = max(0, np.searchsorted(edges, y/a-38*sig)-1)
        jhi = min(m, np.searchsorted(edges, y/a+38*sig))
        den, dx, dy = 0., 0., 0.
        for i in range(ilo, ihi):
            for j in range(jlo, jhi):
                if mass[i, j] <= 0:
                    continue
                base = mass[i, j]/((edges[i+1]-edges[i])*(edges[j+1]-edges[j]))
                sx, sy = slope[i, j, 0], slope[i, j, 1]
                den += base*(u[i, 0]*v[j, 0]+sx*u[i, 1]*v[j, 0]+sy*u[i, 0]*v[j, 1])
                dx += base*(u[i, 2]*v[j, 0]+sx*u[i, 3]*v[j, 0]+sy*u[i, 2]*v[j, 1])
                dy += base*(u[i, 0]*v[j, 2]+sx*u[i, 1]*v[j, 2]+sy*u[i, 0]*v[j, 3])
        out[k] = den
        if den > 1e-280:
            score[k, 0], score[k, 1] = dx/den, dy/den
        else:
            score[k, 0], score[k, 1] = np.nan, np.nan
    return out, score


def sample_cells(cells, count, random, c, t=0.):
    e, mass, slope = cells["edges"], cells["mass"], cells["slope"]
    w, m = np.diff(e), len(e)-1
    ids = random.choice(m*m, size=count, p=mass.ravel())
    i, j = ids//m, ids % m
    centers = np.c_[(e[i]+e[i+1])/2, (e[j]+e[j+1])/2]
    widths, s = np.c_[w[i], w[j]], slope[i, j]
    out, pending = np.empty((count, 2)), np.arange(count)
    while len(pending):
        delta = (random.random((len(pending), 2))-.5)*widths[pending]
        val = 1+(s[pending]*delta).sum(axis=1)
        if np.min(val) < -1e-12:
            raise RuntimeError("Negative affine density")
        accepted = random.random(len(pending))*(1+c.affine_safety) < val
        out[pending[accepted]] = centers[pending[accepted]]+delta[accepted]
        pending = pending[~accepted]
    if t > 0:
        out = np.exp(-t)*out+np.sqrt(-np.expm1(-2*t))*random.normal(size=out.shape)
    return out


def time_grid(n, total_alpha, gamma, c):
    delta = max(n**(-(gamma+1)/(2*gamma+4)), (n*total_alpha)**(-(gamma+1)/(gamma+3)))
    kappa = c.kappa_multiplier*delta**2
    values, T = [1/n], np.log(n)
    while values[-1] < T:
        x = values[-1]
        values.append(min(1., (1+kappa)*x) if x < 1 else min(T, x+kappa))
    tt = np.asarray(values[::-1])
    assert np.all(tt[:-1]-tt[1:] <= kappa*np.minimum(1, tt[1:])*(1+1e-10))
    return tt, delta, kappa


def reverse(cells, n, total_alpha, gamma, seed, c, checkpoint):
    tt, delta, kappa = time_grid(n, total_alpha, gamma, c)
    random = np.random.default_rng(seed)
    y = random.normal(size=(c.particles, 2))
    done, elapsed = 0, 0.
    signature = fingerprint(dict(seed=seed, particles=c.particles, n=n, total_alpha=total_alpha,
        gamma=gamma, kappa=kappa, mass=hash_array(cells["mass"]), slope=hash_array(cells["slope"])))
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        with np.load(checkpoint) as z:
            state = json.loads(str(z["state"]))
            if state["signature"] != signature:
                raise RuntimeError("Reverse checkpoint mismatch")
            y = z["y"]
        done, elapsed = state["done"], state["elapsed"]
        random.bit_generator.state = state["rng"]
    start = time.perf_counter()
    for k in range(done, len(tt)-1):
        p, s = ou_eval(y, tt[k], **cells)
        if not np.isfinite(s).all() or np.any(p <= 0):
            raise RuntimeError("Nonfinite OU score / tail underflow; no silent clipping")
        h = tt[k]-tt[k+1]
        y = np.exp(h)*y+2*np.expm1(h)*s+np.sqrt(np.expm1(2*h))*random.normal(size=y.shape)
        if (k+1) % c.reverse_checkpoint_every == 0 or k == len(tt)-2:
            atomic_npz(checkpoint, y=y, state=json.dumps(dict(signature=signature,
                done=k+1, elapsed=elapsed+time.perf_counter()-start, rng=random.bit_generator.state)))
    return y, dict(N=len(tt)-1, delta=delta, kappa=kappa,
                   reverse_seconds=elapsed+time.perf_counter()-start, reverse_seed=seed)


def density_metrics(cells, c, gamma, order):
    u, w = leggauss(order)
    e = cells["edges"]
    h, centers = np.diff(e), (e[:-1]+e[1:])/2
    x = (centers[:, None]+h[:, None]*u/2).ravel()
    weights = (h[:, None]*w/2).ravel()
    idx = np.repeat(np.arange(len(h)), order)
    total, inside = 0., 0.
    for k in range(len(x)):
        points = np.c_[np.full(len(x), x[k]), x]
        p = truth(points, c, gamma)
        i, j = idx[k], idx
        q = cells["mass"][i, j]/(h[i]*h[j])*(1+cells["slope"][i, j, 0]*(x[k]-centers[i]) +
                                                   cells["slope"][i, j, 1]*(x-centers[j]))
        total += weights[k]*(weights @ abs(q-p))
        inside += weights[k]*(weights @ p)
    return float(.5*(total+max(0., 1-inside))), float(max(0., 1-inside))


def ou_tv(cells, n, random, c, gamma=None, other=None):
    count = c.ou_metric_particles
    choose = random.random(count) < .5
    x = np.empty((count, 2))
    if other is None:
        x[choose] = draw(int(choose.sum()), random, c, gamma)
    else:
        x[choose] = sample_cells(other, int(choose.sum()), random, c, 1/n)
    x[~choose] = sample_cells(cells, int((~choose).sum()), random, c, 1/n)
    p = truth(x, c, gamma) if other is None else ou_eval(x, 1/n, **other)[0]
    q = ou_eval(x, 1/n, **cells)[0]
    if np.min(p) < -1e-12 or np.min(q) < -1e-12:
        raise RuntimeError("Negative OU density in metric")
    p, q = np.maximum(p, 0), np.maximum(q, 0)
    values = abs(p-q)/np.maximum(p+q, 1e-300)
    return float(values.mean()), float(values.std(ddof=1)/np.sqrt(count))


def binfreq(y, c):
    e = np.r_[-np.inf, np.linspace(-c.metric_bound, c.metric_bound, c.bins_per_axis+1), np.inf]
    return np.histogram2d(y[:, 0], y[:, 1], bins=(e, e))[0]/len(y)


def binned_distance(a, b, c):
    return float(.5*abs(binfreq(a, c)-binfreq(b, c)).sum())


def sliced_w1(a, b, c):
    angles = np.arange(c.sliced_directions)*np.pi/c.sliced_directions
    directions = np.c_[np.cos(angles), np.sin(angles)]
    return float(np.mean([wasserstein_distance(a@v, b@v) for v in directions]))


def evaluate_fit(theta, cells, y, mod, test, reference, seed, c):
    gamma, n = mod["gamma"], mod["n"]
    losses = np.sum((basis(test, mod)@theta-true_score(test, c, gamma))**2, axis=1)
    tv, tail = density_metrics(cells, c, gamma, c.quadrature_order)
    coarse, _ = density_metrics(cells, c, gamma, c.quadrature_order//2)
    if abs(tv-coarse) > c.quadrature_tolerance:
        raise RuntimeError(f"Density-TV quadrature needs refinement: {abs(tv-coarse)}")
    otv, ose = ou_tv(cells, n, np.random.default_rng(seed), c, gamma=gamma)
    refou = sample_cells(cells, c.reference_particles, np.random.default_rng(seed+1), c, 1/n)
    iid = sample_cells(cells, c.particles, np.random.default_rng(seed+2), c, 1/n)
    return dict(score_risk=float(losses.mean()), score_risk_inner_se=float(losses.std(ddof=1)/np.sqrt(len(test))),
        density_tv=tv, density_quadrature_delta=abs(tv-coarse), target_mass_outside_R=tail,
        ou_tv=otv, ou_tv_inner_mc_se=ose, generated_binned_tv=binned_distance(y, reference, c),
        reverse_to_ou_binned_tv=binned_distance(y, refou, c), iid_ou_binned_tv=binned_distance(iid, refou, c),
        generated_sliced_wasserstein=sliced_w1(y, reference, c),
        covariance_frobenius_error=float(np.linalg.norm(np.cov(y, rowvar=False)-tc(c, gamma)["covariance"])))


def density_pair_tv(a, b, order):
    if not np.array_equal(a["edges"], b["edges"]):
        raise RuntimeError("Sensitivity requires the same histogram partition")
    h = np.diff(a["edges"])
    u, w = leggauss(order)
    area = h[:, None]*h[None, :]
    constant = (a["mass"]-b["mass"])/area
    slope = (a["mass"][:, :, None]*a["slope"]-b["mass"][:, :, None]*b["slope"])/area[:, :, None]
    total = 0.
    for i in range(order):
        for j in range(order):
            delta = constant+slope[:, :, 0]*h[:, None]*u[i]/2+slope[:, :, 1]*h[None, :]*u[j]/2
            total += w[i]*w[j]*np.sum(area*abs(delta))/4
    return float(total/2)


def affine_moments(cells, t):
    e, p, s = cells["edges"], cells["mass"], cells["slope"]
    v, c = np.diff(e)**2/12, (e[:-1]+e[1:])/2
    mx = np.sum(p*(c[:, None]+s[:, :, 0]*v[:, None]))
    my = np.sum(p*(c[None, :]+s[:, :, 1]*v[None, :]))
    xx = np.sum(p*(c[:, None]**2+v[:, None]+2*c[:, None]*s[:, :, 0]*v[:, None]))
    yy = np.sum(p*(c[None, :]**2+v[None, :]+2*c[None, :]*s[:, :, 1]*v[None, :]))
    xy = np.sum(p*(c[:, None]*c[None, :]+c[None, :]*s[:, :, 0]*v[:, None]+
                  c[:, None]*s[:, :, 1]*v[None, :]))
    mean = np.array([mx, my])
    cov = np.array([[xx, xy], [xy, yy]])-np.outer(mean, mean)
    return np.exp(-t)*mean, np.exp(-2*t)*cov+(-np.expm1(-2*t))*np.eye(2)


def sensitivity_metrics(a, b, mod, test, c, seed):
    delta = a["theta"]-b["theta"]
    fd = basis(test, mod)@delta
    same = np.array_equal(a["cells"]["mass"], b["cells"]["mass"]) and np.array_equal(a["cells"]["slope"], b["cells"]["slope"])
    dtv = density_pair_tv(a["cells"], b["cells"], 24)
    dtvc = density_pair_tv(a["cells"], b["cells"], 12)
    if abs(dtv-dtvc) > c.quadrature_tolerance:
        raise RuntimeError("Density sensitivity quadrature needs refinement")
    otv, ose = (0., 0.) if same else ou_tv(a["cells"], mod["n"], np.random.default_rng(seed), c, other=b["cells"])
    ma, ca = affine_moments(a["cells"], 1/mod["n"])
    mb, cb = affine_moments(b["cells"], 1/mod["n"])
    return dict(coefficient_rms=float(np.sqrt(np.mean(delta**2))), coefficient_l2=float(np.linalg.norm(delta)),
        coefficient_max=float(np.max(abs(delta))), score_function_rms=float(np.sqrt(np.mean(np.sum(fd**2, axis=1)))),
        density_tv_change=dtv, density_pair_quad_delta=abs(dtv-dtvc), ou_tv_change=otv, ou_tv_change_inner_se=ose,
        ou_mean_l2_change=float(np.linalg.norm(ma-mb)), ou_covariance_frobenius_change=float(np.linalg.norm(ca-cb)),
        generated_binned_tv_change=binned_distance(a["y"], b["y"], c),
        generated_sliced_wasserstein_change=sliced_w1(a["y"], b["y"], c),
        paired_particle_rms=float(np.sqrt(np.mean(np.sum((a["y"]-b["y"])**2, axis=1)))))


def fit_pipeline(x, mod, alpha, seeds, private, c, folder, label, grid_alpha=None):
    folder = Path(folder)
    fitpath = folder/f"{label}_fit.npz"
    if fitpath.exists():
        with np.load(fitpath) as z:
            theta = z["theta"]
            cells = {k: z[k] for k in ("edges", "mass", "slope")}
            info = json.loads(str(z["info"]))
    else:
        if private:
            theta, info = lmc(mod, alpha, seeds[0], c, folder/f"{label}_lmc.npz")
        else:
            theta, info = constrained_erm(mod, c)
        cells, diagnostics = reconstruct(x, mod, theta, alpha, seeds[1], c, private)
        info.update(diagnostics)
        atomic_npz(fitpath, theta=theta, **cells, info=json.dumps(info))
    y, revinfo = reverse(cells, mod["n"], 2*(alpha if grid_alpha is None else grid_alpha), mod["gamma"],
                         seeds[2], c, folder/f"{label}_reverse.npz")
    return dict(theta=theta, cells=cells, y=y, info=dict(info, **revinfo))


def panel_job(task, n, rep, alpha, gamma, c, output):
    tag = f"n{n}_r{rep:03d}_g{gamma:g}_a{alpha:g}".replace(".", "p")
    folder = Path(output)/task
    dest = folder/"rows"/(tag+".json")
    if dest.exists():
        return json.loads(dest.read_text())
    started = time.perf_counter()
    gcode, acode = round(gamma*1000), round(alpha*1000)
    # Only the training sample is nested across n. Test/reference are independent
    # of training, and shared across budgets and n within each (rep,gamma).
    x = draw(max(c.n_values), rng(c, rep, gcode, 1), c, gamma)[:n].copy()
    test = draw(c.test_particles, rng(c, rep, gcode, 2), c, gamma)
    ref = draw(c.reference_particles, rng(c, rep, gcode, 3), c, gamma)
    mod = fit_statistics(x, make_class(n, alpha, gamma, c))
    stage = folder/"checkpoints"/tag
    stage.mkdir(parents=True, exist_ok=True)
    rows = []
    arrays = dict(data=x)
    for method, private in (("private", True), ("nonprivate", False)):
        method_code = int(private)
        seeds = [stream_seed(c, n, rep, gcode, acode if private else 0, method_code, k) for k in (10, 11, 12)]
        fitted = fit_pipeline(x, mod, alpha, seeds, private, c, stage, method)
        metrics = evaluate_fit(fitted["theta"], fitted["cells"], fitted["y"], mod, test, ref,
                               stream_seed(c, n, rep, gcode, acode if private else 0, method_code, 13), c)
        rows.append(dict(task=task, method=method, n=n, rep=rep, gamma=gamma,
            alpha_score=alpha, alpha_total=2*alpha, K=c.K, amplitude=c.amplitude,
            train_hash=hash_array(x), test_hash=hash_array(test), reference_hash=hash_array(ref),
            coefficient_count=2*mod["m"]**2, b=mod["b"], b_raw=mod["b_raw"], R=mod["R"],
            holder_L=mod["holder_L"], Q=mod["Q"], truncated_fraction=mod["truncated_fraction"],
            particles=c.particles, test_particles=c.test_particles, reference_particles=c.reference_particles,
            **fitted["info"], **metrics))
        arrays[method+"_theta"] = fitted["theta"]
        arrays[method+"_samples"] = fitted["y"]
        for k, v in fitted["cells"].items():
            arrays[method+"_"+k] = v
    arrays["class_json"] = json.dumps({k: v for k, v in mod.items() if k not in ("gram", "h", "e0", "e1")})
    atomic_npz(folder/"artifacts"/(tag+".npz"), **arrays)
    for row in rows:
        row["job_seconds_this_invocation"] = time.perf_counter()-started
    atomic_json(dest, rows)
    return rows


def sensitivity_job(n, rep, alpha, c, output):
    gamma = c.sensitivity_gamma
    tag = f"n{n}_r{rep:03d}_a{alpha:g}".replace(".", "p")
    folder = Path(output)/"sensitivity"
    dest = folder/"rows"/(tag+".json")
    if dest.exists():
        return json.loads(dest.read_text())
    started = time.perf_counter()
    gcode, acode = round(gamma*1000), round(alpha*1000)
    x = draw(max(c.n_values), rng(c, rep, gcode, 101), c, gamma)[:n].copy()
    random = rng(c, n, rep, 102)
    idx = int(random.integers(n))
    xp = x.copy()
    if c.neighbor_mode == "replace":
        xp[idx] = draw(1, random, c, gamma)[0]
    elif c.neighbor_mode == "shift":
        xp[idx] += np.asarray(c.perturbation)
    else:
        raise ValueError(c.neighbor_mode)
    assert np.count_nonzero(np.any(x != xp, axis=1)) == 1
    cls = make_class(n, c.sensitivity_reference_score_alpha, gamma, c)
    mod, neighbor = fit_statistics(x, cls), fit_statistics(xp, cls)
    test = draw(c.test_particles, rng(c, rep, gcode, 103), c, gamma)
    stage = folder/"checkpoints"/tag
    seeds_a = [stream_seed(c, n, rep, acode, 110+k) for k in range(3)]
    seeds_b = [stream_seed(c, n, rep, acode, 120+k) for k in range(3)]
    # Same Gaussian drivers across neighbors preserve each algorithm's marginal
    # even though its data-dependent Lipschitz step size can differ. Hist noise
    # and reverse Brownian drivers are also coupled. Save both eta values.
    fits = {}
    for name, data, model, seeds, private in (
        ("private_D", x, mod, seeds_a, True),
        ("private_neighbor_crn", xp, neighbor, seeds_a, True),
        ("private_repeat", x, mod, seeds_b, True),
        ("private_neighbor_independent", xp, neighbor, seeds_b, True),
        ("nonprivate_D", x, mod, seeds_a, False),
        ("nonprivate_neighbor_crn", xp, neighbor, seeds_a, False)):
        fits[name] = fit_pipeline(data, model, alpha, seeds, private, c, stage, name,
                                  grid_alpha=c.sensitivity_reference_score_alpha)
    base_np = fits["nonprivate_D"]
    y_np, _ = reverse(base_np["cells"], n, 2*c.sensitivity_reference_score_alpha, gamma,
                       seeds_b[2], c, stage/"nonprivate_repeat_reverse.npz")
    fits["nonprivate_repeat"] = dict(base_np, y=y_np)
    rows, arrays = [], dict(data=x, neighbor_data=xp, test=test, replaced_index=idx)
    pairs = (("private_crn", "private_D", "private_neighbor_crn"),
             ("private_independent", "private_D", "private_neighbor_independent"),
             ("private_repeat", "private_D", "private_repeat"),
             ("nonprivate_crn", "nonprivate_D", "nonprivate_neighbor_crn"),
             ("nonprivate_repeat", "nonprivate_D", "nonprivate_repeat"))
    for pair_id, (mode, a, b) in enumerate(pairs):
        metrics = sensitivity_metrics(fits[a], fits[b], mod, test, c,
                                      stream_seed(c, n, rep, acode, 140+pair_id))
        rows.append(dict(task="sensitivity", mode=mode, n=n, rep=rep, gamma=gamma,
            alpha_score=alpha, alpha_total=2*alpha, reference_score_alpha=c.sensitivity_reference_score_alpha,
            neighbor_mode=c.neighbor_mode, replaced_index=idx,
            old_x1=float(x[idx, 0]), old_x2=float(x[idx, 1]), new_x1=float(xp[idx, 0]), new_x2=float(xp[idx, 1]),
            train_hash=hash_array(x), neighbor_hash=hash_array(xp), coefficient_count=2*mod["m"]**2,
            lmc_eta_base=fits[a]["info"].get("lmc_eta"), lmc_eta_comparator=fits[b]["info"].get("lmc_eta"),
            N=fits[a]["info"]["N"], particles=c.particles,
            **metrics, job_seconds_this_invocation=time.perf_counter()-started))
    for name, fit in fits.items():
        arrays[name+"_theta"] = fit["theta"]
        arrays[name+"_samples"] = fit["y"]
        arrays[name+"_info"] = json.dumps(fit["info"])
        for key, value in fit["cells"].items():
            arrays[name+"_"+key] = value
    arrays["class_json"] = json.dumps({k: v for k, v in cls.items() if k not in ("e0", "e1")})
    atomic_npz(folder/"artifacts"/(tag+".npz"), **arrays)
    atomic_json(dest, rows)
    return rows


def run_job(job):
    task, n, rep, alpha, gamma, c, output = job
    try:
        if task == "sensitivity":
            return sensitivity_job(n, rep, alpha, c, output)
        return panel_job(task, n, rep, alpha, gamma, c, output)
    except BaseException:
        atomic_json(Path(output)/"failures"/f"{task}_n{n}_r{rep}_a{alpha}_g{gamma}.json",
                    dict(task=task, n=n, rep=rep, alpha_score=alpha, gamma=gamma,
                         traceback=traceback.format_exc(), timestamp=time.time()))
        raise


PANEL_METRICS = ("score_risk", "density_tv", "ou_tv", "generated_binned_tv",
                 "reverse_to_ou_binned_tv", "iid_ou_binned_tv", "generated_sliced_wasserstein",
                 "covariance_frobenius_error")
SENSITIVITY_METRICS = ("coefficient_rms", "coefficient_l2", "coefficient_max", "score_function_rms",
    "density_tv_change", "ou_tv_change", "ou_mean_l2_change", "ou_covariance_frobenius_change",
    "generated_binned_tv_change", "generated_sliced_wasserstein_change", "paired_particle_rms")


def aggregate(output):
    import pandas as pd
    output = Path(output)
    for task in ("alpha", "gamma", "sensitivity"):
        root = output/task
        rows = []
        for p in sorted((root/"rows").glob("*.json")):
            rows.extend(json.loads(p.read_text()))
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.to_csv(root/"raw.csv", index=False)
        param = "alpha_score" if task != "gamma" else "gamma"
        method = "mode" if task == "sensitivity" else "method"
        metrics = SENSITIVITY_METRICS if task == "sensitivity" else PANEL_METRICS
        summaries = []
        for (n, value, how), group in df.groupby(["n", param, method]):
            for metric in metrics:
                x = group[metric].to_numpy()
                mean = float(x.mean())
                se = float(x.std(ddof=1)/np.sqrt(len(x))) if len(x) > 1 else np.nan
                half = float(student_t.ppf(.975, len(x)-1)*se) if len(x) > 1 else np.nan
                summaries.append(dict(n=n, parameter=value, method=how, metric=metric,
                    repetitions=len(x), mean=mean, se=se, ci95_low=mean-half, ci95_high=mean+half))
        summary = pd.DataFrame(summaries)
        summary.to_csv(root/"summary.csv", index=False)


def validate(c, output):
    from scipy.optimize import minimize, LinearConstraint, Bounds
    report = {}
    random = rng(c, 881)
    for gamma in sorted(set(c.gammas+(c.alpha_panel_gamma,))):
        target = tc(c, gamma)
        assert np.isfinite(target["holder_L"]) and target["holder_L"] > 0
        report[f"target_gamma_{gamma}"] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in target.items()}
    assert np.max(abs(tc(c, .5)["covariance"]-np.array([[1., .5], [.5, 1.]]))) < 1e-12
    assert np.array_equal(tc(c, .2)["B"], tc(c, 1.)["B"])
    # High-frequency terms invalidate a coarse finite-difference derivative
    # check. Test the same analytic implementation on a resolved K=5 target.
    small = replace(c, K=5)
    points = random.normal(size=(30, 2))
    errors = []
    for gamma in (.2, .5, 1.):
        numeric = np.empty_like(points)
        for axis in (0, 1):
            delta = np.zeros_like(points)
            delta[:, axis] = 1e-6
            numeric[:, axis] = (np.log(truth(points+delta, small, gamma))-
                                     np.log(truth(points-delta, small, gamma)))/(2e-6)
        errors.append(float(np.max(abs(numeric-true_score(points, small, gamma)))))
    assert max(errors) < 1e-6
    report["resolved_target_score_derivative_max"] = max(errors)
    mod = make_class(200, 2., .5, c)
    assert np.max(abs(np.asarray(basis(points, mod).sum(axis=1))-1)) < 1e-12
    for axis in (0, 1):
        assert np.max(abs(np.asarray(basis(points, mod, axis).sum(axis=1)))) < 1e-12
    assert basis(np.array([[1e4, 0.], [-1e4, 0.], [0., 1e4]]), mod).nnz == 0
    m = 3
    vertices = np.arange(m*m).reshape(m, m)
    e0 = np.r_[vertices[:-1].ravel(), vertices[:, :-1].ravel()]
    e1 = np.r_[vertices[1:].ravel(), vertices[:, 1:].ravel()]
    constraint = np.zeros((len(e0), m*m))
    constraint[np.arange(len(e0)), e0] = -1
    constraint[np.arange(len(e0)), e1] = 1
    y = random.normal(size=(9, 2))*3
    projected, _, _, ok = project(y, 2., .4, e0, e1, 10000, 1e-11)
    assert ok
    differences = []
    for axis in (0, 1):
        result = minimize(lambda z: .5*np.sum((z-y[:, axis])**2), np.zeros(9),
            jac=lambda z: z-y[:, axis], bounds=Bounds(-2, 2),
            constraints=LinearConstraint(constraint, -.4, .4), method="SLSQP",
            options=dict(ftol=1e-12, maxiter=1000))
        assert result.success
        differences.append(float(np.max(abs(result.x-projected[:, axis]))))
    assert max(differences) < 1e-6
    report["projection_vs_QP_max"] = max(differences)
    e = np.array([-1., -.2, .7, 1.])
    mass = random.random((3, 3)); mass /= mass.sum()
    slope = random.uniform(-.25, .25, size=(3, 3, 2))
    cells = dict(edges=e, mass=mass, slope=slope)
    nodes, weights = [], []
    u, w = leggauss(24)
    for i in range(3):
        for j in range(3):
            center = np.array([(e[i]+e[i+1])/2, (e[j]+e[j+1])/2])
            width = np.array([e[i+1]-e[i], e[j+1]-e[j]])
            xx, yy = np.meshgrid(u, u, indexing="ij")
            z = center+np.c_[xx.ravel(), yy.ravel()]*width/2
            nodes.append(z)
            weights.append(np.outer(w, w).ravel()/4*mass[i, j]*(1+(z-center)@slope[i, j]))
    nodes, weights = np.concatenate(nodes), np.concatenate(weights)
    perr, serr = [], []
    for t in (.05, .2, 1.):
        p, s = ou_eval(points[:8], t, **cells)
        for k, point in enumerate(points[:8]):
            diff = point-np.exp(-t)*nodes
            var = -np.expm1(-2*t)
            kernel = np.exp(-np.sum(diff**2, axis=1)/(2*var))/(2*np.pi*var)*weights
            pp = kernel.sum()
            ss = -(kernel[:, None]*diff).sum(axis=0)/(var*pp)
            perr.append(abs(pp-p[k])); serr.append(np.max(abs(ss-s[k])))
    assert max(perr) < 1e-9 and max(serr) < 1e-7
    report["ou_density_vs_quadrature_max"] = float(max(perr))
    report["ou_score_vs_quadrature_max"] = float(max(serr))
    mean, cov = affine_moments(cells, 0.)
    mean_quad = np.sum(weights[:, None]*nodes, axis=0)
    assert np.max(abs(mean-mean_quad)) < 1e-12
    cov_quad = (nodes-mean).T@(weights[:, None]*(nodes-mean))
    assert np.max(abs(cov-cov_quad)) < 1e-12
    assert density_pair_tv(cells, cells, 16) == 0
    report["affine_moments_and_identical_density_distance"] = "PASS"
    for n in c.n_values:
        for gamma in c.gammas:
            for alpha in c.score_alphas:
                time_grid(n, 2*alpha, gamma, c)
    report["basis_and_time_grid"] = "PASS"
    # Check the quadratic score-matching gradient and zero-start LMC restart.
    train = draw(200, random, c, .5)
    model = fit_statistics(train, mod)
    theta = random.normal(size=model["h"].shape)*.01
    direction = random.normal(size=theta.shape)
    direction /= np.linalg.norm(direction)
    A, h = model["gram"], model["h"]
    def objective(z):
        return .5*np.sum(z*(A@z))+np.sum(h*z)
    derivative = (objective(theta+1e-5*direction)-objective(theta-1e-5*direction))/2e-5
    assert abs(derivative-np.sum((A@theta+h)*direction)) < 1e-7
    short = replace(c, lmc_steps=10, lmc_chunk=5, lmc_checkpoint_every=5, lmc_step_factor=.01)
    # Validation has its own unique scratch paths; no experiment rows are touched.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="lmc_resume_", dir=output) as tempdir:
        whole, _ = lmc(model, 2., 5512, short, Path(tempdir)/"whole.npz")
        original_writer = globals()["atomic_npz"]
        def interrupt_after_save(path, **arrays):
            original_writer(path, **arrays)
            raise InterruptedError("validation interruption after checkpoint")
        globals()["atomic_npz"] = interrupt_after_save
        try:
            lmc(model, 2., 5512, short, Path(tempdir)/"resumed.npz")
        except InterruptedError:
            pass
        finally:
            globals()["atomic_npz"] = original_writer
        resumed, _ = lmc(model, 2., 5512, short, Path(tempdir)/"resumed.npz")
        assert np.array_equal(whole, resumed)
        small_reverse = replace(c, particles=32, reverse_checkpoint_every=10)
        whole_y, _ = reverse(cells, 50, 4., .5, 117, small_reverse, Path(tempdir)/"whole_reverse.npz")
        globals()["atomic_npz"] = interrupt_after_save
        try:
            reverse(cells, 50, 4., .5, 117, small_reverse, Path(tempdir)/"resumed_reverse.npz")
        except InterruptedError:
            pass
        finally:
            globals()["atomic_npz"] = original_writer
        resumed_y, _ = reverse(cells, 50, 4., .5, 117, small_reverse, Path(tempdir)/"resumed_reverse.npz")
        assert np.array_equal(whole_y, resumed_y)
    report["loss_gradient_and_exact_checkpoint_resume"] = "PASS"
    report["status"] = "PASS"
    atomic_json(Path(output)/"validation.json", report)
    print("Independent numerical validation: PASS", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("validate", "alpha", "gamma", "sensitivity", "all", "summarize"), default="validate")
    parser.add_argument("--profile", choices=("smoke", "pilot", "paper"), default="pilot")
    parser.add_argument("--output", type=Path, default=Path("two_dimensional_outputs"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-values", type=int, nargs="+")
    parser.add_argument("--score-alphas", type=float, nargs="+")
    parser.add_argument("--gammas", type=float, nargs="+")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--rep-start", type=int, default=0,
                        help="First repetition index to run, inclusive (default: 0)")
    parser.add_argument("--rep-stop", type=int,
                        help="Last repetition index to run, exclusive (default: configured repetitions)")
    parser.add_argument("--lmc-steps", type=int)
    parser.add_argument("--step-factor", type=float)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--ridge-mode", choices=("theoretical", "legacy"))
    parser.add_argument("--neighbor-mode", choices=("replace", "shift"))
    parser.add_argument("--perturbation", type=float, nargs=2)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Prevent two invocations from writing the same checkpoints or summary.
    import fcntl
    run_lock = (output/"run.lock").open("a")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another process is using this output directory")
    if args.task == "summarize":
        aggregate(output)
        return
    c = profile(args.profile)
    overrides = {}
    for field in ("n_values", "score_alphas", "gammas", "repetitions", "lmc_steps", "particles", "ridge_mode", "neighbor_mode", "perturbation"):
        value = getattr(args, field)
        if value is not None:
            overrides[field] = tuple(value) if isinstance(value, list) else value
    if args.step_factor is not None:
        overrides["lmc_step_factor"] = args.step_factor
    c = replace(c, **overrides)
    if min(c.n_values) < 2 or min(c.score_alphas) <= 0 or not all(0 < g <= 1 for g in c.gammas):
        parser.error("Require n>=2, positive alpha, and 0<gamma<=1")
    if min(c.repetitions, c.lmc_steps, c.particles, args.workers) < 1:
        parser.error("Counts and worker number must be positive")
    rep_stop = c.repetitions if args.rep_stop is None else args.rep_stop
    if not 0 <= args.rep_start < rep_stop <= c.repetitions:
        parser.error("Require 0 <= rep-start < rep-stop <= configured repetitions")
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = dict(config=asdict(c), source_sha256=source_hash, target="fixed_B_multiscale_product",
                    binned_tv="two_sample_with_overflow", coefficient_model="joint_tensor_spline")
    signature = fingerprint(manifest)
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["signature"] != signature:
            raise RuntimeError("Configuration/source differs from this output. Choose a new output directory.")
    else:
        atomic_json(manifest_path, dict(manifest, signature=signature))
    validate(c, output)
    if args.task == "validate":
        return
    tasks = ("alpha", "gamma", "sensitivity") if args.task == "all" else (args.task,)
    jobs = []
    for task in tasks:
        for rep in range(args.rep_start, rep_stop):
            for n in c.n_values:
                pairs = ((a, c.alpha_panel_gamma) for a in c.score_alphas) if task != "gamma" else (
                    (c.gamma_panel_score_alpha, g) for g in c.gammas)
                for alpha, gamma in pairs:
                    jobs.append((task, n, rep, alpha, gamma, c, str(output)))
    atomic_json(output/"status.json", dict(state="running", task=args.task,
        rep_start=args.rep_start, rep_stop=rep_stop, configured_repetitions=c.repetitions,
        planned_jobs=len(jobs), updated=time.time()))
    start = time.perf_counter()
    try:
        if args.workers == 1:
            for i, job in enumerate(jobs, 1):
                run_job(job)
                aggregate(output)
                print(f"{i}/{len(jobs)} complete: {job[:5]}", flush=True)
                atomic_json(output/"status.json", dict(state="running", task=args.task,
                    rep_start=args.rep_start, rep_stop=rep_stop, configured_repetitions=c.repetitions,
                    completed_jobs=i, planned_jobs=len(jobs), updated=time.time()))
        else:
            # Pool context terminates remaining workers on a numerical failure;
            # successful checkpoints remain available for an explicit restart.
            with get_context("spawn").Pool(args.workers) as pool:
                for i, result in enumerate(pool.imap_unordered(run_job, jobs), 1):
                    aggregate(output)
                    print(f"{i}/{len(jobs)} complete: {result[0]['task']} n={result[0]['n']} rep={result[0]['rep']}", flush=True)
                    atomic_json(output/"status.json", dict(state="running", task=args.task,
                        rep_start=args.rep_start, rep_stop=rep_stop, configured_repetitions=c.repetitions,
                        completed_jobs=i, planned_jobs=len(jobs), updated=time.time()))
        aggregate(output)
    except BaseException:
        atomic_json(output/"status.json", dict(state="stopped_with_error", error=traceback.format_exc(), updated=time.time()))
        raise
    final_state = "complete" if args.rep_start == 0 and rep_stop == c.repetitions else "batch_complete"
    atomic_json(output/"status.json", dict(state=final_state, task=args.task,
        rep_start=args.rep_start, rep_stop=rep_stop, configured_repetitions=c.repetitions,
        completed_jobs=len(jobs), planned_jobs=len(jobs),
        seconds_this_invocation=time.perf_counter()-start, updated=time.time(),
        note="Numerical completion is not an LMC convergence or finite-run privacy certificate."))


if __name__ == "__main__":
    main()
