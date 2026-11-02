#!/usr/bin/env python3
"""Reproducible projected-LMC end-to-end pilot, checkpointing every replicate."""
import os
import argparse, json, time, sys, platform, hashlib, traceback
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.integrate import cumulative_trapezoid
from scipy.stats import wasserstein_distance
import diffusion_extension_lmc as d
import paper_numerics as pn
pn.install()
from dp_score_sim import density
ROOT=Path(__file__).resolve().parents[1]

def target(x,c):return d.target_density(x,c.gamma,c.target,c.amplitude,c.K,c.lam,c.mixture_mu)
def draw(n,rng,c):return d.sample_target(n,c.gamma,rng,c.target,c.amplitude,c.K,c.lam,c.mixture_mu)

def density_tv(cells,c,order):
    z,w=leggauss(order)
    xx=cells.centers[:,None]+cells.widths[:,None]/2*z
    sur=cells.masses[:,None]/cells.widths[:,None]*(1+cells.slopes[:,None]*(xx-cells.centers[:,None]))
    inside=.5*np.sum(np.abs(sur-target(xx,c))*w*cells.widths[:,None]/2)
    # truth outside the compact support contributes half its mass.
    ext=np.array([-12.,cells.left[0],cells.right[-1],12.])
    tail=0.
    for a,b in [(ext[0],ext[1]),(ext[2],ext[3])]:
        xs=(a+b)/2+(b-a)/2*z;tail+=np.sum(target(xs,c)*w)*(b-a)/2
    return float(inside+.5*tail)

def bin_masses(fn,bins,xmax,order):
    # Same bins for every n, with separate left and right tail bins.
    edges=np.r_[-12.,np.linspace(-xmax,xmax,bins+1),12.]
    z,w=leggauss(order);xs=(edges[:-1,None]+edges[1:,None])/2+np.diff(edges)[:,None]/2*z
    vals=fn(xs.ravel()).reshape(xs.shape)
    masses=np.sum(vals*w,axis=1)*np.diff(edges)/2
    if abs(masses.sum()-1)>2e-6 or np.min(masses)<-1e-12:
        raise RuntimeError(f'Invalid quadrature masses: sum={masses.sum()}, min={masses.min()}')
    return masses/masses.sum()

def btv(y,masses,c):
    edges=np.r_[-np.inf,np.linspace(-c.xmax,c.xmax,c.tv_bins+1),np.inf]
    hist=np.histogram(y,bins=edges)[0]/len(y)
    return float(.5*np.abs(hist-masses).sum())

def reverse(cells,n,kappa,c,rng,score_grid=None):
    # Original Euler-Maruyama update and adaptive time grid, with stable OU
    # moments and a grid that follows the deterministic support +/- 8 SD.
    times=d.adaptive_forward_times(np.log(n),1/n,kappa)
    y=rng.normal(size=c.particles)
    pts=c.score_grid if score_grid is None else score_grid
    u=np.linspace(-1,1,pts)
    outside=0;start=time.perf_counter()
    for i,t in enumerate(times[:-1]):
        h=t-times[i+1];a=np.exp(-t);sd=np.sqrt(-np.expm1(-2*t))
        bound=a*max(abs(cells.left[0]),abs(cells.right[-1]))+8*sd
        xx=u*bound
        _,ss=pn.ou_affine_density_score(xx,cells,t)
        if not np.all(np.isfinite(ss)):raise RuntimeError('Nonfinite OU score')
        sv=np.interp(y,xx,ss)
        mask=np.abs(y)>bound
        outside+=int(mask.sum())
        if np.any(mask):sv[mask]=pn.ou_affine_density_score(y[mask],cells,t)[1]
        y+=h*(y+2*sv)+np.sqrt(2*h)*rng.normal(size=len(y))
    if not np.all(np.isfinite(y)):raise RuntimeError('Nonfinite generated sample')
    return y,len(times)-1,outside,time.perf_counter()-start

def one_run(n,rep,c,out):
    total=time.perf_counter(); times={}
    ss=np.random.SeedSequence([c.seed,n,rep]);r=[np.random.default_rng(x) for x in ss.spawn(6)]
    cfg=d.SimConfig(gamma=c.gamma,target=c.target,amplitude=c.amplitude,K=c.K,lam=c.lam,mixture_mu=c.mixture_mu)
    t=time.perf_counter();x=draw(n,r[0],c)
    model=d.build_model(x,c.gamma,c.alpha*c.privacy_split,c.target,c.amplitude,c.K,c.lam,c.mixture_mu,cfg.bandwidth_factor,cfg.R_factor,cfg.coef_bound_factor)
    times['data_model_seconds']=time.perf_counter()-t
    t=time.perf_counter()
    theta,info=d.projected_lmc_draw(model,r[1],c.lmc_steps,c.ridge,c.lmc_step_factor,None,None,0,c.projection_cycles,c.projection_tol)
    times['lmc_seconds']=time.perf_counter()-t
    if info.max_projection_violation>1e-8:raise RuntimeError(f'Projection infeasible: {info}')
    t=time.perf_counter()
    hist=d.histogram_masses(x,d.make_uniform_edges(model.R,c.h_factor*model.b),c.alpha*(1-c.privacy_split),r[2],True)
    slopes=d.score_from_theta(model,theta,hist.centers);cells=d.affine_cells_from_hist(hist,slopes)
    times['reconstruction_seconds']=time.perf_counter()-t
    t=time.perf_counter()
    tv=density_tv(cells,c,192); tvcoarse=density_tv(cells,c,96)
    grid=np.linspace(-c.xmax,c.xmax,c.metric_grid)
    truth=target(grid,c);pt,_=pn.ou_affine_density_score(grid,cells,1/n)
    tvpt=d.grid_tv(truth,pt,grid);tvptcoarse=d.grid_tv(truth[::2],pt[::2],grid[::2])
    ptmass=float(np.trapezoid(pt,grid))
    if abs(ptmass-1)>2e-6:raise RuntimeError(f'OU mass {ptmass}')
    mp=bin_masses(lambda z:pn.ou_affine_density_score(z,cells,1/n)[0],c.tv_bins,c.xmax,128)
    mpcoarse=bin_masses(lambda z:pn.ou_affine_density_score(z,cells,1/n)[0],c.tv_bins,c.xmax,64)
    mt=bin_masses(lambda z:target(z,c),c.tv_bins,c.xmax,128)
    times['density_metrics_seconds']=time.perf_counter()-t
    delta=d.delta_theory(n,c.alpha,c.gamma);kappa=c.kappa_multiplier*delta**2
    y,N,outside,revtime=reverse(cells,n,kappa,c,r[3]);times['reverse_seconds']=revtime
    t=time.perf_counter()
    refpt=d.sample_ptau(cells,1/n,c.reference_particles,r[4]);reftrue=draw(c.reference_particles,r[5],c)
    xt=draw(c.n_test,np.random.default_rng(np.random.SeedSequence([c.seed,n,rep,99])),c)
    risk=d.theta_risk(model,theta,xt,c.gamma,c.target,c.amplitude,c.K,c.lam,c.mixture_mu)
    row=dict(n=n,rep=rep,seed=c.seed,lmc_steps=c.lmc_steps,particles=c.particles,delta_poly=delta,kappa=kappa,N=N,
        TV_surrogate_true=tv,TV_Ptau_true=tvpt,
        TVproxy_generated_true=btv(y,mt,c),TVproxy_generated_Ptau=btv(y,mp,c),
        TVproxy_MC_floor_Ptau=btv(refpt[:c.particles],mp,c),TVproxy_MC_floor_true=btv(reftrue[:c.particles],mt,c),
        W1_generated_true=float(wasserstein_distance(y,reftrue)),W1_generated_Ptau=float(wasserstein_distance(y,refpt)),
        score_risk=risk,lmc_step_size=info.step_size,lmc_proj_violation=info.max_projection_violation,
        lmc_mean_projection_cycles=info.mean_projection_cycles,lmc_lipschitz=info.lipschitz,
        b=model.b,R=model.R,basis_dim=model.idx.size,Q=model.Q,Q_box_fraction=model.Q_box/model.Q,
        density_TV_quad_delta=abs(tv-tvcoarse),OU_TV_grid_delta=abs(tvpt-tvptcoarse),
        OU_bin_mass_quad_delta=float(np.max(abs(mp-mpcoarse))),OU_mass=ptmass,
        reverse_outside_grid_evaluations=outside)
    times['sample_metrics_seconds']=time.perf_counter()-t
    times['total_seconds']=time.perf_counter()-total;row.update(times)
    art=out/'replicates';art.mkdir(exist_ok=True)
    np.savez_compressed(art/f'n{n}_rep{rep}.npz',x=x,theta=theta,generated=y,
        **{'cells_'+k:v for k,v in asdict(cells).items()},grid=grid,truth=truth,ptau=pt,
        bin_masses_true=mt,bin_masses_ptau=mp)
    return row

def summarize(raw,out):
    metrics=[x for x in raw.select_dtypes(include='number').columns if x not in ['n','rep','seed']]
    rows=[]
    for n,g in raw.groupby('n'):
        row={'n':n,'repetitions':len(g)}
        for k in metrics:
            row[k+'_mean']=g[k].mean();row[k+'_sd']=g[k].std(ddof=1);row[k+'_se']=g[k].sem()
        rows.append(row)
    summ=pd.DataFrame(rows);summ.to_csv(out/'summary.csv',index=False)
    return summ



