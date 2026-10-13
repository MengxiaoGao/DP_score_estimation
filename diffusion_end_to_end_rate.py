#!/usr/bin/env python3
"""End-to-end diffusion-extension experiment across sample sizes.

Repeats the full mechanism:
  data -> projected LMC score -> DP histogram/local-affine density
       -> exact OU score -> reverse OU samples.

This complements diffusion_extension_lmc.py, which conditions on one private
surrogate and focuses on reverse-time discretization.
"""
from __future__ import annotations

import argparse, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
from diffusion_extension_lmc import (
    SimConfig, sample_target, build_model, score_from_theta, theta_risk,
    target_density, make_uniform_edges, histogram_masses, evaluate_local_affine, grid_tv,
    projected_lmc_draw, affine_cells_from_hist, ou_affine_density_score,
    reverse_ou_adaptive_samples, sample_ptau, binned_tv_samples_vs_density,
    delta_theory,
)


def one_run(n,args,rep):
    seed=args.seed + 100000*rep + n
    rng=np.random.default_rng(seed)
    cfg=SimConfig(gamma=args.gamma,target=args.target,amplitude=args.amplitude,K=args.K,
                  lam=args.lam,mixture_mu=args.mixture_mu)
    x=sample_target(n,cfg.gamma,rng,cfg.target,cfg.amplitude,cfg.K,cfg.lam,cfg.mixture_mu)
    xt=sample_target(args.n_test,cfg.gamma,rng,cfg.target,cfg.amplitude,cfg.K,cfg.lam,cfg.mixture_mu)
    a_score=args.privacy_split*args.alpha
    a_hist=(1-args.privacy_split)*args.alpha
    model=build_model(x,cfg.gamma,a_score,cfg.target,cfg.amplitude,cfg.K,cfg.lam,cfg.mixture_mu,
                      cfg.bandwidth_factor,cfg.R_factor,cfg.coef_bound_factor)
    theta,info=projected_lmc_draw(model,rng,args.lmc_steps,args.ridge,args.lmc_step_factor,
                                  xt,cfg,0,args.projection_cycles)
    srisk=theta_risk(model,theta,xt,cfg.gamma,cfg.target,cfg.amplitude,cfg.K,cfg.lam,cfg.mixture_mu)
    edges=make_uniform_edges(model.R,args.h_factor*model.b)
    hist=histogram_masses(x,edges,a_hist,rng,replacement_adjacency=True)
    slopes=score_from_theta(model,theta,hist.centers)
    cells=affine_cells_from_hist(hist,slopes)

    xmax=max(args.xmax,model.R+4.0,abs(cfg.mixture_mu)+6 if cfg.target=='mixture' else args.xmax)
    grid=np.linspace(-xmax,xmax,args.metric_grid)
    truth=target_density(grid,cfg.gamma,cfg.target,cfg.amplitude,cfg.K,cfg.lam,cfg.mixture_mu)
    sur=evaluate_local_affine(hist,slopes,grid)
    tv_sur=grid_tv(truth,sur,grid)
    T=math.log(n)
    tau=1/n
    pt,_=ou_affine_density_score(grid,cells,tau)
    pt=pt/max(np.trapezoid(pt,grid),1e-15)
    tv_pt=grid_tv(truth,pt,grid)
    tv_early=grid_tv(sur,pt,grid)

    delta=delta_theory(n,args.alpha,cfg.gamma)
    kappa=args.kappa_multiplier*delta**2
    score_grid=np.linspace(-xmax,xmax,args.score_grid)
    y,N=reverse_ou_adaptive_samples(cells,T,tau,kappa,args.particles,rng,score_grid)
    true_fn=lambda z: target_density(np.asarray(z),cfg.gamma,cfg.target,cfg.amplitude,cfg.K,cfg.lam,cfg.mixture_mu)
    pt_fn=lambda z: ou_affine_density_score(np.asarray(z),cells,tau)[0]
    tv_gen_true=binned_tv_samples_vs_density(y,true_fn,-xmax,xmax,args.tv_bins)
    tv_gen_pt=binned_tv_samples_vs_density(y,pt_fn,-xmax,xmax,args.tv_bins)

    reftrue=sample_target(args.reference_particles,cfg.gamma,rng,cfg.target,cfg.amplitude,cfg.K,cfg.lam,cfg.mixture_mu)
    refpt=sample_ptau(cells,tau,args.reference_particles,rng)
    w_true=float(wasserstein_distance(y,reftrue))
    w_pt=float(wasserstein_distance(y,refpt))
    floor=binned_tv_samples_vs_density(refpt[:args.particles],pt_fn,-xmax,xmax,args.tv_bins)

    return dict(n=n,rep=rep,delta_poly=delta,kappa=kappa,N=N,score_risk=srisk,
                TV_surrogate_true=tv_sur,TV_Ptau_true=tv_pt,TV_surrogate_Ptau=tv_early,
                TVproxy_generated_true=tv_gen_true,TVproxy_generated_Ptau=tv_gen_pt,
                TVproxy_MC_floor_Ptau=floor,W1_generated_true=w_true,W1_generated_Ptau=w_pt,
                lmc_step_size=info.step_size,lmc_proj_violation=info.max_projection_violation,
                b=model.b,R=model.R,basis_dim=model.idx.size)


def run(args):
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for n in args.n_values:
        for rep in range(args.reps):
            row=one_run(n,args,rep); rows.append(row)
            print(f"n={n:5d} rep={rep}: TVsur={row['TV_surrogate_true']:.4f}, "
                  f"TVPtau={row['TV_Ptau_true']:.4f}, TVgen={row['TVproxy_generated_true']:.4f}, "
                  f"TVgen->Ptau={row['TVproxy_generated_Ptau']:.4f}, N={row['N']}")
    raw=pd.DataFrame(rows); raw.to_csv(out/'end_to_end_raw.csv',index=False)
    summ=raw.groupby('n',as_index=False).agg(
        mean_TV_sur=('TV_surrogate_true','mean'),se_TV_sur=('TV_surrogate_true','sem'),
        mean_TV_Ptau=('TV_Ptau_true','mean'),se_TV_Ptau=('TV_Ptau_true','sem'),
        mean_TV_gen=('TVproxy_generated_true','mean'),se_TV_gen=('TVproxy_generated_true','sem'),
        mean_TV_gen_Ptau=('TVproxy_generated_Ptau','mean'),se_TV_gen_Ptau=('TVproxy_generated_Ptau','sem'),
        mean_floor=('TVproxy_MC_floor_Ptau','mean'),mean_W1_gen=('W1_generated_true','mean'),
        mean_score_risk=('score_risk','mean'),mean_N=('N','mean'),mean_delta=('delta_poly','mean'))
    summ.to_csv(out/'end_to_end_summary.csv',index=False)

    fig,ax=plt.subplots(figsize=(7.4,5.2))
    ax.errorbar(summ.n,summ.mean_TV_sur,yerr=summ.se_TV_sur,marker='o',capsize=3,label=r'exact TV$(\tilde\rho_D,\rho_0)$')
    ax.errorbar(summ.n,summ.mean_TV_Ptau,yerr=summ.se_TV_Ptau,marker='s',capsize=3,label=r'exact TV$(P_\tau\tilde\rho_D,\rho_0)$')
    ax.errorbar(summ.n,summ.mean_TV_gen,yerr=summ.se_TV_gen,marker='^',capsize=3,label='reverse samples vs truth (binned TV)')
    # Theory-shape line, rescaled to first surrogate point.
    ref=summ.mean_delta.to_numpy(float)
    scale=summ.mean_TV_sur.iloc[0]/ref[0]
    ax.plot(summ.n,scale*ref,linestyle='--',label=r'polynomial $\delta_n$ shape (rescaled)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('sample size n'); ax.set_ylabel('TV / binned-TV proxy')
    ax.set_title('End-to-end private diffusion sampling')
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(out/'end_to_end_vs_n.png',dpi=200,bbox_inches='tight')
    fig.savefig(out/'end_to_end_vs_n.pdf',bbox_inches='tight'); plt.close(fig)
    print('\n',summ.to_string(index=False))


def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--outdir',default='outputs/diffusion_end_to_end')
    p.add_argument('--target',choices=['multiscale','cusp','mixture'],default='multiscale')
    p.add_argument('--gamma',type=float,default=.5); p.add_argument('--amplitude',type=float,default=.25)
    p.add_argument('--K',type=int,default=10); p.add_argument('--lam',type=float,default=.25)
    p.add_argument('--mixture-mu',type=float,default=2.)
    p.add_argument('--n-values',nargs='+',type=int,default=[800,1600,3200])
    p.add_argument('--alpha',type=float,default=1.); p.add_argument('--privacy-split',type=float,default=.5)
    p.add_argument('--h-factor',type=float,default=1.)
    p.add_argument('--reps',type=int,default=3)
    p.add_argument('--lmc-steps',type=int,default=4000); p.add_argument('--ridge',type=float,default=1e-3)
    p.add_argument('--lmc-step-factor',type=float,default=.02); p.add_argument('--projection-cycles',type=int,default=80)
    p.add_argument('--kappa-multiplier',type=float,default=2.)
    p.add_argument('--particles',type=int,default=5000); p.add_argument('--reference-particles',type=int,default=15000)
    p.add_argument('--score-grid',type=int,default=241); p.add_argument('--metric-grid',type=int,default=2001)
    p.add_argument('--tv-bins',type=int,default=70); p.add_argument('--xmax',type=float,default=7.)
    p.add_argument('--n-test',type=int,default=8000); p.add_argument('--seed',type=int,default=20261013)
    return p.parse_args()

if __name__=='__main__': run(parse())
