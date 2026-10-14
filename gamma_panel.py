"""Formal gamma panel: alpha=2, fixed K/amplitude, projected LMC, R=30."""
from common import *
import argparse,traceback,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from dataclasses import asdict
import pandas as pd
from scipy.stats import wasserstein_distance

def tag(n,rep,gamma):return f'n{n}_rep{rep:02d}_gamma{gamma:.1f}'.replace('.','p')
def ahash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def one(job):
    n,rep,gamma=job;c=cfg(gamma);out=ROOT/'gamma_panel';name=tag(n,rep,gamma);dest=out/'rows'/f'{name}.json'
    if dest.exists():return json.loads(dest.read_text())
    total=time.perf_counter();times={}
    t=time.perf_counter();x=draw(n,row_seed(n,rep,gamma,1),c);times['data_seconds']=time.perf_counter()-t
    base=d.SimConfig(gamma=gamma,target=c.target,amplitude=c.amplitude,K=c.K,lam=c.lam,mixture_mu=c.mixture_mu)
    t=time.perf_counter();model=d.build_model(x,gamma,c.alpha*c.privacy_split,c.target,c.amplitude,c.K,c.lam,c.mixture_mu,base.bandwidth_factor,base.R_factor,base.coef_bound_factor);times['model_seconds']=time.perf_counter()-t
    t=time.perf_counter();theta,info=d.projected_lmc_draw(model,row_seed(n,rep,gamma,2),c.lmc_steps,c.ridge,c.lmc_step_factor,None,None,0,c.projection_cycles,c.projection_tol);times['lmc_seconds']=time.perf_counter()-t
    if info.max_projection_violation>1e-8:raise RuntimeError('Projection violation')
    t=time.perf_counter();hist=d.histogram_masses(x,d.make_uniform_edges(model.R,c.h_factor*model.b),c.alpha*(1-c.privacy_split),row_seed(n,rep,gamma,3),True)
    slopes=d.score_from_theta(model,theta,hist.centers);cells=d.affine_cells_from_hist(hist,slopes);times['reconstruction_seconds']=time.perf_counter()-t
    t=time.perf_counter();density_tv=rp.density_tv(cells,c,192);density_coarse=rp.density_tv(cells,c,96)
    grid=np.linspace(-c.xmax,c.xmax,c.metric_grid);truth=target(grid,c);pt,_=pn.ou_affine_density_score(grid,cells,1/n)
    ou_tv=d.grid_tv(truth,pt,grid);ou_coarse=d.grid_tv(truth[::2],pt[::2],grid[::2]);mass=float(np.trapezoid(pt,grid))
    mp=rp.bin_masses(lambda z:pn.ou_affine_density_score(z,cells,1/n)[0],c.tv_bins,c.xmax,128)
    mt=rp.bin_masses(lambda z:target(z,c),c.tv_bins,c.xmax,128);times['density_metrics_seconds']=time.perf_counter()-t
    y,N,kappa,reverse_seconds=reverse(cells,n,gamma,c.alpha,row_seed(n,rep,gamma,4),c)
    t=time.perf_counter();refou=d.sample_ptau(cells,1/n,c.particles,row_seed(n,rep,gamma,5));reftrue=draw(c.reference_particles,row_seed(n,rep,gamma,6),c)
    row=dict(n=n,rep=rep,gamma=gamma,alpha=c.alpha,K=c.K,amplitude=c.amplitude,dataset_coupling='independent_across_gamma',
        train_hash=ahash(x),lmc_steps=c.lmc_steps,particles=c.particles,b=model.b,b_raw=model.b_raw,b_capped=bool(model.b_capped),R=model.R,basis_dim=len(model.idx),holder_L=model.holder_L,Q=model.Q,
        density_tv=density_tv,ou_tv=ou_tv,generated_binned_tv=rp.btv(y,mt,c),generated_wasserstein=float(wasserstein_distance(y,reftrue)),
        reverse_to_ou_binned_tv=rp.btv(y,mp,c),iid_ou_binned_tv=rp.btv(refou,mp,c),delta=d.delta_theory(n,c.alpha,gamma),kappa=kappa,N=N,
        lmc_step_size=info.step_size,lmc_projection_violation=info.max_projection_violation,ou_mass=mass,
        density_quad_delta=abs(density_tv-density_coarse),ou_grid_delta=abs(ou_tv-ou_coarse),reverse_seconds=reverse_seconds,total_seconds=time.perf_counter()-total,**times)
    np.savez_compressed(out/'replicates'/f'{name}.npz',theta=theta,generated=y,**{'cells_'+k:v for k,v in asdict(cells).items()})
    atomic_json(dest,row);return row

def aggregate():
    out=ROOT/'gamma_panel';rows=[json.loads(f.read_text()) for f in (out/'rows').glob('*.json')]
    df=pd.DataFrame(rows)
    if len(df):df.sort_values(['n','gamma','rep']).to_csv(out/'raw.csv',index=False)
    return df

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    out=ROOT/'gamma_panel'
    for x in ['rows','replicates','logs','figures','tables']:(out/x).mkdir(parents=True,exist_ok=True)
    jobs=[(n,r,g) for r in range(CONFIG['repetitions']) for n in CONFIG['n_values'] for g in CONFIG['gammas']]
    if args.smoke:jobs=[(1000,0,.2),(16000,0,1.)]
    start=time.time();done=0
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
        fs={pool.submit(one,j):j for j in jobs}
        for f in as_completed(fs):
            try:f.result()
            except BaseException:
                print('FAILED',fs[f],traceback.format_exc(),flush=True);raise
            done+=1;df=aggregate();atomic_json(out/'logs/status.json',dict(complete=len(df)==750,saved=len(df),processed_this_run=done,planned=750,elapsed_seconds=time.time()-start))
            print(f'gamma {done}/{len(jobs)} total_saved={len(df)}',flush=True)
    aggregate()

if __name__=='__main__':main()
