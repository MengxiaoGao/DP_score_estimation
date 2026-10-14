"""Single-record replacement pilot comparing private and non-private outputs.

All estimators use one frozen spline class (the alpha=2 class) so coefficient
vectors are directly comparable. Gibbs is never used.
"""
from common import *
import argparse,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from dataclasses import replace
import pandas as pd
from scipy.optimize import minimize,Bounds,LinearConstraint
from dp_score_sim.core import solve_constrained_erm

STAB=dict(n_values=[1000,4000,16000],repetitions=10,gamma=.5,total_alphas=[.5,2.,8.],reference_alpha=2.,typical_replacements=3,stress_replacements=2,reference_points=10000)

def mode(model,ridge):
    m=len(model.h);H=model.c*model.A+ridge*np.eye(m);h=model.c*model.h
    scale=max(float(np.trace(H))/max(m,1),1e-12);H=H/scale;h=h/scale
    fun=lambda z:float(.5*z@H@z+h@z);jac=lambda z:H@z+h
    cons=[]
    if m>1:
        D=np.zeros((m-1,m));i=np.arange(m-1);D[i,i]=-1;D[i,i+1]=1
        cons=[LinearConstraint(D,-model.diff_bound*np.ones(m-1),model.diff_bound*np.ones(m-1))]
    result=minimize(fun,np.zeros(m),jac=jac,method='SLSQP',bounds=Bounds(-model.coef_bound*np.ones(m),model.coef_bound*np.ones(m)),constraints=cons,options=dict(maxiter=2000,ftol=1e-10,disp=False))
    if not result.success:raise RuntimeError('private mode failed: '+result.message)
    return np.asarray(result.x),int(result.nit)

def effects(model,a,b,xref,prefix):
    delta=a-b;fa=d.score_from_theta(model,a,xref);fb=d.score_from_theta(model,b,xref);fd=fa-fb
    return {prefix+'_coef_rms':float(np.sqrt(np.mean(delta**2))),prefix+'_coef_max':float(np.max(abs(delta))),
        prefix+'_function_rms':float(np.sqrt(np.mean(fd**2))),prefix+'_function_max':float(np.max(abs(fd)))}

def objective(model,theta):return float(.5*theta@model.A@theta+model.h@theta)

def one(job):
    n,rep=job;out=ROOT/'stability';dest=out/'rows'/f'n{n}_rep{rep:02d}.json'
    if dest.exists():return json.loads(dest.read_text())
    c=cfg(STAB['gamma']);c.alpha=STAB['reference_alpha'];rng=row_seed(n,rep,c.gamma,31);x=draw(n,rng,c)
    base_cfg=d.SimConfig(gamma=c.gamma,target=c.target,amplitude=c.amplitude,K=c.K,lam=c.lam,mixture_mu=c.mixture_mu)
    alpha_score_ref=STAB['reference_alpha']*c.privacy_split
    base=d.build_model(x,c.gamma,alpha_score_ref,c.target,c.amplitude,c.K,c.lam,c.mixture_mu,base_cfg.bandwidth_factor,base_cfg.R_factor,base_cfg.coef_bound_factor)
    theta_np,info_np=solve_constrained_erm(base);xref=draw(STAB['reference_points'],row_seed(n,rep,c.gamma,32),c)
    replacements=[]
    for j in range(STAB['typical_replacements']):
        rr=row_seed(n,rep,c.gamma,40+j);idx=int(rr.integers(n));new=float(draw(1,rr,c)[0]);replacements.append((f'typical_{j}',idx,new,'typical'))
    replacements.extend([('stress_low_to_high',int(np.argmin(x)),float(base.R),'stress'),('stress_high_to_low',int(np.argmax(x)),float(-base.R),'stress')])
    noise={};arrays={};rows=[]
    for alpha in STAB['total_alphas']:
        m=replace(base,c=(alpha*c.privacy_split)/(4*base.Q))
        r1=row_seed(n,rep,c.gamma,100+int(alpha*10));r2=row_seed(n,rep,c.gamma,200+int(alpha*10))
        t1,_=d.projected_lmc_draw(m,r1,c.lmc_steps,c.ridge,c.lmc_step_factor,None,None,0,c.projection_cycles,c.projection_tol)
        t2,_=d.projected_lmc_draw(m,r2,c.lmc_steps,c.ridge,c.lmc_step_factor,None,None,0,c.projection_cycles,c.projection_tol)
        noise[alpha]=effects(m,t1,t2,xref,'noise_floor');arrays[f'noise_a{alpha}_1']=t1;arrays[f'noise_a{alpha}_2']=t2
    for j,(label,idx,new,kind) in enumerate(replacements):
        xp=x.copy();old=float(xp[idx]);xp[idx]=new
        neighbor=d.build_model(xp,c.gamma,alpha_score_ref,c.target,c.amplitude,c.K,c.lam,c.mixture_mu,base_cfg.bandwidth_factor,base_cfg.R_factor,base_cfg.coef_bound_factor)
        if not (np.array_equal(base.idx,neighbor.idx) and base.b==neighbor.b and base.R==neighbor.R):raise RuntimeError('Model class changed across neighbors')
        theta_np_p,info_np_p=solve_constrained_erm(neighbor);np_effect=effects(base,theta_np,theta_np_p,xref,'nonprivate')
        arrays[f'nonprivate_neighbor_{j}']=theta_np_p
        for alpha in STAB['total_alphas']:
            cval=(alpha*c.privacy_split)/(4*base.Q);m=replace(base,c=cval);mp=replace(neighbor,c=cval)
            seed=np.random.SeedSequence([CONFIG['seed'],n,rep,j,int(alpha*1000),77])
            td,ld=d.projected_lmc_draw(m,np.random.default_rng(seed),c.lmc_steps,c.ridge,c.lmc_step_factor,None,None,0,c.projection_cycles,c.projection_tol)
            tdp,ldp=d.projected_lmc_draw(mp,np.random.default_rng(seed),c.lmc_steps,c.ridge,c.lmc_step_factor,None,None,0,c.projection_cycles,c.projection_tol)
            md,nit=mode(m,c.ridge);mdp,nitp=mode(mp,c.ridge)
            row=dict(n=n,rep=rep,gamma=c.gamma,alpha=alpha,K=c.K,amplitude=c.amplitude,replacement=label,replacement_kind=kind,replaced_index=idx,old_value=old,new_value=new,
                basis_dim=len(base.idx),b=base.b,R=base.R,Q=base.Q,lmc_steps=c.lmc_steps,coupling='synchronous_gaussian_innovations',
                nonprivate_erm_nit=info_np['erm_nit'],nonprivate_neighbor_nit=info_np_p['erm_nit'],private_mode_nit=nit,private_neighbor_mode_nit=nitp,
                unnormalized_log_target_shift_at_private_draw=float(-cval*(objective(base,td)-objective(neighbor,td))),
                **np_effect,**effects(base,td,tdp,xref,'private_coupled'),**effects(base,md,mdp,xref,'private_mode'),**noise[alpha])
            rows.append(row);arrays[f'private_a{alpha}_neighbor_{j}_D']=td;arrays[f'private_a{alpha}_neighbor_{j}_Dp']=tdp
    arrays['data']=x;arrays['reference']=xref;arrays['nonprivate_base']=theta_np
    np.savez_compressed(out/'replicates'/f'n{n}_rep{rep:02d}.npz',**arrays);atomic_json(dest,rows);return rows

def aggregate():
    out=ROOT/'stability';allrows=[]
    for f in (out/'rows').glob('*.json'):allrows.extend(json.loads(f.read_text()))
    df=pd.DataFrame(allrows)
    if len(df):df.sort_values(['n','rep','replacement','alpha']).to_csv(out/'raw.csv',index=False)
    return df

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    out=ROOT/'stability'
    for x in ['rows','replicates','logs','figures','tables']:(out/x).mkdir(parents=True,exist_ok=True)
    atomic_json(out/'design.json',STAB)
    jobs=[(n,r) for r in range(STAB['repetitions']) for n in STAB['n_values']]
    if args.smoke:jobs=[(1000,0)]
    start=time.time()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
        fs={pool.submit(one,j):j for j in jobs}
        for k,f in enumerate(as_completed(fs),1):
            try:f.result()
            except BaseException:print('FAILED',fs[f],traceback.format_exc(),flush=True);raise
            df=aggregate();atomic_json(out/'logs/status.json',dict(complete=len(df)==30*5*3,base_jobs_saved=len(list((out/'rows').glob('*.json'))),rows_saved=len(df),planned_rows=450,elapsed_seconds=time.time()-start));print(f'stability {k}/{len(jobs)} rows={len(df)}',flush=True)
    aggregate()

if __name__=='__main__':main()
