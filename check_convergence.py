"""Thirteen representative settings; four chains; LMC iteration-prefix diagnostics."""
from pathlib import Path
import tempfile
import experiment_common as c
import numpy as np
import pandas as pd

CASES=tuple((16000,a,g) for a in (.5,8.) for g in (.2,.6,.8))+((1000,.5,.8),(1000,8.,.8),(8000,2.,.4),(16000,.5,1.),(16000,8.,1.),(1000,.5,1.),(1000,8.,1.))
LENGTHS=(20000,50000,100000,150000,200000)
NAMES=('zero','ERM','positive constant','negative constant')

def worst_index(values,minimum=False):
    values=np.asarray(values)
    mapped=np.where(np.isnan(values),-np.inf if minimum else np.inf,values)
    return int(np.argmin(mapped) if minimum else np.argmax(mapped))

def severity(row):
    value=float(row['max_coefficient_rhat'])
    return np.inf if not np.isfinite(value) or row['undefined_coefficient_rhat'] else value

def check_case(job):
    name,seed,n_test,case_id,lengths,burn_fraction,reference,rep,trace_dir,report,options=job
    n,alpha,gamma=CASES[case_id];p=c.load_module(name)
    x,test=c.data(seed,rep,n,gamma,n_test);model=c.build_model(p,x,gamma,alpha)
    calibration=c.resolved_ridge(p,model,options)
    start=c.erm(p,model)
    scale=max(1.,np.max(np.abs(start))/model.coef_bound,np.max(np.abs(np.diff(start)))/model.diff_bound)
    start=start/scale*(1-1e-12)
    starts=(np.zeros_like(start),start,np.full_like(start,.8*model.coef_bound),np.full_like(start,-.8*model.coef_bound))
    longest=max(lengths);m=len(start);G,cross,constant=c.risk_quadratic(p,model,test,gamma)
    print(f'Case {case_id}: n={n}, alpha={alpha}, gamma={gamma}; {m} coefficients',flush=True)
    with tempfile.TemporaryDirectory(prefix='score-traces-',dir=trace_dir) as tmp:
        traces=[]
        try:
            for chain,initial in enumerate(starts):
                trace=np.memmap(Path(tmp)/f'chain_{chain}.dat',mode='w+',dtype='float64',shape=(longest,m))
                traces.append(trace)
                random=c.rng(seed,rep,int(round(gamma*1000)),n,int(round(alpha*1000)),3,chain)
                c.lmc(p,model,random,longest,initial,trace,f'case {case_id}/{NAMES[chain]}',report,**options)
                trace.flush()
            risks=np.empty((4,longest))
            for i,t in enumerate(traces):
                for begin in range(0,longest,1000):
                    end=min(begin+1000,longest);theta=t[begin:end]
                    risks[i,begin:end]=np.einsum('ij,jk,ik->i',theta,G,theta,optimize=True)-2*theta@cross+constant
            if not np.isfinite(risks).all() or risks.min() < -1e-8:raise ArithmeticError('Invalid quadratic risk')
            risks=np.maximum(risks,0.)
            rows=[];reference_rhats=reference_esses=None
            for length in lengths:
                burn=int(length*burn_fraction)
                ds=[c.diagnostics(np.stack([t[burn:length,j] for t in traces])) for j in range(m)]
                rhats=np.array([d['rhat'] for d in ds]);esses=np.array([d['ess'] for d in ds])
                rd=c.diagnostics(risks[:,burn:length],with_mcse=True)
                rows.append(dict(case_id=case_id,n=n,alpha=alpha,gamma=gamma,dataset_rep=rep,
                    steps=length,burn=burn,chains=4,coefficients=m,K=c.cutoff(gamma),b=model.b,
                    max_coefficient_rhat=float(np.max(rhats)),min_coefficient_ess=float(np.min(esses)),
                    undefined_coefficient_rhat=int(np.isnan(rhats).sum()),
                    undefined_coefficient_ess=int(np.isnan(esses).sum()),stuck_coefficients=sum(d['stuck'] for d in ds),
                    risk_rhat=rd['rhat'],risk_ess=rd['ess'],mean_risk=rd['mean'],risk_mcse=rd['mcse'],
                    seed=seed,n_test=n_test,holder_L=model.holder_L,ridge=calibration['ridge'],
                    ridge_mode=calibration['mode'],ridge_r_n=calibration['r_n'],
                    ridge_upsilon=calibration['upsilon'],ridge_tv_cap=calibration['tv_cap'],
                    coefficient_radius=calibration['coefficient_radius'],
                    privacy_beta=calibration['privacy_beta'],
                    bar_lambda=calibration['ridge']+n*alpha/(4*model.Q),
                    step_factor=options["step_factor"],
                    step_size=options["step_factor"]/(model.c*float(np.linalg.eigvalsh(model.A).max())+calibration['ridge'])))
                if length==reference:reference_rhats,reference_esses=rhats,esses
            selected=list(dict.fromkeys((worst_index(reference_rhats),worst_index(reference_esses,minimum=True))))
            stride=max(1,int(np.ceil(longest/5000)))
            points=np.unique(np.r_[np.arange(0,longest,stride),longest-1])
            ref_row=next(row for row in rows if row['steps']==reference)
            payload=dict(case_id=np.array(case_id),n=np.array(n),alpha=np.array(alpha),gamma=np.array(gamma),
                reference=np.array(reference),burn=np.array(int(reference*burn_fraction)),
                severity=np.array(severity(ref_row)),steps=points+1,risk=risks[:,points],
                coefficients=np.stack([t[points][:,selected] for t in traces]),indices=np.array(selected),
                rhats=reference_rhats[selected],esses=reference_esses[selected])
            return rows,payload
        finally:
            for t in traces:t._mmap.close()

def save_trace_data(payload,path):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**payload)
    tmp.replace(path)

def trace_plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(out/'trace_data.npz',allow_pickle=False) as packed:data={k:packed[k] for k in packed.files}
    indices=data['indices'];fig,axes=plt.subplots(1+len(indices),1,figsize=(10,3*(1+len(indices))),sharex=True,squeeze=False)
    axes=axes[:,0]
    for chain,name in enumerate(NAMES):
        axes[0].plot(data['steps'],data['risk'][chain],lw=.6,alpha=.7,label=name)
        for k,ax in enumerate(axes[1:]):ax.plot(data['steps'],data['coefficients'][chain,:,k],lw=.6,alpha=.7)
    axes[0].set_ylabel('Squared score risk');axes[0].legend(ncol=2)
    for k,ax in enumerate(axes[1:]):
        ax.set_ylabel(f'Coefficient {indices[k]}')
    for ax in axes:ax.axvline(int(data['burn']),color='black',ls='--',lw=1);ax.grid(alpha=.2)
    axes[-1].set_xlabel('LMC steps per chain; dashed line = reference-prefix burn-in')
    fig.tight_layout();fig.savefig(out/'trace.png',dpi=170);plt.close(fig)

def convergence_plot(frame,out,reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    refs=frame[frame.steps==reference].copy()
    refs['severity']=[severity(r) for _,r in refs.iterrows()]
    def worst(group):return int(group.sort_values(['severity','case_id'],ascending=[False,True]).iloc[0].case_id)
    chosen=[]
    if (refs.n==16000).any():chosen.append(worst(refs[refs.n==16000]))
    if (refs.n==1000).any():chosen.append(worst(refs[refs.n==1000]))
    if (refs.case_id==8).any():chosen.append(8)
    if not chosen:chosen=[worst(refs)]
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    colors=('#0072B2','#D55E00','#009E73');markers=('o','s','^')
    for case_id,color,marker in zip(chosen,colors,markers):
        s=frame[frame.case_id==case_id].sort_values('steps');last=s.iloc[-1]
        kw=dict(color=color,marker=marker,lw=1.6,label=rf'$n={int(last.n):,},\alpha={last.alpha:g},\gamma={last.gamma:g}$')
        axes[0,0].plot(s.steps,s.max_coefficient_rhat,**kw)
        axes[0,1].plot(s.steps,s.min_coefficient_ess,**kw)
        axes[1,0].plot(s.steps,s.risk_rhat,**kw)
        axes[1,1].errorbar(s.steps,s.mean_risk/last.mean_risk,yerr=2*s.risk_mcse/last.mean_risk,capsize=3,**kw)
    axes[0,0].set_ylabel('Maximum coefficient R-hat')
    axes[0,1].set_ylabel('Minimum coefficient bulk ESS');axes[0,1].set_yscale('log',base=2)
    axes[1,0].set_ylabel('Risk R-hat')
    axes[1,1].set_ylabel(f'Mean risk / {int(frame.steps.max()):,}-step mean')
    for ax in (axes[0,0],axes[1,0]):ax.axhline(1.01,color='grey',ls='--',lw=1)
    axes[0,1].axhline(400,color='grey',ls='--',lw=1);axes[1,1].axhline(1,color='grey',ls='--',lw=1)
    lengths=sorted(frame.steps.unique())
    for ax in axes.flat:
        ax.set_xscale('log',base=2);ax.set_xticks(lengths,[f'{int(t):,}' for t in lengths]);ax.set_xlabel('LMC steps per chain');ax.grid(alpha=.2)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.98),ncol=len(chosen),frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(out/'convergence.png',dpi=180);plt.close(fig)

def main():
    p=c.base_parser(__doc__,'results/convergence')
    p.add_argument('--lengths',nargs='+',type=int,default=LENGTHS)
    p.add_argument('--burn-fraction',type=float,default=.1)
    p.add_argument('--reference-steps','--reference-sweeps',dest='reference_steps',type=int,help='Default 200000 if available, otherwise longest prefix')
    p.add_argument('--dataset-rep',type=int,default=0)
    p.add_argument('--case-ids',nargs='+',type=int,default=list(range(len(CASES))))
    p.add_argument('--list-cases',action='store_true')
    p.add_argument('--trace-dir',help='Existing fast scratch directory for temporary full coefficient traces')
    p.add_argument('--report-every',type=int,default=10000)
    args=p.parse_args();c.validate_args(p,args)
    if args.list_cases:
        for i,case in enumerate(CASES):print(i,dict(zip(('n','alpha','gamma'),case)))
        return
    lengths=tuple(sorted(set(args.lengths)));ids=tuple(sorted(set(args.case_ids)))
    reference=args.reference_steps or (200000 if 200000 in lengths else max(lengths))
    if reference not in lengths:p.error('Reference step count must be a requested prefix')
    if not ids or any(i<0 or i>=len(CASES) for i in ids):p.error('Invalid case IDs')
    if not 0<=args.burn_fraction<1 or min(lengths)<20 or any(t-int(t*args.burn_fraction)<8 for t in lengths):p.error('Insufficient post-burn draws')
    if args.dataset_rep<0 or args.report_every<0:p.error('Invalid dataset/progress setting')
    if args.trace_dir and not Path(args.trace_dir).is_dir():p.error('Trace directory must exist')
    out=Path(args.output_dir);expected=c.settings(args,'convergence')
    expected.update(cases=[list(CASES[i]) for i in ids],case_ids=list(ids),lengths=list(lengths),
                    reference_steps=reference,burn_fraction=args.burn_fraction,dataset_rep=args.dataset_rep,
                    starts=list(NAMES),chains=4)
    with c.output_lock(out):
        c.check_settings(out,expected,plot_only=args.plot_only);file=out/'convergence.csv';cache=out/'trace_data.npz'
        frame=pd.read_csv(file) if file.exists() else pd.DataFrame()
        done=set()
        if not frame.empty:
            if frame.duplicated(['case_id','steps']).any():raise ValueError('Duplicate stored diagnostics')
            for case_id,part in frame.groupby('case_id'):
                if int(case_id) not in ids or set(part.steps)!=set(lengths):raise ValueError('Invalid stored case/checkpoints')
                done.add(int(case_id))
            if not cache.exists():raise ValueError('Missing trace cache: use a fresh output directory')
        jobs=[(args.module,args.seed,args.n_test,i,lengths,args.burn_fraction,reference,args.dataset_rep,args.trace_dir,args.report_every,c.lmc_options(args)) for i in ids if i not in done]
        if args.plot_only and jobs:raise ValueError('Convergence study incomplete')
        best=-np.inf
        if cache.exists():
            with np.load(cache,allow_pickle=False) as saved:best=float(saved['severity'])
        if not args.plot_only:
            for rows,payload in c.parallel_results(check_case,jobs,args.workers):
                # Retain only the globally worst setting's display-subsampled trace.
                if float(payload['severity'])>best:
                    best=float(payload['severity']);save_trace_data(payload,cache)
                frame=pd.concat((frame,pd.DataFrame(rows)),ignore_index=True).sort_values(['case_id','steps'])
                c.atomic_csv(frame,file)
                print(f'Completed case {rows[0]["case_id"]}; {len(frame)}/{len(ids)*len(lengths)} rows saved',flush=True)
        convergence_plot(frame,out,reference);trace_plot(out)
        print(f'Saved {file}, convergence.png and trace.png',flush=True)
if __name__=='__main__':main()
