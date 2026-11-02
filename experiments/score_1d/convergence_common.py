"""Shared four-chain diagnostics for the production experiment grids."""
from pathlib import Path
import tempfile
import experiment_common as c
import numpy as np

NAMES=tuple(f'Chain {i+1} (zero start)' for i in range(4))

def worst_index(values,minimum=False):
    values=np.asarray(values)
    mapped=np.where(np.isnan(values),-np.inf if minimum else np.inf,values)
    return int(np.argmin(mapped) if minimum else np.argmax(mapped))

def severity(row):
    value=float(row['max_coefficient_rhat'])
    return np.inf if not np.isfinite(value) or row['undefined_coefficient_rhat'] else value

def check_case(job):
    name,seed,n_test,case_id,lengths,burn_fraction,reference,rep,trace_dir,report,options,configuration=job
    n,alpha,gamma=configuration;p=c.load_module(name)
    x,test=c.data(seed,rep,n,gamma,n_test);model=c.build_model(p,x,gamma,alpha)
    calibration=c.resolved_ridge(p,model,options)
    starts=tuple(np.zeros_like(model.h) for _ in NAMES)
    longest=max(lengths);m=model.h.size;G,cross,constant=c.risk_quadratic(p,model,test,gamma)
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
            payload=dict(step_factor=np.array(options['step_factor']),case_id=np.array(case_id),n=np.array(n),alpha=np.array(alpha),gamma=np.array(gamma),
                reference=np.array(reference),burn=np.array(int(reference*burn_fraction)),
                severity=np.array(severity(ref_row)),steps=points+1,risk=risks[:,points],
                coefficients=np.stack([t[points][:,selected] for t in traces]),indices=np.array(selected),
                rhats=reference_rhats[selected],esses=reference_esses[selected])
            return rows,payload
        finally:
            for t in traces:t._mmap.close()

