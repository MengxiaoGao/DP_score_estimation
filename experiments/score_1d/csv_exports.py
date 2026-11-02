"""CSV exports and statistical summaries; no plotting dependencies."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

def score_rate(n,gamma,alpha):
    """Polynomial score-MSE reference used by the bundle's ridge calibration."""
    n=np.asarray(n,dtype=float)
    statistical=n**(-2*gamma/(2*gamma+3))
    if alpha is None:return statistical
    return np.maximum(statistical,(n*alpha)**(-2*gamma/(gamma+2)))


def summarise(values):
    if values.ndim!=2 or values.shape[1]<2 or not np.isfinite(values).all() or np.any(values<=0):
        raise ValueError('Need complete, strictly positive risks and at least two repetitions')
    mean=values.mean(axis=1)
    se=values.std(axis=1,ddof=1)/np.sqrt(values.shape[1])
    half=student_t.ppf(.975,values.shape[1]-1)*se
    return mean,se,mean-half,mean+half


def analyse_curve(values,ns,gamma,alpha,B,random):
    mean,se,low,high=summarise(values)
    x=np.log(np.asarray(ns,dtype=float));w=(x-x.mean())/np.sum((x-x.mean())**2)
    slope=float(w@np.log(mean))
    # One sampled repetition ID selects the whole vector across sample sizes.
    ids=random.integers(values.shape[1],size=(B,values.shape[1]))
    boot_means=values[:,ids].mean(axis=-1)
    boot_slopes=w@np.log(boot_means)
    ci=np.quantile(boot_slopes,[.025,.975])
    ref=score_rate(ns,gamma,alpha)
    # A visual scale fit, not a prediction of the theorem's unknown constant.
    scaled=ref*np.exp(np.mean(np.log(mean)-np.log(ref)))
    summary=pd.DataFrame(dict(n=ns,gamma=gamma,alpha=np.nan if alpha is None else alpha,
        repetitions=values.shape[1],mean=mean,se=se,ci95_low=low,ci95_high=high,
        polynomial_score_reference=ref,scaled_score_reference=scaled))
    slopes=dict(gamma=gamma,alpha=np.nan if alpha is None else alpha,
        slope=slope,ci95_low=ci[0],ci95_high=ci[1],
        sampling_reference_slope=-2*gamma/(2*gamma+3),
        composite_reference_grid_slope=float(w@np.log(ref)),
        bootstrap_repetitions=B,bootstrap_unit='repetition vector across n')
    return summary,slopes


def load_study(folder,study):
    settings=json.loads((folder/'settings.json').read_text())
    if settings['study']!=study:raise ValueError(f'{folder}: wrong study')
    ns=settings['ns'];alphas=settings['alphas'];gammas=settings['gammas']
    if len(ns)<2 or len(set(ns))!=len(ns) or list(ns)!=sorted(ns):raise ValueError('Need distinct ordered sample sizes')
    name='privacy_risk.npy' if study=='privacy' else 'smoothness_risk.npy'
    values=np.load(folder/name,allow_pickle=False)
    reps=settings['reps']
    shape=(len(alphas)+1,len(ns),reps) if study=='privacy' else (len(ns),len(gammas),reps)
    if values.shape!=shape:raise ValueError(f'{folder}: unexpected array shape')
    curves=[]
    if study=='privacy':
        for i,a in enumerate(alphas):curves.append((rf'$\alpha={a:g}$',settings['privacy_gamma'],a,values[i],False))
        curves.append(('ERM',settings['privacy_gamma'],None,values[-1],True))
    else:
        for j,g in enumerate(gammas):curves.append((rf'$\gamma={g:g}$',g,settings['smoothness_alpha'],values[:,j],False))
    return settings,ns,curves


def export_study(folder,study,B=2000,seed=1):
    folder=Path(folder)
    settings,ns,curves=load_study(folder,study)
    summaries=[];slopes=[];raw=[]
    for k,(label,gamma,alpha,values,is_erm) in enumerate(curves):
        table,slope=analyse_curve(values,ns,gamma,alpha,B,np.random.default_rng(seed+k))
        table['curve']=label;slope['curve']=label
        summaries.append(table);slopes.append(slope)
        for j,n in enumerate(ns):
            for rep,value in enumerate(values[j]):
                raw.append(dict(n=n,gamma=gamma,alpha=alpha,rep=rep,score_risk=value,is_erm=is_erm,curve=label))
    tables=folder/'tables';tables.mkdir(exist_ok=True)
    pd.concat(summaries,ignore_index=True).to_csv(tables/'score_summary.csv',index=False)
    pd.DataFrame(slopes).to_csv(tables/'score_slopes.csv',index=False)
    # Preserve the additional smoothness ERM risks as CSV as well.
    if study=='smoothness' and (folder/'smoothness_erm_risk.npy').exists():
        erms=np.load(folder/'smoothness_erm_risk.npy',allow_pickle=False)
        expected=(len(ns),len(settings['gammas']),settings['reps'])
        if erms.shape!=expected:raise ValueError('Invalid smoothness ERM shape')
        erm_tables=[]
        for j,gamma in enumerate(settings['gammas']):
            mean,se,low,high=summarise(erms[:,j])
            erm_tables.append(pd.DataFrame(dict(n=ns,gamma=gamma,mean=mean,se=se,ci95_low=low,ci95_high=high,repetitions=settings['reps'])))
            for i,n in enumerate(ns):
                for rep,value in enumerate(erms[i,j]):raw.append(dict(n=n,gamma=gamma,alpha=None,rep=rep,score_risk=value,is_erm=True,curve='ERM'))
        pd.concat(erm_tables,ignore_index=True).to_csv(tables/'score_erm_summary.csv',index=False)
    pd.DataFrame(raw).to_csv(tables/'score_raw.csv',index=False)
    calibration=folder/'ridge_calibration.json'
    if calibration.exists():
        entries=json.loads(calibration.read_text())
        pd.DataFrame([dict(cell=key,**value) for key,value in entries.items()]).to_csv(tables/'ridge_calibration.csv',index=False)
    (tables/'summary_settings.json').write_text(json.dumps(dict(bootstrap_repetitions=B,seed=seed,
        mean_interval=f"pointwise Student t, df={settings['reps']-1}",
        slope_interval='paired percentile bootstrap; repetition vectors across n'),indent=2))
    print(f'Saved CSV data and summaries in {tables}')


def export_trace(cache,out):
    """Export the selected chain traces to long CSV for a standalone plotter."""
    with np.load(cache,allow_pickle=False) as p:data={k:p[k] for k in p.files}
    rows=[]
    for chain in range(data['risk'].shape[0]):
        for j,step in enumerate(data['steps']):
            row=dict(chain=chain+1,steps=int(step),score_risk=float(data['risk'][chain,j]),
                n=int(data['n']),alpha=float(data['alpha']),gamma=float(data['gamma']),
                step_factor=float(data['step_factor']),burn=int(data['burn']))
            for i,index in enumerate(data['indices']):row[f'coefficient_{int(index)}']=float(data['coefficients'][chain,j,i])
            rows.append(row)
    pd.DataFrame(rows).to_csv(Path(out)/'trace.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=Path(__file__).resolve().parent/'results')
    parser.add_argument('--study',choices=['both','privacy','smoothness'],default='both')
    parser.add_argument('--bootstrap-repetitions',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=1)
    args=parser.parse_args()
    if args.bootstrap_repetitions<2 or args.seed<0:parser.error('Invalid bootstrap count or seed')
    for study in ('privacy','smoothness'):
        folder=args.results_dir/study
        if args.study in ('both',study) and (folder/'settings.json').exists():export_study(folder,study,args.bootstrap_repetitions,args.seed)

if __name__=='__main__':main()
