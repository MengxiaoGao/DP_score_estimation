"""Shared target, model calibration, projected LMC kernel and file utilities."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, contextlib, hashlib, importlib, importlib.util, json, math, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

MODULE='private_score_experiments_v3'
NS=(1000,2000,4000,8000,16000)
ALPHAS=(.5,1.,2.,4.,8.)
GAMMAS=(.2,.4,.6,.8,1.)
PRIVACY_GAMMA=.5
SMOOTHNESS_ALPHA=2.
ALL_GAMMAS=tuple(sorted(set(GAMMAS+(PRIVACY_GAMMA,))))
AMPLITUDE=.25
K=40
SEED=1
N_TEST=10000
_MODULES={}

def coefficient(gamma):return AMPLITUDE
def cutoff(gamma):return K

def score(x,gamma):
    x=np.asarray(x,dtype=float);result=-x.copy()
    for k in range(cutoff(gamma)+1):
        result+=coefficient(gamma)*2**(-k*gamma)*np.sin(np.ldexp(x,k))
    return result

def potential(x,gamma):
    x=np.asarray(x,dtype=float);result=np.zeros_like(x)
    for k in range(cutoff(gamma)+1):
        result-=coefficient(gamma)*2**(-k*(1+gamma))*np.cos(np.ldexp(x,k))
    return result

def sample(n,gamma,rng):
    envelope=coefficient(gamma)*sum(2**(-k*(1+gamma)) for k in range(K+1))
    parts=[];need=n
    while need:
        x=rng.normal(size=max(1024,2*need))
        accepted=x[rng.uniform(size=x.size)<np.exp(potential(x,gamma)-envelope)]
        take=min(need,len(accepted))
        if take:parts.append(accepted[:take]);need-=take
    return np.concatenate(parts)

def rng(seed,*parts):return np.random.default_rng(np.random.SeedSequence([seed,*parts]))
def data(seed,rep,n,gamma,n_test):
    """Paired data design: alpha never enters the data RNG seeds.

    Every repetition has fresh training and independent test samples. Within
    (rep, gamma), all privacy budgets use exactly the same training and test
    arrays, including the ERM baseline. Training arrays are nested across n;
    the independent test array is shared across n as well. Reconstructing
    these arrays from seeds makes pairing independent of worker scheduling.
    """
    gid=int(round(gamma*1000))
    # Independent gamma families; matched data across alpha and nested n.
    return (sample(max(NS),gamma,rng(seed,rep,gid,0))[:n],
            sample(n_test,gamma,rng(seed,rep,gid,1)))

def release_rng(seed,rep,n,gamma,alpha):
    """Independent LMC stream for each repetition, n, gamma and privacy budget.

    Retains the previous seed mapping; training/test RNGs use separate keys.
    """
    return rng(seed,rep,int(round(gamma*1000)),n,int(round(alpha*1000)),2)


def load_module(name):
    if name in _MODULES:return _MODULES[name]
    if Path(name).suffix=='.py':
        path=Path(name).resolve();sys.path.insert(0,str(path.parent))
        key='_score_'+hashlib.sha256(str(path).encode()).hexdigest()[:12]
        spec=importlib.util.spec_from_file_location(key,path)
        if spec is None or spec.loader is None:raise ImportError(str(path))
        p=importlib.util.module_from_spec(spec);sys.modules[key]=p;spec.loader.exec_module(p)
    else:p=importlib.import_module(name)
    required=('projected_lmc_draw','ridge_calibration','build_model','score_from_theta','solve_constrained_erm','score_sup_bound',
              'holder_constant','_sample_truncated_normal','_sample_truncated_exponential',
              'spline_design','cutoff_chi')
    missing=[k for k in required if not callable(getattr(p,k,None))]
    if missing:raise AttributeError(f'Module missing functions: {missing}')
    draws=[p._sample_truncated_normal(100.,1.,0.,1.,rng(719,i)) for i in range(3)]
    if all(v==.5 for v in draws):raise RuntimeError('Replace the old extreme-tail midpoint fallback first.')
    if not all(np.isfinite(v) and 0<=v<=1 for v in draws):raise RuntimeError('Invalid extreme-tail conditional draws')
    _MODULES[name]=p;return p

def build_model(p,x,gamma,alpha):
    """Use the core's gamma-specific Holder and score-supremum bounds."""
    return p.build_model(x=x,gamma=gamma,alpha=alpha,target='multiscale',
                         amplitude=AMPLITUDE,K=K)


def feasible(theta,model,tolerance=1e-8):
    return (np.isfinite(theta).all() and np.max(np.abs(theta))<=model.coef_bound+tolerance
            and (len(theta)<2 or np.max(np.abs(np.diff(theta)))<=model.diff_bound+tolerance))

def erm(p,model):
    theta,info=p.solve_constrained_erm(model,maxiter=5000,ftol=1e-10)
    theta=np.asarray(theta,dtype=float)
    if not info.get('erm_success',False) or not feasible(theta,model):
        raise RuntimeError(f'Constrained ERM failed: {info}')
    return theta

def lmc(p,model,random,steps,initial=None,trace=None,label='',report_every=10000,
        ridge=None,step_factor=.01,projection_cycles=80,projection_tol=1e-10,
        ridge_rn=None,ridge_upsilon=None,privacy_beta=None):
    """Same projected-LMC kernel for convergence and production releases."""
    theta,info=p.projected_lmc_draw(model,random,n_steps=steps,ridge=ridge,
        step_factor=step_factor,projection_cycles=projection_cycles,
        projection_tol=projection_tol,initial=initial,trace=trace,
        label=label,report_every=report_every,ridge_rn=ridge_rn,
        ridge_upsilon=ridge_upsilon,privacy_beta=privacy_beta)
    if not feasible(theta,model):raise ArithmeticError('Infeasible final draw')
    return theta


def lmc_options(args):
    return dict(ridge=args.ridge,step_factor=args.step_factor,
                projection_cycles=args.projection_cycles,projection_tol=args.projection_tol,
                ridge_rn=getattr(args,'ridge_rn',None),
                ridge_upsilon=getattr(args,'ridge_upsilon',None),
                privacy_beta=getattr(args,'privacy_beta',None))


def resolved_ridge(p,model,options):
    return p.ridge_calibration(model,**{k:options.get(k) for k in
        ('ridge','ridge_rn','ridge_upsilon','privacy_beta')})


def risk(p,model,theta,test,gamma):
    value=float(np.mean((p.score_from_theta(model,theta,test)-score(test,gamma))**2))
    if not np.isfinite(value) or value<0:raise ArithmeticError('Invalid risk')
    return value

def risk_quadratic(p,model,test,gamma):
    psi,_=p.spline_design(test,model.idx,model.b)
    psi*=p.cutoff_chi(test,model.R)[:,None];s=score(test,gamma)
    return psi.T@psi/len(test),psi.T@s/len(test),float(np.mean(s*s))

def atomic_array(array,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:np.save(f,array,allow_pickle=False)
    tmp.replace(path)

def atomic_json(value,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def atomic_csv(frame,path):
    path=Path(path);tmp=path.with_suffix('.tmp');frame.to_csv(tmp,index=False);tmp.replace(path)

def atomic_coefficients(values,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**values)
    tmp.replace(path)


def load_coefficients(path):
    if not Path(path).exists():return {}
    with np.load(path,allow_pickle=False) as packed:
        result={key:packed[key].copy() for key in packed.files}
    if any(v.ndim!=1 or not len(v) or not np.isfinite(v).all() for v in result.values()):
        raise ValueError('Invalid saved coefficient vectors')
    return result


def coefficient_key(i,j,rep,is_erm=False):
    return f'{"erm" if is_erm else "private"}_{i}_{j}_{rep}'


def settings(args,study):
    p=load_module(args.module)
    code_files=['experiment_common.py','check_convergence.py','samples_and_privacy.py','samples_and_smoothness.py','coefficient_stability.py']
    hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in code_files}
    result=dict(study=study,sampler="projected_lmc",lmc=lmc_options(args),target='fixed-amplitude finite Weierstrass k=0..K',
        amplitude=AMPLITUDE,K=K,holder_calibration='core gamma-specific bound',
        holder_L_by_gamma={str(g):float(p.holder_constant(g,'multiscale',AMPLITUDE,K,.25)) for g in ALL_GAMMAS},
        calibration_gammas=ALL_GAMMAS,ns=NS,alphas=ALPHAS,gammas=GAMMAS,
        privacy_gamma=PRIVACY_GAMMA,smoothness_alpha=SMOOTHNESS_ALPHA,
        ridge_calibration=dict(version=1,
            formula='min(r_n, 2*log1p(Upsilon/2)/coefficient_radius**2)/coefficient_count',
            r_n='max(n**(-2*gamma/(1+2*gamma+2)), (n*alpha)**(-2*gamma/(1+gamma+1)))',
            radius='model.coef_bound; not the localisation radius model.R',
            upsilon='explicit, or privacy_beta/(1+exp(alpha))',
            default='fixed ridge 0.001 unless theorem inputs are supplied',
            fixed_override='--ridge may override the fixed default'),
        repetition_design=dict(training='fresh per repetition; paired across alpha; nested across n',
            test='independent of training; shared across alpha and n within repetition and gamma',
            lmc='independent streams by repetition, n, gamma and alpha',
            gamma='separate data streams for each target gamma',seed_mapping_version=1),
        seed=args.seed,n_test=args.n_test,module_sha256=hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest(),
        bundle_sha256=hashes)
    return json.loads(json.dumps(result))

def risk_settings(args,study):
    expected=settings(args,study)
    expected.update(steps=args.steps,reps=args.reps,retained_draws=1,initial_state='zero',
        coefficient_storage='final_coefficients_npz_v1',
        axes=['privacy','n','repetition'] if study=='privacy' else ['n','gamma','repetition'])
    return expected


def check_settings(out,expected,plot_only=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);path=out/'settings.json'
    if path.exists():
        stored=json.loads(path.read_text())
        ignored={'bundle_sha256','module_sha256'} if plot_only else set()
        same={k:v for k,v in stored.items() if k not in ignored}=={k:v for k,v in expected.items() if k not in ignored}
        if not same:raise ValueError('Settings/source changed: use a new output directory.')
    elif any(out.iterdir()):
        # A lock is present before settings are initialised.
        if any(p.name!='running.lock' for p in out.iterdir()):raise ValueError('Nonempty output directory without settings.json')
        atomic_json(expected,path)
    else:atomic_json(expected,path)

@contextlib.contextmanager
def output_lock(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);lock=out/'running.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError(f'{lock} exists. Another run may be active; remove only after confirming it stopped.')
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
    try:yield
    finally:lock.unlink(missing_ok=True)

def parallel_results(function,jobs,workers):
    if workers==1:
        for job in jobs:yield function(job)
    elif jobs:
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(function,job) for job in jobs]
            for future in as_completed(futures):yield future.result()

def base_parser(description,output):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--module',default=MODULE);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--seed',type=int,default=SEED);p.add_argument('--n-test',type=int,default=N_TEST)
    p.add_argument('--output-dir',default=str(Path(__file__).resolve().parent/output));p.add_argument('--plot-only',action='store_true')
    ridge=p.add_mutually_exclusive_group()
    ridge.add_argument('--ridge-upsilon',type=float,
        help='Upsilon for the theorem ridge calibration')
    ridge.add_argument('--privacy-beta',type=float,
        help='Select theorem ridge with Upsilon=beta/(1+exp(alpha)); default ridge is fixed at 0.001')
    ridge.add_argument('--ridge',type=float,default=None,
        help='Fixed gradient-ridge coefficient; default 0.001 unless theorem inputs are supplied')
    p.add_argument('--ridge-rn',type=float,
        help='Optional explicit r_n; default is the theorem statistical rate for each n/alpha/gamma')
    p.add_argument('--step-factor',type=float,default=.01,help='eta=step_factor/L; use smaller values to check discretisation bias')
    p.add_argument('--projection-cycles',type=int,default=80)
    p.add_argument('--projection-tol',type=float,default=1e-10)
    return p

def validate_args(p,args):
    if min(args.workers,args.n_test)<1 or args.seed<0:p.error('Positive workers/test size and nonnegative seed required')
    if args.ridge is not None and (not np.isfinite(args.ridge) or args.ridge<0):p.error('Fixed ridge must be finite and nonnegative')
    if args.ridge_rn is not None and (not np.isfinite(args.ridge_rn) or args.ridge_rn<=0):p.error('r_n must be finite and positive')
    if args.ridge_upsilon is not None and (not np.isfinite(args.ridge_upsilon) or args.ridge_upsilon<=0):p.error('Upsilon must be finite and positive')
    if args.privacy_beta is not None and (not np.isfinite(args.privacy_beta) or not 0<args.privacy_beta<1):p.error('privacy-beta must lie in (0,1)')
    if args.ridge is not None and args.ridge_rn is not None:p.error('Do not combine a fixed ridge with --ridge-rn')
    if args.ridge is None and args.ridge_upsilon is None and args.privacy_beta is None:
        core=load_module(args.module)
        if args.ridge_rn is None:
            args.ridge=core.DEFAULT_FIXED_RIDGE
        else:
            args.privacy_beta=core.DEFAULT_PRIVACY_BETA
    if not np.isfinite(args.step_factor) or not 0<args.step_factor<2:p.error('Step factor must lie in (0,2)')
    if args.projection_cycles<1 or not np.isfinite(args.projection_tol) or args.projection_tol<=0:p.error('Positive projection cycles/tolerance required')
    if Path(args.module).suffix=='.py':args.module=str(Path(args.module).resolve())


def split_chains(x):
    x = np.asarray(x, dtype=float)
    half = x.shape[1] // 2
    if half < 4:
        raise ValueError("Need at least eight post-burn draws per chain")
    return np.concatenate((x[:, :half], x[:, -half:]), axis=0)


def rank_normalise(x):
    ranks = rankdata(x.ravel(), method="average").reshape(x.shape)
    return ndtri((ranks - 3/8) / (x.size + 1/4))


def variance_parts(x):
    n = x.shape[1]
    within = float(np.var(x, axis=1, ddof=1).mean())
    variance = (n-1)/n * within + float(np.var(x.mean(axis=1), ddof=1))
    return within, variance


def basic_rhat(x):
    within, variance = variance_parts(x)
    if within == 0:
        return np.inf if variance > 0 else np.nan
    return float(np.sqrt(variance / within))


def effective_sample_size(x):
    """Split multichain ESS, using Geyer's positive/monotone pair sequence."""
    chains, n = x.shape
    within, variance = variance_parts(x)
    if within <= 0 or variance <= 0:
        return np.nan
    centred = x - x.mean(axis=1, keepdims=True)
    fft_size = 1 << (2*n - 1).bit_length()
    fft = np.fft.rfft(centred, n=fft_size, axis=1)
    autocov = np.fft.irfft(fft * fft.conj(), n=fft_size, axis=1)[:, :n] / n
    rho = 1 - (within - autocov.mean(axis=0)) / variance
    rho[0] = 1.
    total, previous = 0., np.inf
    for lag in range(0, n-1, 2):
        pair = float(rho[lag] + rho[lag+1])
        if pair <= 0:
            break
        pair = min(pair, previous)
        total += pair
        previous = pair
    tau = max(1., -1 + 2*total)
    return float(min(chains*n, chains*n/tau))


def diagnostics(x, with_mcse=False):
    split = split_chains(x)
    rank = rank_normalise(split)
    folded = rank_normalise(np.abs(split - np.median(split)))
    rank_rhat, folded_rhat = basic_rhat(rank), basic_rhat(folded)
    rhat = (np.nan if np.isnan(rank_rhat) or np.isnan(folded_rhat)
            else max(rank_rhat, folded_rhat))
    result = dict(rhat=rhat, ess=effective_sample_size(rank),
                  stuck=bool(np.any(np.ptp(x, axis=1) == 0)))
    if with_mcse:
        raw_ess = effective_sample_size(split)
        _, variance = variance_parts(split)
        result.update(mean=float(x.mean()),
                      mcse=float(np.sqrt(variance/raw_ess)) if raw_ess > 0 else np.nan)
    return result




# Main risk studies share the same helper and sampler as convergence checks.
def release(job):
    name,seed,steps,n_test,study,i,j,rep,is_erm,options,cached_theta=job
    p=load_module(name)
    if study=='privacy':
        n=NS[j];gamma=PRIVACY_GAMMA;alpha=None if is_erm else ALPHAS[i]
    else:
        n=NS[i];gamma=GAMMAS[j];alpha=None if is_erm else SMOOTHNESS_ALPHA
    x,test=data(seed,rep,n,gamma,n_test)
    model=build_model(p,x,gamma,alpha)
    if cached_theta is not None:
        theta=np.asarray(cached_theta,dtype=float)
        if theta.shape!=model.h.shape or not feasible(theta,model):raise ValueError('Invalid cached draw')
    elif is_erm:theta=erm(p,model)
    else:
        # All private experiments use exactly the same kernel as the checker.
        random=release_rng(seed,rep,n,gamma,alpha)
        theta=lmc(p,model,random,steps,label=f'{study}: n={n}, alpha={alpha}, gamma={gamma}, rep={rep+1}',**options)
    calibration=(dict(mode='unregularised_erm',ridge=0.,coefficients=len(theta),
                      coefficient_radius=model.coef_bound) if is_erm
                 else resolved_ridge(p,model,options))
    calibration.update(n=n,alpha=alpha,gamma=gamma)
    return i,j,rep,is_erm,risk(p,model,theta,test,gamma),theta,calibration

def plot(primary,erm,args,study):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not np.isfinite(primary).all() or (erm is not None and not np.isfinite(erm).all()):
        raise ValueError('Incomplete arrays: finish the experiment before plotting')
    def add(ax,values,label,**style):
        mean=values.mean(axis=-1)
        se=values.std(axis=-1,ddof=1)/np.sqrt(values.shape[-1]) if values.shape[-1]>1 else np.zeros_like(mean)
        ax.errorbar(NS,mean,yerr=np.vstack((np.minimum(se,.99*mean),se)),marker='o',capsize=3,label=label,**style)
    if study=='privacy':
        fig,ax=plt.subplots(figsize=(8,5))
        for i,alpha in enumerate(ALPHAS):add(ax,primary[i],rf'$\alpha={alpha:g}$')
        add(ax,primary[-1],'ERM',color='black',ls='--')
        axes=[ax]
        filename='n_privacy_risk.png'
    else:
        fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
        for j,gamma in enumerate(GAMMAS):
            add(axes[0],primary[:,j],rf'$\gamma={gamma:g}$')
            add(axes[1],erm[:,j],rf'$\gamma={gamma:g}$')
        
        filename='n_smoothness_risk.png'
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_yscale('log',base=2)
        ax.set_xticks(NS,[str(n) for n in NS]);ax.set_xlabel('Sample size n')
        ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Private mean squared score risk' if study=='smoothness' else 'Mean squared score risk')
    if study=='smoothness':axes[1].set_ylabel('ERM mean squared score risk')
    fig.tight_layout();fig.savefig(Path(args.output_dir)/filename,dpi=180);plt.close(fig)

def run_risk_study(study):
    p=base_parser(f'Fixed-amplitude Weierstrass {study} risk: one final coefficient draw per dataset.',f'results/{study}')
    p.add_argument('--steps','--sweeps',dest='steps',type=int,default=100000,help='LMC iterations; --sweeps is a legacy alias')
    p.add_argument('--reps',type=int,default=30)
    args=p.parse_args();validate_args(p,args)
    if min(args.steps,args.reps)<1:p.error('Positive steps/repetitions required')
    out=Path(args.output_dir)
    expected=risk_settings(args,study)
    with output_lock(out):
        check_settings(out,expected,plot_only=args.plot_only)
        shape=(6,len(NS),args.reps) if study=='privacy' else (len(NS),len(GAMMAS),args.reps)
        file=out/('privacy_risk.npy' if study=='privacy' else 'smoothness_risk.npy')
        primary=np.load(file,allow_pickle=False) if file.exists() else np.full(shape,np.nan)
        erm_file=out/'smoothness_erm_risk.npy'
        erms=(np.load(erm_file,allow_pickle=False) if erm_file.exists() else np.full(shape,np.nan)) if study=='smoothness' else None
        for a in [primary]+([erms] if erms is not None else []):
            if a.shape!=shape or np.isinf(a).any() or np.any(a<0):raise ValueError('Invalid stored risk array')
        coefficient_file=out/'coefficients.npz'
        coefficients=load_coefficients(coefficient_file)
        calibration_file=out/'ridge_calibration.json'
        calibrations=json.loads(calibration_file.read_text()) if calibration_file.exists() else {}
        jobs=[]
        for rep in range(args.reps):
            for i in range(shape[0]):
                for j in range(shape[1]):
                    is_erm=study=='privacy' and i==5
                    key=coefficient_key(i,j,rep,is_erm)
                    if np.isfinite(primary[i,j,rep]) and key not in coefficients:
                        raise ValueError('Saved risks have no matching coefficients; original draws cannot be recovered. Use a fresh output directory.')
                    if np.isnan(primary[i,j,rep]) or (not args.plot_only and key not in calibrations):
                        jobs.append((args.module,args.seed,args.steps,args.n_test,study,i,j,rep,is_erm,lmc_options(args),coefficients.get(key)))
                    if erms is not None:
                        key=coefficient_key(i,j,rep,True)
                        if np.isfinite(erms[i,j,rep]) and key not in coefficients:
                            raise ValueError('Saved ERM risk has no matching coefficients')
                        if np.isnan(erms[i,j,rep]) or (not args.plot_only and key not in calibrations):
                            jobs.append((args.module,args.seed,args.steps,args.n_test,study,i,j,rep,True,lmc_options(args),coefficients.get(key)))
        if args.plot_only and jobs:raise ValueError('Incomplete arrays')
        if not args.plot_only:
            atomic_array(primary,file)
            if erms is not None:atomic_array(erms,erm_file)
            print(f'{len(jobs)} pending cells; private releases use {args.steps:,} LMC steps; ERM uses none',flush=True)
            for count,result in enumerate(parallel_results(release,jobs,args.workers),1):
                i,j,rep,is_erm,value,theta,calibration=result
                key=coefficient_key(i,j,rep,is_erm)
                coefficients[key]=theta
                calibrations[key]=calibration
                atomic_json(calibrations,calibration_file)
                # Save the draw before its risk; a partially committed result can
                # regenerate risk from the cache without rerunning its chain.
                atomic_coefficients(coefficients,coefficient_file)
                if study=='smoothness' and is_erm:erms[i,j,rep]=value;atomic_array(erms,erm_file)
                else:primary[i,j,rep]=value;atomic_array(primary,file)
                print(f'{count}/{len(jobs)} pending cells completed: indices=({i},{j},{rep}), ERM={is_erm}, risk={value:.6g}',flush=True)
        plot(primary,erms,args,study)
        print(f'Saved {file}; shape={shape}',flush=True)
