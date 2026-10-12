"""Shared target, model calibration, coordinate Gibbs kernel and file utilities."""
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
GAMMAS=(.2,.4,.6,.8)
PRIVACY_GAMMA=.5
SMOOTHNESS_ALPHA=2.
ALL_GAMMAS=tuple(sorted(set(GAMMAS+(PRIVACY_GAMMA,))))
AMPLITUDE=.5
SCORE_TOL=1e-3
SEED=1
N_TEST=10000
_MODULES={}

def coefficient(gamma):return AMPLITUDE*np.expm1(gamma*np.log(2.))
def cutoff(gamma):return int(np.ceil(np.log2(AMPLITUDE/SCORE_TOL)/gamma))
def holder_bound(gamma):
    return 1+coefficient(gamma)*(2**(1-gamma)/(2**(1-gamma)-1)+2/(1-2**(-gamma)))
COMMON_HOLDER=max(holder_bound(g) for g in ALL_GAMMAS)

def score(x,gamma):
    x=np.asarray(x,dtype=float);result=-x.copy()
    for k in range(1,cutoff(gamma)+1):
        result+=coefficient(gamma)*2**(-k*gamma)*np.sin(np.ldexp(x,k))
    return result

def potential(x,gamma):
    x=np.asarray(x,dtype=float);result=np.zeros_like(x)
    for k in range(1,cutoff(gamma)+1):
        result-=coefficient(gamma)*2**(-k*(1+gamma))*np.cos(np.ldexp(x,k))
    return result

def sample(n,gamma,rng):
    envelope=coefficient(gamma)/(2**(1+gamma)-1)
    parts=[];need=n
    while need:
        x=rng.normal(size=max(1024,2*need))
        accepted=x[rng.uniform(size=x.size)<np.exp(potential(x,gamma)-envelope)]
        take=min(need,len(accepted))
        if take:parts.append(accepted[:take]);need-=take
    return np.concatenate(parts)

def rng(seed,*parts):return np.random.default_rng(np.random.SeedSequence([seed,*parts]))
def data(seed,rep,n,gamma,n_test):
    gid=int(round(gamma*1000))
    # Independent gamma families; matched data across alpha and nested n.
    return (sample(max(NS),gamma,rng(seed,rep,gid,0))[:n],
            sample(n_test,gamma,rng(seed,rep,gid,1)))

def load_module(name):
    if name in _MODULES:return _MODULES[name]
    if Path(name).suffix=='.py':
        path=Path(name).resolve();sys.path.insert(0,str(path.parent))
        key='_score_'+hashlib.sha256(str(path).encode()).hexdigest()[:12]
        spec=importlib.util.spec_from_file_location(key,path)
        if spec is None or spec.loader is None:raise ImportError(str(path))
        p=importlib.util.module_from_spec(spec);sys.modules[key]=p;spec.loader.exec_module(p)
    else:p=importlib.import_module(name)
    required=('build_model','score_from_theta','solve_constrained_erm','score_sup_bound',
              'holder_constant','_sample_truncated_normal','_sample_truncated_exponential',
              'spline_design','cutoff_chi')
    missing=[k for k in required if not callable(getattr(p,k,None))]
    if missing:raise AttributeError(f'Module missing functions: {missing}')
    draws=[p._sample_truncated_normal(100.,1.,0.,1.,rng(719,i)) for i in range(3)]
    if all(v==.5 for v in draws):raise RuntimeError('Replace the old extreme-tail midpoint fallback first.')
    if not all(np.isfinite(v) and 0<=v<=1 for v in draws):raise RuntimeError('Invalid extreme-tail conditional draws')
    _MODULES[name]=p;return p

def build_model(p,x,gamma,alpha):
    old=p.score_sup_bound,p.holder_constant
    # Only calibration helpers are replaced temporarily. The builder uses data,
    # not the reference module's target score. Each worker is a separate process.
    p.score_sup_bound=lambda xmax,*args,**kwargs:xmax+AMPLITUDE
    p.holder_constant=lambda *args,**kwargs:COMMON_HOLDER
    try:
        return p.build_model(x=x,gamma=gamma,alpha=alpha,target='multiscale',
                             amplitude=float(coefficient(gamma)),K=cutoff(gamma))
    finally:p.score_sup_bound,p.holder_constant=old

def feasible(theta,model,tolerance=1e-8):
    return (np.isfinite(theta).all() and np.max(np.abs(theta))<=model.coef_bound+tolerance
            and (len(theta)<2 or np.max(np.abs(np.diff(theta)))<=model.diff_bound+tolerance))

def erm(p,model):
    theta,info=p.solve_constrained_erm(model,maxiter=5000,ftol=1e-10)
    theta=np.asarray(theta,dtype=float)
    if not info.get('erm_success',False) or not feasible(theta,model):
        raise RuntimeError(f'Constrained ERM failed: {info}')
    return theta

def gibbs(p,model,random,sweeps,initial=None,trace=None,label='',report_every=10000):
    """Same sweep/conditionals for convergence and production releases.

    Import the scalar conditional samplers; retain the existing 1e-12 curvature
    threshold. Empty intervals raise, rather than fabricating a midpoint.
    trace, if supplied, receives every full-sweep vector (disk memmap permitted).
    """
    theta=np.zeros(len(model.h)) if initial is None else np.asarray(initial,dtype=float).copy()
    if not feasible(theta,model,tolerance=1e-12):raise ValueError('Infeasible initial state')
    A,h,c=model.A,model.h,model.c;B,D=model.coef_bound,model.diff_bound
    Atheta=A@theta
    quad=c*np.diag(A)
    for sweep in range(sweeps):
        for j in range(len(theta)):
            lo,hi=-B,B
            if j:lo,hi=max(lo,theta[j-1]-D),min(hi,theta[j-1]+D)
            if j+1<len(theta):lo,hi=max(lo,theta[j+1]-D),min(hi,theta[j+1]+D)
            if lo>hi:raise ArithmeticError(f'Empty feasible interval: {label}, sweep={sweep}, j={j}')
            old=theta[j];lin=c*(h[j]+Atheta[j]-A[j,j]*old)
            if lo==hi:new=lo
            elif quad[j]>1e-12:
                new=p._sample_truncated_normal(-lin/quad[j],1/math.sqrt(quad[j]),lo,hi,random)
            else:new=p._sample_truncated_exponential(lo,hi,lin,random)
            if not np.isfinite(new) or new<lo-1e-12 or new>hi+1e-12:raise ArithmeticError('Invalid conditional draw')
            theta[j]=new
            if new!=old:Atheta+=(new-old)*A[:,j]
        if trace is not None:trace[sweep]=theta
        if report_every and (sweep+1)%report_every==0:
            print(f'{label}: {sweep+1:,}/{sweeps:,} sweeps',flush=True)
    if not feasible(theta,model):raise ArithmeticError('Infeasible final draw')
    return theta.copy()

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

def settings(args,study):
    p=load_module(args.module)
    code_files=['experiment_common.py','check_convergence.py','samples_and_privacy.py','samples_and_smoothness.py']
    hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in code_files}
    result=dict(study=study,target='normalised finite Weierstrass k=1..K',
        amplitude=AMPLITUDE,score_tolerance=SCORE_TOL,holder_L=float(COMMON_HOLDER),
        calibration_gammas=ALL_GAMMAS,ns=NS,alphas=ALPHAS,gammas=GAMMAS,
        privacy_gamma=PRIVACY_GAMMA,smoothness_alpha=SMOOTHNESS_ALPHA,
        seed=args.seed,n_test=args.n_test,module_sha256=hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest(),
        bundle_sha256=hashes)
    return json.loads(json.dumps(result))

def check_settings(out,expected):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);path=out/'settings.json'
    if path.exists():
        if json.loads(path.read_text())!=expected:raise ValueError('Settings/source changed: use a new output directory.')
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
    p.add_argument('--output-dir',default=output);p.add_argument('--plot-only',action='store_true')
    return p

def validate_args(p,args):
    if min(args.workers,args.n_test)<1 or args.seed<0:p.error('Positive workers/test size and nonnegative seed required')
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
    name,seed,sweeps,n_test,study,i,j,rep,is_erm=job
    p=load_module(name)
    if study=='privacy':
        n=NS[j];gamma=PRIVACY_GAMMA;alpha=None if is_erm else ALPHAS[i]
    else:
        n=NS[i];gamma=GAMMAS[j];alpha=None if is_erm else SMOOTHNESS_ALPHA
    x,test=data(seed,rep,n,gamma,n_test)
    model=build_model(p,x,gamma,alpha)
    if is_erm:theta=erm(p,model)
    else:
        # All private experiments use exactly the same kernel as the checker.
        random=rng(seed,rep,int(round(gamma*1000)),n,int(round(alpha*1000)),2)
        theta=gibbs(p,model,random,sweeps,label=f'{study}: n={n}, alpha={alpha}, gamma={gamma}, rep={rep+1}')
    return i,j,rep,is_erm,risk(p,model,theta,test,gamma)

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
        axes=[ax];ax.set_title(rf'Weierstrass: $\gamma={PRIVACY_GAMMA:g}$')
        filename='n_privacy_risk.png'
    else:
        fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
        for j,gamma in enumerate(GAMMAS):
            add(axes[0],primary[:,j],rf'$\gamma={gamma:g}$')
            add(axes[1],erm[:,j],rf'$\gamma={gamma:g}$')
        axes[0].set_title(rf'Private: $\alpha={SMOOTHNESS_ALPHA:g}$');axes[1].set_title('Non-private ERM')
        filename='n_smoothness_risk.png'
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_yscale('log',base=2)
        ax.set_xticks(NS,[str(n) for n in NS]);ax.set_xlabel('Sample size n')
        ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Mean squared score risk')
    fig.text(.5,.015,f'{args.reps} datasets; {args.sweeps:,} sweeps/private release; one final draw; bars = 1 SE',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(Path(args.output_dir)/filename,dpi=180);plt.close(fig)

def run_risk_study(study):
    p=base_parser(f'Weierstrass {study} risk: one final coefficient draw per dataset.',f'results/{study}')
    p.add_argument('--sweeps',type=int,default=100000)
    p.add_argument('--reps',type=int,default=50)
    args=p.parse_args();validate_args(p,args)
    if min(args.sweeps,args.reps)<1:p.error('Positive sweeps/repetitions required')
    out=Path(args.output_dir)
    expected=settings(args,study)
    expected.update(sweeps=args.sweeps,reps=args.reps,retained_draws=1,initial_state='zero',
        axes=['privacy','n','repetition'] if study=='privacy' else ['n','gamma','repetition'])
    with output_lock(out):
        check_settings(out,expected)
        shape=(6,len(NS),args.reps) if study=='privacy' else (len(NS),len(GAMMAS),args.reps)
        file=out/('privacy_risk.npy' if study=='privacy' else 'smoothness_risk.npy')
        primary=np.load(file,allow_pickle=False) if file.exists() else np.full(shape,np.nan)
        erm_file=out/'smoothness_erm_risk.npy'
        erms=(np.load(erm_file,allow_pickle=False) if erm_file.exists() else np.full(shape,np.nan)) if study=='smoothness' else None
        for a in [primary]+([erms] if erms is not None else []):
            if a.shape!=shape or np.isinf(a).any() or np.any(a<0):raise ValueError('Invalid stored risk array')
        jobs=[]
        for rep in range(args.reps):
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if np.isnan(primary[i,j,rep]):
                        jobs.append((args.module,args.seed,args.sweeps,args.n_test,study,i,j,rep,study=='privacy' and i==5))
                    if erms is not None and np.isnan(erms[i,j,rep]):
                        jobs.append((args.module,args.seed,args.sweeps,args.n_test,study,i,j,rep,True))
        if args.plot_only and jobs:raise ValueError('Incomplete arrays')
        if not args.plot_only:
            atomic_array(primary,file)
            if erms is not None:atomic_array(erms,erm_file)
            print(f'{len(jobs)} pending cells; private releases use {args.sweeps:,} sweeps; ERM uses none',flush=True)
            for count,result in enumerate(parallel_results(release,jobs,args.workers),1):
                i,j,rep,is_erm,value=result
                if study=='smoothness' and is_erm:erms[i,j,rep]=value;atomic_array(erms,erm_file)
                else:primary[i,j,rep]=value;atomic_array(primary,file)
                print(f'{count}/{len(jobs)} pending cells completed: indices=({i},{j},{rep}), ERM={is_erm}, risk={value:.6g}',flush=True)
        plot(primary,erms,args,study)
        print(f'Saved {file}; shape={shape}',flush=True)
