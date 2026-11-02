"""Shared numerical routines for the formal 1D diffusion experiments."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
ROOT=Path(__file__).resolve().parent
import sys,json,math,time
sys.path.insert(0,str(ROOT/'numerics'))
import numpy as np
from numba import njit
import diffusion_extension_lmc as d
import paper_numerics as pn
import run_paper as rp



def atomic_json(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

@njit(cache=True)
def fast_moments(x,left,right,centers,widths,masses,slopes,t):
    a=math.exp(-t);v=-math.expm1(-2*t);sv=math.sqrt(v);sig=sv/a
    p=np.zeros(len(x));sout=np.zeros(len(x));bad=np.zeros(len(x),np.bool_)
    norm=1/math.sqrt(2*math.pi);root2=math.sqrt(2.)
    for i in range(len(x)):
        mu=x[i]/a;den=0.;deriv=0.
        lo=np.searchsorted(right,mu-38*sig);hi=np.searchsorted(left,mu+38*sig,side='right')
        for j in range(lo,hi):
            if masses[j]<=0:continue
            zl=(a*left[j]-x[i])/sv;zu=(a*right[j]-x[i])/sv
            if zl>0:cdf=.5*(math.erfc(zl/root2)-math.erfc(zu/root2))
            else:cdf=.5*(math.erfc(-zu/root2)-math.erfc(-zl/root2))
            pl=norm*math.exp(-.5*zl*zl);pu=norm*math.exp(-.5*zu*zu)
            base=masses[j]/widths[j];sl=slopes[j]
            den+=base/a*((1+sl*(mu-centers[j]))*cdf+sl*sig*(pl-pu))
            deriv+=base/a*(((1+sl*(left[j]-centers[j]))*pl-(1+sl*(right[j]-centers[j]))*pu)/sv+sl/a*cdf)
        p[i]=max(0.,den)
        if den>1e-200:
            sout[i]=deriv/den;ym=(x[i]+v*sout[i])/a
            bad[i]=not math.isfinite(sout[i]) or ym<left[0]-1e-5 or ym>right[-1]+1e-5
        else:bad[i]=True
    return p,sout,bad

def exact_score(y,cells,t):
    _,s,bad=fast_moments(y,cells.left,cells.right,cells.centers,cells.widths,cells.masses,cells.slopes,t)
    if np.any(bad):s[bad]=pn.ou_affine_density_score(y[bad],cells,t)[1]
    return s

@njit(cache=True)
def uniform_interp(y,ss,bound):
    dx=2*bound/(len(ss)-1);out=np.empty_like(y)
    for j in range(len(y)):
        q=(y[j]+bound)/dx;i=int(math.floor(q))
        if i<0:out[j]=ss[0]
        elif i>=len(ss)-1:out[j]=ss[-1]
        else:out[j]=ss[i]+(q-i)*(ss[i+1]-ss[i])
    return out

def interpolated_score(y,cells,t,points):
    a=np.exp(-t);sd=np.sqrt(-np.expm1(-2*t));bound=a*max(abs(cells.left[0]),abs(cells.right[-1]))+8*sd
    xx=np.linspace(-bound,bound,points);ss=exact_score(xx,cells,t)
    out=uniform_interp(y,ss,bound);mask=np.abs(y)>bound
    if np.any(mask):out[mask]=exact_score(y[mask],cells,t)
    return out

def theory_grid(n,kappa):
    v=[1/n];T=np.log(n)
    while v[-1]<T:
        x=v[-1];v.append(min(1.,(1+kappa)*x) if x<1 else min(T,x+kappa))
    t=np.asarray(v[::-1]);h=t[:-1]-t[1:]
    if not np.all(h<=kappa*np.minimum(1,t[1:])*(1+1e-10)):raise RuntimeError('Invalid Lemma 1.7 grid')
    return t

def reverse(cells,n,gamma,alpha,rng,c):
    delta=d.delta_theory(n,alpha,gamma);kappa=c.kappa_multiplier*delta**2
    tt=theory_grid(n,kappa);y=rng.normal(size=c.particles);start=time.perf_counter()
    for a,b in zip(tt[:-1],tt[1:]):
        h=a-b;s=interpolated_score(y,cells,a,c.score_grid)
        y=np.exp(h)*y+2*np.expm1(h)*s+np.sqrt(np.expm1(2*h))*rng.normal(size=len(y))
    if not np.isfinite(y).all():raise RuntimeError('Nonfinite reverse sample')
    return y,len(tt)-1,kappa,time.perf_counter()-start

def target(x,c):return d.target_density(x,c.gamma,c.target,c.amplitude,c.K,c.lam,c.mixture_mu)
def draw(n,rng,c):return d.sample_target(n,c.gamma,rng,c.target,c.amplitude,c.K,c.lam,c.mixture_mu)

