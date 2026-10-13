"""Numerical safeguards for the supplied diffusion-extension implementation.
No Gibbs calls. Numba accelerates the same cyclic Dykstra projection.
OU moments use survival probabilities in the positive tail, a boundary
formula for the density derivative, and log-scaled Gaussian quadrature only
when moment evaluation underflows or yields a conditional mean outside support.
"""
import math
import numpy as np
from numba import njit
from scipy.special import logsumexp
from numpy.polynomial.legendre import leggauss
import diffusion_extension_lmc as original
from dp_score_sim import density

@njit(cache=True)
def dykstra_project(y, B, D, max_cycles=500, tol=1e-10):
    x=y.copy(); m=len(x); pb=np.zeros(m); pp=np.zeros((max(m-1,0),2))
    violation=0.
    for cyc in range(1,max_cycles+1):
        old=x.copy()
        for j in range(m):
            z=x[j]+pb[j]; q=min(B,max(-B,z)); pb[j]=z-q; x[j]=q
        for j in range(m-1):
            z0=x[j]+pp[j,0]; z1=x[j+1]+pp[j,1]; diff=z1-z0
            e=.5*max(0.,abs(diff)-D)*(1. if diff>=0 else -1.)
            q0=z0+e; q1=z1-e
            pp[j,0]=z0-q0; pp[j,1]=z1-q1; x[j]=q0; x[j+1]=q1
        violation=0.; change=0.
        for j in range(m):
            violation=max(violation,abs(x[j])-B); change=max(change,abs(x[j]-old[j]))
            if j+1<m: violation=max(violation,abs(x[j+1]-x[j])-D)
        if change<=tol and violation<=10*tol:return x,cyc,violation
    return x,max_cycles,violation

@njit(cache=True)
def _moments(x,left,right,centers,widths,masses,slopes,t):
    a=math.exp(-t); v=-math.expm1(-2*t); sv=math.sqrt(v); sigma=sv/a
    p=np.zeros(len(x)); score=np.zeros(len(x)); bad=np.zeros(len(x),np.bool_)
    root2=math.sqrt(2.); norm=1/math.sqrt(2*math.pi)
    for i in range(len(x)):
        mu=x[i]/a; den=0.; deriv=0.
        for j in range(len(left)):
            if masses[j]<=0:continue
            zl=(a*left[j]-x[i])/sv; zu=(a*right[j]-x[i])/sv
            if zl>0:cdf=.5*(math.erfc(zl/root2)-math.erfc(zu/root2))
            else:cdf=.5*(math.erfc(-zu/root2)-math.erfc(-zl/root2))
            pl=norm*math.exp(-.5*zl*zl); pu=norm*math.exp(-.5*zu*zu)
            base=masses[j]/widths[j]; s=slopes[j]
            den+=base/a*((1+s*(mu-centers[j]))*cdf+s*sigma*(pl-pu))
            wl=1+s*(left[j]-centers[j]); wr=1+s*(right[j]-centers[j])
            deriv+=base/a*((wl*pl-wr*pu)/sv+s/a*cdf)
        p[i]=max(den,0.)
        if den>1e-200:
            score[i]=deriv/den
            ym=(x[i]+v*score[i])/a
            bad[i]=not math.isfinite(score[i]) or ym<left[0]-1e-5 or ym>right[-1]+1e-5
        else:bad[i]=True
    return p,score,bad

_GLX,_GLW=leggauss(24)

@njit(cache=True)
def _tail_fallback(x,p,s,bad,left,right,centers,widths,masses,slopes,t,gx,gw):
    a=math.exp(-t);v=-math.expm1(-2*t);sig=math.sqrt(v)/a
    for i in range(len(x)):
        if not bad[i]:continue
        mu=x[i]/a; logmax=-math.inf
        lw=np.full((len(left),len(gx)),-math.inf); yy=np.zeros_like(lw)
        for j in range(len(left)):
            if masses[j]<=0:continue
            l=left[j];r=right[j];near=min(r,max(l,mu));dist=abs(mu-near)
            span=min(r-l,40*sig*sig/max(dist,1e-300)) if dist>sig else min(r-l,12*sig)
            lo=max(l,near-span);hi=min(r,near+span)
            for k in range(len(gx)):
                z=(lo+hi)/2+(hi-lo)/2*gx[k]
                weight=masses[j]/widths[j]*(1+slopes[j]*(z-centers[j]))*gw[k]*(hi-lo)/2
                val=math.log(weight)-.5*(x[i]-a*z)**2/v-.5*math.log(2*math.pi*v)
                lw[j,k]=val;yy[j,k]=z;logmax=max(logmax,val)
        den=0.;num=0.
        for j in range(len(left)):
            for k in range(len(gx)):
                wt=math.exp(lw[j,k]-logmax);den+=wt;num+=wt*yy[j,k]
        lp=logmax+math.log(den);p[i]=math.exp(lp) if lp>-745 else 0.
        s[i]=-(x[i]-a*num/den)/v
    return p,s

def ou_affine_density_score(x,cells,t):
    x=np.atleast_1d(np.asarray(x,dtype=float))
    p,s,bad=_moments(x,cells.left,cells.right,cells.centers,cells.widths,cells.masses,cells.slopes,t)
    if np.any(bad):
        p,s=_tail_fallback(x,p,s,bad,cells.left,cells.right,cells.centers,cells.widths,cells.masses,cells.slopes,t,_GLX,_GLW)
    return p,s

def install():
    original.dykstra_project=dykstra_project
    original.ou_affine_density_score=ou_affine_density_score
    # Resolve the oscillatory multiscale target's normalizer with composite
    # Gauss-Legendre quadrature instead of a warning-producing infinite quad.
    old=density._normalizer_cached
    from functools import lru_cache
    @lru_cache(maxsize=64)
    def normalizer(gamma,target,amplitude,K,lam,mixture_mu):
        if target!='multiscale':return old(gamma,target,amplitude,K,lam,mixture_mu)
        edges=np.linspace(-12,12,2401); z,w=leggauss(32)
        xx=(edges[:-1,None]+edges[1:,None])/2+.005*z
        ff=np.exp(density.log_unnormalized_density(xx,gamma,target,amplitude,K,lam,mixture_mu))
        return float(np.sum(ff*w)*.005)
    density._normalizer_cached=normalizer
