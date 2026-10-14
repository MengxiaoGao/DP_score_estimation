"""Diagnostic only: true-score reconstruction, not a private competitor."""
from revised import *
rows=[]
for _,r in pd.read_csv(ROOT/'tables/ablation_raw.csv').iterrows():
    c=cells_for(int(r.n),int(r.rep),r.alpha)
    s=d.true_score(c.centers,C.gamma,C.target,C.amplitude,C.K,C.lam,C.mixture_mu)
    safe=d._safe_slopes(s,c.widths);base=c.masses/c.widths
    oracle=replace(c,slopes=safe,intercept=base*(1-safe*c.centers),slopecoef=base*safe)
    rows.append(dict(n=int(r.n),rep=int(r.rep),alpha=r.alpha,oracle_density_tv=rp.density_tv(oracle,C,192),
        histogram_tv=r.histogram_tv,private_density_tv=r.density_tv))
df=pd.DataFrame(rows);df.to_csv(ROOT/'tables/oracle_diagnostic_raw.csv',index=False)
df['oracle_minus_histogram']=df.oracle_density_tv-df.histogram_tv
print(df.groupby(['n','alpha'])[['oracle_minus_histogram']].mean().to_string())
