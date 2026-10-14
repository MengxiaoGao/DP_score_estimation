"""Publication figures and slope analysis for the gamma panel."""
from common import *
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from matplotlib.ticker import ScalarFormatter
import base64,html

METRICS=['density_tv','ou_tv','generated_binned_tv','generated_wasserstein','reverse_to_ou_binned_tv','iid_ou_binned_tv']
LABEL={'density_tv':'Private density TV','ou_tv':'Early-stopped OU TV','generated_binned_tv':'Generated binned TV','generated_wasserstein':'Generated Wasserstein-1','reverse_to_ou_binned_tv':'Reverse-to-OU binned TV','iid_ou_binned_tv':'IID-to-OU binned TV'}
COLORS=['#B86238','#B3546B','#8961A3','#437FA8','#237F79']

def summary(df):
    rows=[]
    for (n,g),x in df.groupby(['n','gamma']):
        r=dict(n=n,gamma=g,R=len(x))
        for m in METRICS:
            se=x[m].sem();h=student_t.ppf(.975,len(x)-1)*se;r.update({m+'_mean':x[m].mean(),m+'_se':se,m+'_low':x[m].mean()-h,m+'_high':x[m].mean()+h})
        for m in ['N','reverse_seconds','b','basis_dim','holder_L']:r[m+'_mean']=x[m].mean();r[m+'_se']=x[m].sem()
        rows.append(r)
    return pd.DataFrame(rows)

def slopes(df,B=2000):
    rng=np.random.default_rng(20261016);rows=[];ns=sorted(df.n.unique())
    for m in ['density_tv','ou_tv','generated_binned_tv','generated_wasserstein']:
        for g in sorted(df.gamma.unique()):
            arr=[df[(df.n==n)&(df.gamma==g)][m].to_numpy() for n in ns];mean=np.array([x.mean() for x in arr]);x=np.log(ns);w=(x-x.mean())/sum((x-x.mean())**2);s=float(w@np.log(mean))
            bs=[]
            for _ in range(B):bs.append(float(w@np.log([rng.choice(a,len(a),replace=True).mean() for a in arr])))
            lo,hi=np.quantile(bs,[.025,.975]);theory=-(g+1)/(2*g+3) if m!='generated_wasserstein' else np.nan
            rows.append(dict(metric=m,gamma=g,slope=s,ci95_low=lo,ci95_high=hi,theory_reference=theory,bootstrap_repetitions=B))
    return pd.DataFrame(rows)

def style():plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B7C1CD','text.color':'#25344A','axes.labelcolor':'#25344A','grid.color':'#E4EAF0','legend.frameon':False,'pdf.fonttype':42})
def save(fig,out,name):
    for ext in ['png','pdf','svg']:fig.savefig(out/'figures'/f'{name}.{ext}',dpi=210,bbox_inches='tight')
    plt.close(fig)

def plots(df,s,sl,out):
    # The alpha-panel stores the density/OU diagnostics and generated-sample
    # diagnostics in separate, key-aligned tables.  Merge them before making
    # the external gamma=0.5 bridge markers.
    alpha_tables=ROOT/'alpha_panel'/'tables'
    anchor_density=pd.read_csv(alpha_tables/'ablation_raw.csv')
    anchor_generated=pd.read_csv(alpha_tables/'panel_raw.csv').drop(columns=['density_tv','histogram_tv'])
    anchor_raw=anchor_density.merge(anchor_generated,on=['n','rep','alpha'],validate='one_to_one')
    anchor=anchor_raw[anchor_raw.alpha==2].groupby('n')[METRICS].agg(['mean','sem'])
    for m in METRICS:
        fig,ax=plt.subplots(figsize=(7.5,5))
        for (n,x),color in zip(s.groupby('n'),COLORS):
            x=x.sort_values('gamma');y=x[m+'_mean'];ax.errorbar(x.gamma,y,yerr=y-x[m+'_low'],marker='o',capsize=2,lw=1.7,color=color,label=f'n = {n:,}')
            a=anchor.loc[n,(m,'mean')];ae=anchor.loc[n,(m,'sem')]*student_t.ppf(.975,29);ax.errorbar([.5],[a],yerr=[ae],marker='D',mfc='white',mec=color,color=color,ls='none',ms=5)
        ax.set_xticks([.2,.4,.5,.6,.8,1.]);ax.set_xlabel('Smoothness gamma');ax.set_ylabel(LABEL[m]);ax.set_title(LABEL[m]+' vs smoothness',loc='left',fontweight='bold');ax.grid(axis='y');ax.legend(ncol=3,fontsize=8)
        fig.text(.12,-.02,'Circles: gamma panel (independent targets/datasets). Hollow diamonds: alpha-panel anchor at alpha=2, gamma=0.5.',fontsize=8,color='#657186');save(fig,out,m+'_vs_gamma')
        fig,ax=plt.subplots(figsize=(7.5,5))
        for (g,x),color in zip(s.groupby('gamma'),COLORS):
            x=x.sort_values('n');y=x[m+'_mean'];ax.errorbar(x.n,y,yerr=y-x[m+'_low'],marker='o',capsize=2,lw=1.7,color=color,label=f'gamma = {g:g}')
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(CONFIG['n_values'],[f'{n:,}' for n in CONFIG['n_values']]);ax.yaxis.set_major_formatter(ScalarFormatter());ax.set_xlabel('Training sample size n (log scale)');ax.set_ylabel(LABEL[m]+' (log scale)');ax.set_title(LABEL[m]+' vs sample size',loc='left',fontweight='bold');ax.grid();ax.legend(ncol=3,fontsize=8);save(fig,out,m+'_vs_n')
    for m in ['density_tv','ou_tv','generated_binned_tv']:
        x=sl[sl.metric==m];fig,ax=plt.subplots(figsize=(7.2,4.5));ax.errorbar(x.gamma,x.slope,yerr=np.vstack([x.slope-x.ci95_low,x.ci95_high-x.slope]),marker='o',capsize=4,color='#237F79',label='Empirical slope');ax.plot(x.gamma,x.theory_reference,'--',color='#B86238',label='Theory reference')
        ax.set_xticks(CONFIG['gammas']);ax.set_xlabel('Smoothness gamma');ax.set_ylabel('Slope of log(mean error) vs log(n)');ax.set_title(LABEL[m]+' — empirical and theoretical slopes',loc='left',fontweight='bold');ax.grid();ax.legend();save(fig,out,m+'_slopes')

def main():
    out=ROOT/'gamma_panel'
    # Prefer complete atomic checkpoints after a fresh simulation.  A
    # repository clone can regenerate every table and figure directly from
    # the checked-in aggregate CSV without the large checkpoint directory.
    row_files=list((out/'rows').glob('*.json'))
    if len(row_files)==750:
        df=pd.DataFrame([json.loads(p.read_text()) for p in row_files])
        df=df.sort_values(['n','gamma','rep']).reset_index(drop=True)
        raw_tmp=out/'raw.tmp.csv';df.to_csv(raw_tmp,index=False);raw_tmp.replace(out/'raw.csv')
    else:
        df=pd.read_csv(out/'raw.csv').sort_values(['n','gamma','rep']).reset_index(drop=True)
    if len(df)!=750 or not (df.groupby(['n','gamma']).size()==30).all():raise RuntimeError('Gamma panel incomplete')
    for x in ['figures','tables']:(out/x).mkdir(exist_ok=True)
    style();s=summary(df);sl=slopes(df);s.to_csv(out/'tables/summary.csv',index=False);sl.to_csv(out/'tables/slopes.csv',index=False)
    df.groupby('n').agg(N_min=('N','min'),N_max=('N','max'),mean_reverse_seconds=('reverse_seconds','mean')).to_csv(out/'tables/runtime.csv')
    plots(df,s,sl,out)
    structure='''# Organic paper structure for the two panels

1. **Privacy-budget sensitivity at fixed smoothness.** Fix gamma=0.5 and vary alpha. This isolates the finite-sample privacy–accuracy trade-off.
2. **Smoothness sensitivity at fixed privacy.** Fix alpha=2 and vary gamma in {0.2,0.4,0.6,0.8,1}. K=40 and amplitude=0.25 remain fixed. This examines adaptation to target regularity while keeping privacy strength constant.
3. Use the same n grid, 30 repetitions, projected LMC, density construction, reverse sampler, and metrics. The alpha=2, gamma=0.5 result from the alpha panel appears only as a hollow bridge marker and is excluded from gamma-slope fits.
4. Datasets are independent across gamma because gamma changes the data-generating distribution. Comparisons across alpha remain paired because that panel holds the distribution fixed.
5. For d=1 and alpha=2, compare the empirical n slope at each gamma with -(gamma+1)/(2gamma+3). Treat it as the polynomial part of a class-wise upper bound, not an equality for the fixed K=40 target.
6. Follow with the single-record replacement experiment as a mechanism-level stability diagnostic. It is distinct from the statistical rate panels and does not serve as a proof of differential privacy.
'''
    (out/'PAPER_STRUCTURE.md').write_text(structure)
    imgs=[]
    for p in sorted((out/'figures').glob('*.png')):imgs.append(f'<h2>{html.escape(p.stem)}</h2><img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" style="max-width:100%">')
    css='<style>body{font:16px/1.65 system-ui;color:#25344a;max-width:1100px;margin:40px auto;padding:24px}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border-bottom:1px solid #ddd}h2{margin-top:42px}</style>'
    intro='<h1>Gamma panel | alpha=2, K=40, amplitude=0.25</h1><p>750 complete runs: 5 sample sizes × 5 gamma values × 30 independent datasets. No clean score risk. Circles are the formal gamma panel; hollow gamma=0.5 diamonds are an external bridge to the alpha panel and are excluded from the slope fit.</p>'
    (out/'report.html').write_text('<meta charset="utf-8">'+css+intro+'<h2>Summary</h2>'+s.to_html(index=False,float_format=lambda x:f'{x:.5f}')+'<h2>Slopes</h2>'+sl.to_html(index=False,float_format=lambda x:f'{x:.4f}')+''.join(imgs))
    print('Gamma analysis complete')

if __name__=='__main__':main()
