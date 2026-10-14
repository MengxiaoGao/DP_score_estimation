"""Dataset-level analysis of the single-record replacement pilot."""
from common import *
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
import base64,html

COLORS=['#B86238','#8961A3','#237F79'];ALPHAS=[.5,2.,8.]

def independent_units(raw):
    metrics=[c for c in raw if any(c.startswith(x) for x in ['nonprivate_','private_coupled_','private_mode_','noise_floor_']) and c.endswith(('_rms','_max'))]
    return raw.groupby(['n','rep','alpha','replacement_kind'],as_index=False)[metrics].mean()

def summary(units):
    metrics=[c for c in units if c.endswith(('_rms','_max'))];rows=[]
    for key,g in units.groupby(['n','alpha','replacement_kind']):
        r=dict(zip(['n','alpha','replacement_kind'],key),datasets=len(g))
        for m in metrics:
            se=g[m].sem();h=student_t.ppf(.975,len(g)-1)*se;r.update({m+'_mean':g[m].mean(),m+'_se':se,m+'_low':g[m].mean()-h,m+'_high':g[m].mean()+h})
        rows.append(r)
    return pd.DataFrame(rows)

def style():plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B7C1CD','text.color':'#25344A','axes.labelcolor':'#25344A','grid.color':'#E4EAF0','legend.frameon':False,'pdf.fonttype':42})
def save(fig,out,name):
    for ext in ['png','pdf','svg']:fig.savefig(out/'figures'/f'{name}.{ext}',dpi=210,bbox_inches='tight')
    plt.close(fig)

def main():
    out=ROOT/'stability';raw=pd.read_csv(out/'raw.csv')
    if len(raw)!=450 or not (raw.groupby(['n','rep','replacement','alpha']).size()==1).all():raise RuntimeError('Stability pilot incomplete')
    units=independent_units(raw);s=summary(units);units.to_csv(out/'tables/dataset_units.csv',index=False);s.to_csv(out/'tables/summary.csv',index=False)
    for kind in ['typical','stress']:
        q=s[s.replacement_kind==kind]
        for suffix,label in [('coef_rms','Coefficient RMS shift'),('function_rms','Output-function RMS shift')]:
            for estimator,title in [('private_mode','Private target mode'),('private_coupled','Private LMC, synchronous coupling')]:
                fig,axs=plt.subplots(1,3,figsize=(12,3.9),sharey=True)
                for ax,(n,g),color in zip(axs,q.groupby('n'),COLORS):
                    g=g.sort_values('alpha');m=estimator+'_'+suffix;nm='nonprivate_'+suffix;y=g[m+'_mean'];yn=g[nm+'_mean']
                    ax.errorbar(g.alpha,y,yerr=y-g[m+'_low'],marker='o',capsize=3,color=color,label=title)
                    ax.errorbar(g.alpha,yn,yerr=yn-g[nm+'_low'],marker='s',capsize=3,color='#687589',label='Non-private ERM')
                    ax.set_xscale('log',base=4);ax.set_xticks(ALPHAS,[str(a) for a in ALPHAS]);ax.set_xlabel('Total privacy budget alpha');ax.set_title(f'n = {n:,}');ax.grid(axis='y')
                axs[0].set_ylabel(label);axs[-1].legend(fontsize=8);fig.suptitle(f'{kind.capitalize()} one-record replacements | {title}',fontweight='bold');fig.tight_layout();save(fig,out,f'{kind}_{estimator}_{suffix}')
            fig,axs=plt.subplots(1,3,figsize=(12,3.9),sharey=True)
            for ax,(n,g),color in zip(axs,q.groupby('n'),COLORS):
                g=g.sort_values('alpha');m='private_coupled_'+suffix;nm='noise_floor_'+suffix;y=g[m+'_mean'];yn=g[nm+'_mean']
                ax.errorbar(g.alpha,y,yerr=y-g[m+'_low'],marker='o',capsize=3,color=color,label='Input replacement shift')
                ax.errorbar(g.alpha,yn,yerr=yn-g[nm+'_low'],marker='^',capsize=3,color='#B86238',label='Independent-run noise floor')
                ax.set_xscale('log',base=4);ax.set_xticks(ALPHAS,[str(a) for a in ALPHAS]);ax.set_xlabel('Total privacy budget alpha');ax.set_title(f'n = {n:,}');ax.grid(axis='y')
            axs[0].set_ylabel(label);axs[-1].legend(fontsize=8);fig.suptitle(f'{kind.capitalize()} replacements | input response vs mechanism randomness',fontweight='bold');fig.tight_layout();save(fig,out,f'{kind}_signal_noise_{suffix}')
    # Ratios are computed per independent dataset after averaging its replacements.
    for pre in ['private_mode','private_coupled']:
        for suffix in ['coef_rms','function_rms']:
            units[f'{pre}_{suffix}_ratio_to_nonprivate']=units[f'{pre}_{suffix}']/units[f'nonprivate_{suffix}'].replace(0,np.nan)
    ratio_cols=[c for c in units if 'ratio_to_nonprivate' in c]
    ratio=units.groupby(['n','alpha','replacement_kind'])[ratio_cols].agg(['mean','median','sem']).reset_index();ratio.to_csv(out/'tables/ratios.csv',index=False)
    notes='''# Private versus non-private single-record replacement experiment

## Question
Does replacing one training record produce a smaller change in a private output than in a non-private constrained score-matching estimator?

## Design
- Fix gamma=0.5, K=40, amplitude=0.25. Use n in {1000,4000,16000}, total alpha in {0.5,2,8}, and 10 independent base datasets per n.
- Freeze the spline basis, bandwidth, truncation radius, and constraints at the alpha=2 model class for every private alpha and the non-private estimator. This makes coefficient vectors directly comparable and prevents bandwidth/dimension changes from masquerading as stability.
- For each dataset create three in-distribution random replacements and two labeled stress replacements. First average replacements within each dataset; inference uses the 10 datasets as independent units.
- Non-private output: constrained empirical score-matching ERM. Private deterministic center: mode of the ridge-regularized private target. Released private output: 100,000-step projected LMC draw.
- For D and D' the private chains start at zero and use identical Gaussian innovations. This synchronous coupling isolates controlled input response, but it is coupling-dependent and is not itself a differential-privacy theorem.
- Separately compare two independent private runs on the same D. This is the mechanism-randomness floor. Report coefficient RMS/max shifts and basis-invariant RMS/max changes of the output score function on 10,000 independent reference points. These are output-shift diagnostics, not clean-score risk.

## Interpretation limits
Differential privacy bounds output distributions for neighboring datasets; it does not require every realized coefficient vector to move less than a non-private ERM under an arbitrary coupling. A smaller private mode shift supports regularization-induced stability. A smaller synchronously coupled LMC shift is useful numerical evidence, while independent-run variability shows whether that response would be visible from one released draw. Stress replacements are diagnostic, not an optimized worst-case search. The private LMC inherits the finite-step mixing/privacy caveat of the main simulations.
'''
    (out/'DESIGN.md').write_text(notes)
    imgs=[]
    for p in sorted((out/'figures').glob('*.png')):imgs.append(f'<h2>{html.escape(p.stem)}</h2><img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" style="max-width:100%">')
    css='<style>body{font:16px/1.65 system-ui;color:#25344a;max-width:1100px;margin:40px auto;padding:24px}table{display:block;overflow-x:auto;border-collapse:collapse;font-size:12px}td,th{padding:8px;border-bottom:1px solid #ddd}h2{margin-top:42px}</style>'
    intro='<h1>Private vs non-private single-record stability pilot</h1><p>30 independent base datasets; 450 neighbor/alpha rows. Common model class, no Gibbs, no clean-score risk. Pointwise intervals use datasets—not individual replacements—as the independent units.</p>'
    (out/'report.html').write_text('<meta charset="utf-8">'+css+intro+'<h2>Summary</h2>'+s.to_html(index=False,float_format=lambda x:f'{x:.5g}')+'<h2>Ratios</h2>'+ratio.to_html(index=False,float_format=lambda x:f'{x:.4g}')+''.join(imgs))
    print('Stability analysis complete')

if __name__=='__main__':main()
