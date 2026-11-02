"""Plot both maximum-coefficient R-hat grids from CSVs; no simulation imports."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from paper_style import set_paper_style


def save(fig,out,name):
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'{name}.{ext}',dpi=210,bbox_inches='tight')
    plt.close(fig)


def plot_rhat_heatmap(path,out,study):
    df=pd.read_csv(path);vary='alpha' if study=='privacy' else 'gamma'
    # Also accept pivoted grids exported as parameter rows and n columns.
    if 'max_coefficient_rhat' not in df and vary in df:
        cols=[name for name in df if name!=vary]
        if not cols or not all(str(name).isdigit() for name in cols):
            raise ValueError('Wide grids require numeric sample-size column names')
        df=df.melt(id_vars=vary,value_vars=cols,var_name='n',value_name='max_coefficient_rhat')
        df['n']=df['n'].astype(int)
        df['gamma' if study=='privacy' else 'alpha']=.5 if study=='privacy' else 2.
        df['steps']=500000;df['step_factor']=.1
    required={'n','alpha','gamma','steps','step_factor','max_coefficient_rhat'}
    if not required<=set(df):raise ValueError('Convergence grid CSV lacks required columns')
    if df.duplicated(['n',vary]).any():raise ValueError('Duplicate heatmap cells')
    fixed='gamma' if study=='privacy' else 'alpha'
    if df[fixed].nunique()!=1 or df.steps.nunique()!=1 or df.step_factor.nunique()!=1:
        raise ValueError('Heatmap needs one fixed parameter, length and step factor')
    grid=df.pivot(index=vary,columns='n',values='max_coefficient_rhat').sort_index(ascending=False).sort_index(axis=1)
    counts=df.groupby([vary,'n']).size().unstack()
    if counts.isna().any().any():raise ValueError('Incomplete convergence grid')
    values=grid.to_numpy(dtype=float)
    invalid=~np.isfinite(values)
    if 'undefined_coefficient_rhat' in df:
        invalid |= df.pivot(index=vary,columns='n',values='undefined_coefficient_rhat').reindex(index=grid.index,columns=grid.columns).to_numpy()>0
    finite=values[~invalid]
    vmax=max(1.01,float(finite.max())) if finite.size else 1.01
    cmap=LinearSegmentedColormap.from_list('neutral_blue_grey',['#F3F5F7','#AAB9C8','#40566D']);cmap.set_bad('#D9DEE5')
    fig,ax=plt.subplots(figsize=(7,4.6))
    im=ax.imshow(np.ma.array(values,mask=invalid),cmap=cmap,vmin=1.,vmax=vmax,aspect='auto')
    for i in range(len(grid.index)):
        for j in range(len(grid.columns)):
            label=(r'$\infty$' if np.isposinf(values[i,j]) else 'undefined') if invalid[i,j] else f'{values[i,j]:.3f}'
            colour='white' if not invalid[i,j] and (values[i,j]-1)/(vmax-1)>.6 else '#25344A'
            ax.text(j,i,label,ha='center',va='center',fontsize=9,color=colour)
    ax.set_xticks(range(len(grid.columns)),[f'{int(n):,}' for n in grid.columns])
    ax.set_yticks(range(len(grid.index)),[f'{v:g}' for v in grid.index])
    ax.set_xlabel('Training sample size n');ax.set_ylabel(r'Privacy budget $\alpha$' if study=='privacy' else r'Smoothness $\gamma$')
    fig.colorbar(im,ax=ax,label=r'Maximum coefficient $\widehat R$')
    fig.text(.1,.015,f'{fixed} = {df[fixed].iloc[0]:g}; {int(df.steps.iloc[0]):,} iterations; step factor {df.step_factor.iloc[0]:g}. Grey: undefined.',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));save(fig,out,f'{study}_rhat_heatmap')



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv-dir',type=Path,default=Path('results/rhat_grids'))
    parser.add_argument('--privacy-csv',type=Path)
    parser.add_argument('--smoothness-csv',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('figures/rhat_grids'))
    args=parser.parse_args()
    explicit=args.privacy_csv is not None or args.smoothness_csv is not None
    inputs={name:getattr(args,name+'_csv') for name in ('privacy','smoothness')}
    if not explicit:
        inputs={name:args.csv_dir/name/'convergence.csv' for name in inputs}
    for name,path in inputs.items():
        if path is not None and not path.is_file():parser.error(f'CSV not found: {path}')
    set_paper_style()
    plt.rcParams.update({'axes.edgecolor':'#B7C1CD','text.color':'#25344A',
        'axes.labelcolor':'#25344A','svg.fonttype':'none'})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for study,path in inputs.items():
        if path is not None:plot_rhat_heatmap(path,args.output_dir,study)
    print(f'Saved heatmaps in {args.output_dir.resolve()}')

if __name__=='__main__':main()
