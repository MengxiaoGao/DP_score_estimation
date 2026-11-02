"""Generate both R-hat grid CSVs using the original experiment implementation.
Place beside experiment_common.py and convergence_common.py from the bundle.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import experiment_common as c
from convergence_common import check_case

def main():
    parser=c.base_parser(__doc__, 'results/rhat_grids')
    parser.add_argument('--study', choices=('both','privacy','smoothness'), default='both')
    parser.add_argument('--steps', type=int, default=500000)
    parser.add_argument('--burn-fraction', type=float, default=.1)
    parser.add_argument('--dataset-rep', type=int, default=0)
    parser.add_argument('--trace-dir')
    parser.add_argument('--report-every', type=int, default=10000)
    args=parser.parse_args();c.validate_args(parser,args)
    if args.steps<20 or not 0<=args.burn_fraction<1 or args.steps-int(args.steps*args.burn_fraction)<8:
        parser.error('Insufficient post-burn draws')
    if args.dataset_rep<0 or args.report_every<0:parser.error('Invalid dataset/progress setting')
    if args.trace_dir and not Path(args.trace_dir).is_dir():parser.error('Trace directory must exist')
    studies=('privacy','smoothness') if args.study=='both' else (args.study,)
    for study in studies:
        cases=[(n,a,.5) for n in c.NS for a in c.ALPHAS] if study=='privacy' else [(n,2.,g) for n in c.NS for g in c.GAMMAS]
        out=Path(args.output_dir)/study
        expected=c.settings(args,'convergence_grids')
        expected['grid_source_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                       for name in ('generate_rhat_grids.py','convergence_common.py')}
        expected.update(cases=[list(v) for v in cases],study=study,steps=args.steps,
                        burn_fraction=args.burn_fraction,dataset_rep=args.dataset_rep,chains=4,initial_state='zero',backend='experiment_python')
        with c.output_lock(out):
            c.check_settings(out,expected,plot_only=args.plot_only)
            path=out/'convergence.csv'
            frame=pd.read_csv(path) if path.exists() else pd.DataFrame()
            done=set()
            if not frame.empty:
                if frame.duplicated(['n','alpha','gamma']).any():raise ValueError('Duplicate grid rows')
                for r in frame.itertuples():
                    key=(r.n,r.alpha,r.gamma)
                    if key not in cases or r.steps!=args.steps or not np.isclose(r.step_factor,args.step_factor):
                        raise ValueError('Stored diagnostics do not match requested grid')
                    done.add(key)
            jobs=[]
            for i,case in enumerate(cases):
                if case not in done:
                    jobs.append((args.module,args.seed,args.n_test,i,(args.steps,),args.burn_fraction,args.steps,
                                 args.dataset_rep,args.trace_dir,args.report_every,c.lmc_options(args),case))
            if args.plot_only and jobs:raise ValueError('Grid incomplete; new chain simulations are required')
            for rows,_ in c.parallel_results(check_case,jobs,args.workers):
                row=rows[0];row.update(study=study,initial_state='all_zero',backend='experiment_python')
                frame=pd.concat((frame,pd.DataFrame([row])),ignore_index=True).sort_values(['n','alpha','gamma'])
                c.atomic_csv(frame,path)
                print(f'{study}: saved {len(frame)}/{len(cases)} settings to {path}',flush=True)
            print(f'Complete: {path}',flush=True)

if __name__=='__main__':main()
