"""Checkpoint-resuming formal run and final analysis."""
from pathlib import Path
import subprocess,sys,time,json
ROOT=Path(__file__).resolve().parent
LOG=ROOT/'logs';LOG.mkdir(exist_ok=True)

def run(name,args,log):
    log.write(f'\nSTART {name} {time.ctime()}\n');log.flush()
    result=subprocess.run([sys.executable,'-u']+args,stdout=log,stderr=log)
    if result.returncode:raise RuntimeError(f'{name} failed with code {result.returncode}')
    log.write(f'FINISH {name} {time.ctime()}\n');log.flush()

def main():
    start=time.time()
    with open(LOG/'run.log','a',buffering=1) as log:
        # A resumed run should not send all 750 already-complete jobs back
        # through a new process pool.  Individual simulation rows are the
        # authoritative checkpoints.
        gamma_rows=sum(1 for _ in (ROOT/'gamma_panel/rows').glob('*.json'))
        if gamma_rows!=750:
            run('gamma_panel',['gamma_panel.py','--workers','6'],log)
        else:
            log.write(f'\nSKIP gamma_panel: found {gamma_rows}/750 checkpoint rows {time.ctime()}\n');log.flush()
        run('analyze_gamma',['analyze_gamma.py'],log)
        run('stability',['stability.py','--workers','6'],log)
        run('analyze_stability',['analyze_stability.py'],log)
    completion=dict(complete=True,finished=time.time(),elapsed_seconds=time.time()-start,
        gamma_rows=sum(1 for _ in (ROOT/'gamma_panel/rows').glob('*.json')),
        stability_base_jobs=sum(1 for _ in (ROOT/'stability/rows').glob('*.json')))
    (LOG/'completion.json').write_text(json.dumps(completion,indent=2))
    print(json.dumps(completion),flush=True)

if __name__=='__main__':main()
