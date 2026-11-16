"""Run audit, model training, reporting and validation from repository root."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['audit','main','all'],default='audit')
    p.add_argument('--epochs',type=int,default=60)
    args=p.parse_args()
    env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='2'
    cmd=[sys.executable,str(ROOT/'scripturi'/'benchmark.py'),'--stage',args.stage,'--epochs',str(args.epochs)]
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    if args.stage=='all':
        for filename in ['summarize_results.py','validate_experiments.py']:
            subprocess.run([sys.executable,str(ROOT/'scripturi'/filename)],cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
