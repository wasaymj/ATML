"""Deterministically generate Task 2B training/development MQAR data only.

Final test cases are already frozen in data/mqar/mqar_test.jsonl.gz.
This script does not implement a model or evaluation method.
"""
import argparse, sys
from pathlib import Path
from data_io import ROOT, write_records
from mqar_generator import make_example

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['train','development'],required=True)
    p.add_argument('--k',type=int,choices=[4,8,16],required=True)
    p.add_argument('--regime',choices=['near','far','mixed'],required=True)
    p.add_argument('--count',type=int,default=1024)
    p.add_argument('--start-index',type=int,default=0)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.count<=0 or args.start_index<0:p.error('count must be positive; start-index nonnegative')
    out=args.output or ROOT/'data'/'mqar'/f'{args.split}_K{args.k}_{args.regime}_{args.start_index}_{args.count}.jsonl.gz'
    if out.exists():p.error(f'Output already exists: {out}; use a new path to avoid accidental overwrite')
    write_records(out,(make_example(args.split,args.k,args.regime,i) for i in range(args.start_index,args.start_index+args.count)))
    print(out)
if __name__=='__main__':main()
