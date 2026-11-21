"""Verify exact bytes of all files in SHA256SUMS.json."""
import argparse,json,sys
from data_io import ROOT,digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',default=str(ROOT/'SHA256SUMS.json'));a=p.parse_args()
 checks=json.load(open(a.manifest,encoding='utf8'));fail=[]
 for relative,expected in checks.items():
  path=ROOT/relative
  if not path.is_file() or digest(path)!=expected:fail.append(relative)
 if fail:
  print('CHECKSUM FAILURES:',*fail,sep='\n - ');sys.exit(1)
 print(f'OK: {len(checks)} files matched SHA256SUMS.json')
if __name__=='__main__':main()
