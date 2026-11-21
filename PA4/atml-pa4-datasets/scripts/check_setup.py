"""Check student-run public assets and optional GPU-generated prefix banks."""
from pathlib import Path
import json, gzip, sys
ROOT=Path(__file__).resolve().parents[1]
FILES={
 'C4 calibration': ('data/public_locked/c4_calibration_locked.jsonl.gz',128),
 'C4 development': ('data/public_locked/c4_development_locked.jsonl.gz',32),
 'C4 validation windows': ('data/public_locked/c4_validation_windows_locked.jsonl.gz',None),
 'WikiText-2 test windows': ('data/public_locked/wikitext2_test_windows_locked.jsonl.gz',None),
 'HellaSwag examples': ('data/public_locked/hellaswag_val_locked.jsonl.gz',2000),
 'GSM8K train': ('data/public_locked/gsm8k_train_locked.jsonl.gz',500),
 'GSM8K test': ('data/public_locked/gsm8k_test_locked.jsonl.gz',60),
}
def load(path):
 with gzip.open(path,'rt',encoding='utf8') as f:return [json.loads(line) for line in f]
def main():
 bad=0
 for name,(file,count) in FILES.items():
  p=ROOT/file
  if not p.exists():print('MISSING',name,'— run python scripts/setup_data.py --public');bad+=1;continue
  arr=load(p)
  if count and len(arr)!=count:print('BAD COUNT',name,len(arr),'expected',count);bad+=1
  elif name.endswith('windows') and sum(x['scored_tokens'] for x in arr)!=50000:print('BAD TOKEN QUOTA',name);bad+=1
  else:print('OK',name,len(arr))
 bp=ROOT/'data/prefix_banks/prefix_banks_locked.jsonl.gz'
 print(('OK' if bp.exists() else 'OPTIONAL NOT GENERATED'),'prefix banks')
 lock=ROOT/'data/local_source_revisions.json'
 if not lock.exists():print('MISSING local revision lock');bad+=1
 else:print('OK source snapshot lock')
 if bad:sys.exit(1)
 print('PASS: public assets prepared locally. Model correctness checks remain your experiment responsibility.')
if __name__=='__main__':main()
