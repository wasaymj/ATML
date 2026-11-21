"""Regenerate checksums after instructor-reviewed release assets are added."""
import json
from data_io import ROOT,digest
paths=[]
for base in ['data','docs','scripts','tests']:
 for p in (ROOT/base).rglob('*'):
  if p.is_file() and p.suffix not in ['.pyc'] and '__pycache__' not in p.parts:
   paths.append(p.relative_to(ROOT).as_posix())
checks={str(p):digest(ROOT/p) for p in sorted(set(paths))}
(ROOT/'SHA256SUMS.json').write_text(json.dumps(checks,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print('Recorded',len(checks),'asset/code checksums')
