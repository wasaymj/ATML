"""Small standard-library data IO helpers. No assignment methods are implemented here."""
from __future__ import annotations
import gzip, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def records(path):
    path = Path(path)
    op = gzip.open if path.suffix == '.gz' else open
    with op(path, 'rt', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_no}: {exc}') from exc

def write_records(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    op = gzip.open if path.suffix == '.gz' else open
    with op(path, 'wt', encoding='utf-8') as out:
        for obj in rows:
            out.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + '\n')

def digest(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda: f.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()
