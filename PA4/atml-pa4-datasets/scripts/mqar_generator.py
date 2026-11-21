"""PA4 fixed synthetic MQAR data (query-position classification, NOT next-token LM).
Public data recipe, no pretrained models. Version 1, seed 6304.
The values are 128..191, keys 64..127, filler 2..63.
"""
import hashlib
import random
from typing import Dict

T = 256
MASTER_SEED = 6304

def stable_seed(split: str, k: int, regime: str, index: int) -> int:
    msg = f"PA4-MQAR-v1|{MASTER_SEED}|{split}|{k}|{regime}|{index}".encode()
    return int.from_bytes(hashlib.sha256(msg).digest()[:8], "big")

def make_example(split: str, k: int, regime: str, index: int) -> Dict:
    assert split in ('train', 'development', 'test')
    assert k in (4, 8, 16)
    assert regime in ('near', 'far', 'mixed')
    rng = random.Random(stable_seed(split, k, regime, index))
    keys = rng.sample(range(64,128), k)
    vals = rng.sample(range(128,192), k)
    tokens = [rng.randrange(2,64) for _ in range(T)]
    for j in range(k):
        tokens[2*j] = keys[j]
        tokens[2*j+1] = vals[j]
    order = list(range(k))
    rng.shuffle(order)
    used = set(range(2*k))
    positions = {}
    for j in order:
        src = 2*j
        candidates = list(range(2*k+1, T))
        if regime == 'near':
            candidates = [q for q in candidates if 1 <= q-src <= 63]
        elif regime == 'far':
            candidates = [q for q in candidates if 128 <= q-src <= 250]
        else:
            candidates = [q for q in candidates if 1 <= q-src <= 250]
        rng.shuffle(candidates)
        choices = [q for q in candidates if q not in used]
        if not choices:
            raise ValueError(f'No legal query position: {split} K={k} {regime} i={index}')
        pos = choices[0]
        positions[j] = pos
        used.add(pos)
        tokens[pos] = keys[j]
    queries = sorted((pos,vals[j],2*j,pos-2*j) for j,pos in positions.items())
    out = {
        'id':f'{split}-{regime}-K{k}-{index:05d}',
        'split':split,'regime':regime,'K':k,'tokens':tokens,
        'query_positions':[x[0] for x in queries],
        'answer_ids':[x[1] for x in queries],
        'source_positions':[x[2] for x in queries],
        'distances':[x[3] for x in queries],
    }
    validate(out)
    return out

def validate(ex):
    x=ex['tokens']; q=ex['query_positions']; d=ex['distances']; sources=ex['source_positions']; answers=ex['answer_ids']
    assert len(x)==T and len(q)==len(set(q))==ex['K']
    assert len(answers)==ex['K']
    assert all(2<=v<256 for v in x)
    assert all(x[s] == x[p] and x[s+1] == y and p-s==dist for p,s,y,dist in zip(q,sources,answers,d))
    assert all(s%2==0 and s<2*ex['K'] and p>2*ex['K'] for p,s in zip(q,sources))
    if ex['regime']=='near':assert all(1<=v<64 for v in d)
    if ex['regime']=='far':assert all(128<=v<=250 for v in d)
    if ex['regime']=='mixed':assert all(1<=v<=250 for v in d)

if __name__ == '__main__':
    for regime in ('near','far','mixed'):
        for k in (4,8,16):
            for idx in range(1024):make_example('test',k,regime,idx)
    print('MQAR generator passed 9,216 held-out invariant checks')
