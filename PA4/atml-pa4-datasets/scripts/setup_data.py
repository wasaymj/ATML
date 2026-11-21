"""Student-run PA4 dataset materializer; no assignment algorithms included.

python scripts/setup_data.py --public
python scripts/setup_data.py --prefix-banks   # T4 GPU recommended
python scripts/setup_data.py --all

Public source snapshots are resolved to immutable HF commit hashes on first run,
then recorded in data/local_source_revisions.json. Re-running uses those hashes.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, pathlib, random, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
LOCK=ROOT/'data/local_source_revisions.json'
MAN=ROOT/'data/public_manifests'
PUBLIC=ROOT/'data/public_locked'
BANK=ROOT/'data/prefix_banks'
REPOS={'c4':('dataset','allenai/c4'), 'wikitext':('dataset','Salesforce/wikitext'),
       'hellaswag':('dataset','Rowan/hellaswag'),'gsm8k':('dataset','openai/gsm8k'),
       'model_base':('model','Qwen/Qwen2.5-1.5B'),
       'model_target':('model','Qwen/Qwen2.5-1.5B-Instruct'),
       'model_draft':('model','Qwen/Qwen2.5-0.5B-Instruct')}

def read(path):
    with open(path, encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]

def save_lines(path, items):
    path.parent.mkdir(parents=True,exist_ok=True)
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'wt',encoding='utf8') as f:
        for x in items:f.write(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n')
    print(f'Wrote {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)',flush=True)

def hash_text(s): return hashlib.sha256(s.encode('utf8')).hexdigest()

def revisions(reset=False):
    from huggingface_hub import HfApi
    if LOCK.exists() and not reset:
        obj=json.loads(LOCK.read_text())
        assert all(k in obj['revisions'] for k in REPOS),'Incomplete local revision lock'
        return obj['revisions']
    api=HfApi(); out={}
    for name,(kind,repo) in REPOS.items():
        out[name]=api.repo_info(repo,repo_type=kind).sha
        if not out[name] or len(out[name])<40:raise RuntimeError(f'Could not resolve {repo} commit')
    LOCK.write_text(json.dumps({'repos':{k:v[1] for k,v in REPOS.items()},'revisions':out},indent=2)+'\n')
    print('Locked upstream revisions in data/local_source_revisions.json')
    return out

def corpus_windows(rows,tok,source,quota=50000):
    scored=0
    for idx,raw in rows:
        ids=tok(raw,add_special_tokens=False)['input_ids']
        for start in range(0,len(ids),512):
            chunk=ids[start:start+512]
            if len(chunk)<2:continue
            n=min(len(chunk)-1,quota-scored)
            if n<=0:return
            yield dict(source=source,source_row=idx,token_offset=start,
                token_ids=chunk[:n+1],score_target_offsets=[1,n+1],scored_tokens=n,
                raw_text_sha256=hash_text(raw))
            scored+=n
            if scored==quota:return
    if scored!=quota:raise RuntimeError(f'{source}: only {scored}/{quota} scored tokens')

def public_assets(rev,overwrite=False):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    PUBLIC.mkdir(exist_ok=True,parents=True)
    tok=AutoTokenizer.from_pretrained(REPOS['model_base'][1],revision=rev['model_base'],use_fast=True)
    def needed(file):return overwrite or not (PUBLIC/file).exists()
    # Only select source rows in the fixed index order; never replace these with new random draws.
    for tag,spec,split,name in [
        ('hellaswag','default','validation','hellaswag_val'),
        ('gsm8k','main','train','gsm8k_train'),
        ('gsm8k','main','test','gsm8k_test')]:
        fn=f'{name}_locked.jsonl.gz'
        if not needed(fn):continue
        selected=read(MAN/f'{name}_ids.jsonl')
        ds=load_dataset(REPOS[tag][1],spec,split=split,revision=rev[tag])
        if max(x['row_index'] for x in selected)>=len(ds):raise RuntimeError(f'Invalid {name} indices')
        save_lines(PUBLIC/fn, (dict(order=x['order'],source_row_index=x['row_index'],
            example=dict(ds[x['row_index']]),source_sha256=hash_text(json.dumps(dict(ds[x['row_index']]),sort_keys=True,default=str))) for x in selected))
    if needed('c4_calibration_locked.jsonl.gz') or needed('c4_development_locked.jsonl.gz'):
        candidates={int(x['row_index']) for x in read(MAN/'c4_candidate_rows.jsonl')}
        last=max(candidates); collected=[]
        stream=load_dataset(REPOS['c4'][1],'en',split='train',streaming=True,revision=rev['c4'])
        for idx,row in enumerate(stream):
            if idx not in candidates:continue
            ids=tok(row['text'],add_special_tokens=False)['input_ids']
            if len(ids)>=512:
                collected.append(dict(source_row=idx,token_offset=0,token_ids=ids[:512],raw_text_sha256=hash_text(row['text'])))
                if len(collected)==160:break
            if idx>=last:break
        if len(collected)<160:raise RuntimeError(f'Only {len(collected)}/160 eligible C4 windows; check source revision and tokenizer')
        save_lines(PUBLIC/'c4_calibration_locked.jsonl.gz',collected[:128])
        save_lines(PUBLIC/'c4_development_locked.jsonl.gz',collected[128:])
    if needed('c4_validation_windows_locked.jsonl.gz'):
        stream=load_dataset(REPOS['c4'][1],'en',split='validation',streaming=True,revision=rev['c4'])
        save_lines(PUBLIC/'c4_validation_windows_locked.jsonl.gz',corpus_windows(((i,row['text']) for i,row in enumerate(stream)),tok,'c4_validation'))
    if needed('wikitext2_test_windows_locked.jsonl.gz'):
        ds=load_dataset(REPOS['wikitext'][1],'wikitext-2-raw-v1',split='test',revision=rev['wikitext'])
        save_lines(PUBLIC/'wikitext2_test_windows_locked.jsonl.gz',corpus_windows(((i,x['text']) for i,x in enumerate(ds)),tok,'wikitext2_test'))
    print('Public data available locally. Preserve data/local_source_revisions.json for reproducibility.')

def prefix_banks(rev,overwrite=False,limit=None):
    """Fixed teacher and original-draft prefix banks; speculative prefixes are generated by student decoders.

    This file intentionally does NOT implement speculative decoding.
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not torch.cuda.is_available():raise RuntimeError('Run --prefix-banks on Colab T4 or another CUDA GPU')
    BANK.mkdir(exist_ok=True,parents=True)
    dest=BANK/'prefix_banks_locked.jsonl.gz'
    if dest.exists() and not overwrite:
        print('Prefix banks already exist; use --force to rebuild');return
    tok=AutoTokenizer.from_pretrained(REPOS['model_target'][1],revision=rev['model_target'])
    draft_tok=AutoTokenizer.from_pretrained(REPOS['model_draft'][1],revision=rev['model_draft'])
    for s in ['Example 123','Hello there','A complicated example with 𝜋']:
        if tok.encode(s)!=draft_tok.encode(s):raise RuntimeError('Target/draft tokenizers differ')
    gsm=load_dataset(REPOS['gsm8k'][1],'main',split='test',revision=rev['gsm8k'])
    ms=read(MAN/'gsm8k_test_ids.jsonl')
    prompts=[(f'GSM8K-{r["order"]:04d}',gsm[r['row_index']]['question'],'math') for r in ms]
    prompts.extend((x['id'],x['input'],'short') for x in read(ROOT/'data/short_form/short_test.jsonl'))
    if limit is not None:prompts=prompts[:limit]
    output=[]
    for which,key in [('teacher','model_target'),('original_draft','model_draft')]:
        print('Loading',which,flush=True)
        model=AutoModelForCausalLM.from_pretrained(REPOS[key][1],revision=rev[key],torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
        for j,(pid,prompt,domain) in enumerate(prompts):
            ids=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,return_tensors='pt').to('cuda')
            gen=torch.Generator(device='cuda').manual_seed(6304+10007*j+(0 if which=='teacher' else 123))
            with torch.inference_mode():
                seq=model.generate(ids,max_new_tokens=192,do_sample=True,temperature=1.0,top_p=1.0,top_k=0,
                     generator=gen,pad_token_id=tok.eos_token_id)[0,ids.shape[1]:].tolist()
            rng=random.Random(f'PA4-banks-v1:6304:{pid}:{which}')
            positions=[]
            for lo,hi,n in [(1,16,8),(17,32,8),(33,64,16),(65,192,16)]:
                candidates=list(range(lo,min(hi,len(seq))+1))
                positions.extend(sorted(rng.sample(candidates,min(n,len(candidates)))))
            for pos in positions:
                output.append(dict(prompt_id=pid,domain=domain,bank=which,position=pos,prompt=prompt,
                    response_prefix_token_ids=seq[:pos-1],source_length=len(seq),source_generation_seed=6304+10007*j+(0 if which=='teacher' else 123)))
            if (j+1)%20==0:print(which,j+1,'/',len(prompts),flush=True)
        del model
        import gc;gc.collect();torch.cuda.empty_cache()
    save_lines(dest,output)
    print('Model-generated prefixes are fixed for this local lock. Students generate speculative-policy prefixes with their own sampler.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--public',action='store_true');g.add_argument('--prefix-banks',action='store_true');g.add_argument('--all',action='store_true')
    p.add_argument('--force',action='store_true',help='Rebuild existing generated assets')
    p.add_argument('--refresh-revisions',action='store_true',help='Resolve new upstream commits and update the local lock')
    p.add_argument('--limit-prefix-prompts',type=int,help='Smoke test only; do not use for final evaluation')
    args=p.parse_args()
    rev=revisions(args.refresh_revisions)
    if args.public or args.all:public_assets(rev,args.force)
    if args.prefix_banks or args.all:prefix_banks(rev,args.force,args.limit_prefix_prompts)
if __name__=='__main__':main()
