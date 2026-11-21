"""Inspect counts and example field names, without loading entire files into RAM."""
from data_io import ROOT,records
FILES={
 'mqar_test':'data/mqar/mqar_test.jsonl.gz',
 'kv_retrieval':'data/kv_retrieval/kv_retrieval_cases.jsonl.gz',
 'short_train':'data/short_form/short_train.jsonl',
 'short_test':'data/short_form/short_test.jsonl',
 'hellaswag_ids':'data/public_manifests/hellaswag_val_ids.jsonl',
 'gsm8k_train_ids':'data/public_manifests/gsm8k_train_ids.jsonl',
 'gsm8k_test_ids':'data/public_manifests/gsm8k_test_ids.jsonl',
}
for name,p in FILES.items():
 count=0;first=None
 for x in records(ROOT/p):
  count+=1
  if first is None:first=x
 print(f'{name:20} count={count:6} keys={list(first) if first else []}')
