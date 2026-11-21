"""Validate schema, counts, split isolation, marked evidence and numeric MQAR invariants.

Validation of model accuracy and Qwen token lengths requires the separate STAFF GPU gate.
"""
import collections,json,sys
from pathlib import Path
from data_io import ROOT,records
from mqar_generator import validate as validate_mqar,make_example

def require(b,msg):
 if not b:raise AssertionError(msg)

def check():
 mqar=list(records(ROOT/'data/mqar/mqar_test.jsonl.gz'))
 require(len(mqar)==9216,f'MQAR count {len(mqar)} != 9216')
 n=collections.Counter();ids=set()
 for row in mqar:
  validate_mqar(row)
  require(row['id'] not in ids,'Duplicate MQAR id')
  ids.add(row['id']);n[(row['K'],row['regime'])]+=1
  # Frozen rows must exactly reproduce their stated deterministic seed/ID.
  index=int(row['id'].rsplit('-',1)[1]);
  require(row==make_example('test',row['K'],row['regime'],index),f'MQAR mismatch: {row["id"]}')
 require(set(n.values())=={1024} and len(n)==9, 'MQAR incomplete factorial grid')
 print('PASS MQAR:',len(mqar),'deterministic rows, 9 cells')
 del mqar
 cases=list(records(ROOT/'data/kv_retrieval/kv_retrieval_cases.jsonl.gz'))
 require(len(cases)==180,'KV retrieval requires 180 cases')
 ids=set();n=collections.Counter()
 for row in cases:
  require(row['id'] not in ids,'Duplicate KV ID');ids.add(row['id'])
  text=row['user_text'];phrase=row['evidence_text'];a=row['evidence_char_start'];b=row['evidence_char_end']
  require(text[a:b]==phrase,'Gold evidence character offsets incorrect')
  require(row['code'] in phrase and len(row['code'])==5 and row['code'].isdigit(),'Invalid gold code')
  require(text.endswith('Reply with only five digits.'),'Missing answer instruction')
  n[(row['target_tokens'],row['position'],row['distractors'])]+=1
 require(len(n)==18 and set(n.values())=={10},'KV grid not balanced')
 require({k[0] for k in n}=={2048,4096},'KV token target mismatch')
 print('PASS KV:',len(cases),'prompt cases, char spans (not tokenizer lengths)')
 train=list(records(ROOT/'data/short_form/short_train.jsonl'))
 test=list(records(ROOT/'data/short_form/short_test.jsonl'))
 require(len(train)==300 and len(test)==60,'Short-form cardinality mismatch')
 require(len({x['id'] for x in train+test})==360,'Short-form duplicate ID')
 require({x['split'] for x in train}=={'train'},'Short-form train label mismatch')
 require(collections.Counter(x['split'] for x in test)=={'familiar':30,'shifted':30},'Short test split mismatch')
 require({x['input'] for x in train}.isdisjoint(x['input'] for x in test),'Short prompt text overlaps')
 print('PASS short:',len(train),len(test))
 for name,count in [('hellaswag_val_ids',2000),('gsm8k_train_ids',500),('gsm8k_test_ids',60)]:
  arr=list(records(ROOT/f'data/public_manifests/{name}.jsonl'))
  require(len(arr)==count,f'{name} count wrong')
  require([x['order'] for x in arr]==list(range(count)),f'{name} unsorted order')
  require(len({x['row_index'] for x in arr})==count,f'{name} duplicate row')
  print('PASS',name,count)
 print('\nPASS: available local assets structurally valid.')
 print('NOT CHECKED: dataset revision hashes, locked token windows, prefix banks, model retrieval accuracy.')
if __name__=='__main__':check()
