"""Standard library tests (python -m unittest discover -s tests)."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from data_io import ROOT,records
from mqar_generator import make_example,validate
class DataTests(unittest.TestCase):
 def test_mqar_determinism(self):
  for k in (4,8,16):
   for regime in ('near','far','mixed'):
    ex=make_example('train',k,regime,123)
    self.assertEqual(ex,make_example('train',k,regime,123))
    validate(ex)
    self.assertNotEqual(ex['tokens'],make_example('test',k,regime,123)['tokens'])
 def test_retrieval_markers(self):
  row=next(records(ROOT/'data/kv_retrieval/kv_retrieval_cases.jsonl.gz'))
  self.assertEqual(row['user_text'][row['evidence_char_start']:row['evidence_char_end']],row['evidence_text'])
 def test_short_test_disjoint(self):
  train={x['input'] for x in records(ROOT/'data/short_form/short_train.jsonl')}
  test={x['input'] for x in records(ROOT/'data/short_form/short_test.jsonl')}
  self.assertTrue(train.isdisjoint(test))
if __name__=='__main__':unittest.main()
