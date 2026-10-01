from attncopy.paths import model_path
import json
from pathlib import Path
import unittest
from transformers import AutoTokenizer
from attncopy.bfcl_engine import BFCLEngine


class FrozenInputTests(unittest.TestCase):
    def test_all_canonical_inputs_and_source_alignment(self):
        engine=BFCLEngine.__new__(BFCLEngine)
        engine.tok=AutoTokenizer.from_pretrained(model_path(),local_files_only=True)
        path=Path('runs/bfcl-658/runtime-inputs.jsonl')
        with path.open() as f:rows=[json.loads(l) for l in f]
        self.assertEqual(len(rows),658)
        for r in rows:
            ids,_,_=engine.prompt(r)
            self.assertEqual(ids,r['input_ids'])
            self.assertEqual(engine.tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),r['prompt_text'])
            self.assertFalse(r['context_rejected'])
            self.assertLess(len(ids)+r['max_new_tokens'],40960)
