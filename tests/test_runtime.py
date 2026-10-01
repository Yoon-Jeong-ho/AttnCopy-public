from attncopy.paths import model_path
import unittest
import torch
from transformers import AutoTokenizer, DynamicCache
from attncopy.engine import Engine
from attncopy.safety import verified_prefix

MODEL = model_path()


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine.__new__(Engine)
        cls.engine.tok = AutoTokenizer.from_pretrained(MODEL,local_files_only=True)

    def test_full_chat_alignment(self):
        tok = self.engine.tok
        tools = [{'name':'f','parameters':{'x':{'type':'str'}}}]
        for query in [' leading space', '\nnew line', 'quote "abc"', '한글 café🙂', '\\escapes']:
            rec = {'tools':tools,'query':query}
            ids, offsets, qoffset = self.engine.prompt(rec)
            text = tok.decode(ids, skip_special_tokens=False)
            # Canonical template rendering must reproduce the full prompt IDs.
            start = text.index('<|im_start|>user\n') + len('<|im_start|>user\n')
            self.assertEqual(text[start:start+len(query)],query)
            canonical = tok.apply_chat_template([{'role':'system','content':text.split('<|im_start|>system\n')[1].split('<|im_end|>')[0]},
                         {'role':'user','content':query}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            if isinstance(canonical, dict) or hasattr(canonical,'keys'):
                canonical = canonical['input_ids']
            self.assertEqual(ids,canonical)
            self.assertEqual(tok.encode(text,add_special_tokens=False),ids)
            self.assertLessEqual(offsets[0][0],0)
            self.assertGreaterEqual(offsets[-1][1],len(query))
            self.assertGreater(qoffset,0)

    def test_rejected_cache_crop(self):
        for draft, next_id, block_next, accepted in [([9,2,3],1,[2,3,4],0),([1,9,3],1,[2,3,4],1),([1,2,3],1,[2,3,4],3)]:
            cache = DynamicCache()
            keys = torch.arange(6.).reshape(1,1,6,1)
            cache.update(keys,keys.clone(),0)
            n, _ = verified_prefix(draft,next_id,block_next)
            self.assertEqual(n,accepted)
            cache.crop(3+n)
            self.assertEqual(cache.get_seq_length(),3+accepted)
            self.assertEqual(cache.layers[0].keys.flatten().tolist(),list(map(float,range(3+accepted))))


if __name__ == '__main__':
    unittest.main()
