import copy
import json
import unittest
from attncopy.bfcl_native import native_prompt,score_official,UPSTREAM,handler


class OfficialIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base=UPSTREAM/'berkeley-function-call-leaderboard/bfcl_eval/data'
        with (base/'BFCL_v4_simple_python.json').open() as f:cls.entry=json.loads(next(f))
        with (base/'possible_answer/BFCL_v4_simple_python.json').open() as f:cls.gold=json.loads(next(f))

    def test_native_prompt_preserves_input_and_offsets(self):
        original=copy.deepcopy(self.entry)
        prompt,sources=native_prompt(self.entry)
        self.assertEqual(self.entry,original)
        self.assertEqual(prompt,handler()._format_prompt(self.entry['question'][0],self.entry['function']))
        for source in sources:
            a=source['prompt_char_start']
            self.assertEqual(prompt[a:a+len(source['text'])],source['text'])
        self.assertTrue(prompt.endswith('<|im_start|>assistant\n'))

    def test_official_normalization_separate_from_raw_em(self):
        raw='<tool_call>\n{"name":"calculate_triangle_area","arguments":{"base":10,"height":5,"unit":"UNITS"}}\n</tool_call>'
        s=score_official(self.entry,self.gold,raw)
        self.assertTrue(s['official_ast_valid'])
        self.assertFalse(s['parsed_raw_value_call_em'])
        self.assertEqual((s['gold_argument_n'],s['correct_arguments']), (3,2))
        raw='<tool_call>\n{"name":"calculate_triangle_area","arguments":{"base":10,"height":5}}\n</tool_call>'
        s=score_official(self.entry,self.gold,raw)
        self.assertTrue(s['official_ast_valid'])
        self.assertTrue(s['parsed_raw_value_call_em'])

    def test_invalid_scalar_and_missing_keep_denominator(self):
        for raw in ['', '<tool_call>\n123\n</tool_call>','<tool_call>\n{bad}\n</tool_call>']:
            s=score_official(self.entry,self.gold,raw)
            self.assertFalse(s['official_ast_valid'])
            self.assertEqual((s['gold_argument_n'],s['correct_arguments']), (3,0))

    def test_official_reasoning_cleaner(self):
        fake='<tool_call>\n{"name":"wrong","arguments":{}}\n</tool_call>'
        call='<tool_call>\n{"name":"calculate_triangle_area","arguments":{"base":10,"height":5}}\n</tool_call>'
        s=score_official(self.entry,self.gold,'<think>\n'+fake+'\n</think>\n'+call)
        self.assertTrue(s['official_ast_valid'])
        self.assertEqual(len(s['parsed_calls']),1)


if __name__=='__main__':
    unittest.main()
