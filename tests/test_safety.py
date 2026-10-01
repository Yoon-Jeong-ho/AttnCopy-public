import unittest
from attncopy.safety import current_leaf, source_candidates, choose, verified_prefix
from attncopy.metrics import score


class CharTokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


class SafetyTests(unittest.TestCase):
    def test_json_paths(self):
        self.assertIsNone(current_leaf('[{"name":"abcd'))
        self.assertIsNone(current_leaf('[{"arguments":{"abcd'))
        self.assertIsNone(current_leaf('{"arguments":{"x":"abcd'))
        self.assertIsNone(current_leaf('[{"name":"f","name":"g","arguments":{"x":"abcd'))
        self.assertIsNone(current_leaf('[{"arguments":{"a":1 "b":"abcd'))
        leaf = current_leaf('[{"name":"f","arguments":{"nested":[{"x":"abcd')
        self.assertEqual(leaf.path, (0, 'arguments', 'nested', 0, 'x'))
        self.assertEqual(leaf.prefix, 'abcd')
        wrapped = '```json\n[{"name":"f","arguments":{"x":"abcd'
        leaf = current_leaf(wrapped)
        self.assertEqual(wrapped[leaf.start:], 'abcd')
        self.assertIsNone(current_leaf('some prose\n'+wrapped))

    def proposals(self, query, prefix='abc', budget=100):
        raw = '[{"name":"f","arguments":{"x":"' + prefix
        leaf = current_leaf(raw)
        tok = CharTokenizer()
        return source_candidates(query, leaf, tok, tok.encode(raw), [(i, i+1) for i in range(len(query))], 7, budget)

    def test_bounds_offsets(self):
        c = self.proposals('한 abcdefghijklmnop rest')[0]
        self.assertEqual(c['char_start'], 2)
        self.assertEqual(c['byte_start'], 4)
        self.assertEqual(c['suffix'], 'defghijk')
        self.assertEqual(len(c['draft']), 8)
        self.assertEqual(self.proposals('abcdef gh')[0]['suffix'], 'def')
        self.assertEqual(len(self.proposals('abcdef', budget=1)[0]['draft']), 1)

    def test_boundaries_fallback(self):
        for s in ['abc"tail', 'abc\\tail', 'abc tail', 'abc\ttail', 'abc한글']:
            self.assertEqual(self.proposals(s), [])
        self.assertEqual(self.proposals('abcdef', prefix='ab'), [])
        self.assertEqual(self.proposals('abc\\ndef', prefix='abc\\n'), [])
        self.assertEqual(self.proposals('한글abcd', prefix='한글a'), [])

    def test_duplicates(self):
        self.assertIsNone(choose(self.proposals('abcdef abcXYZ')))
        self.assertIsNotNone(choose(self.proposals('abcdef abcdef')))
        cs = self.proposals('abcdef abcXYZ')
        row = [0.] * 100
        for j in cs[1]['token_span']:
            row[j] = .1
        self.assertEqual(choose(cs, row=ListRow(row), threshold=.2, margin=.01)['suffix'], 'XYZ')
        self.assertIsNone(choose(cs, row=ListRow([0.]*100), threshold=0., margin=.01))

    def test_pld_shift_rejection_eos_budget(self):
        self.assertEqual(verified_prefix([1,2,3], 1, [2,3,4]), (3,None))
        self.assertEqual(verified_prefix([9,2,3], 1, [2,3,4]), (0,1))
        self.assertEqual(verified_prefix([1,9,3], 1, [2,3,4]), (1,2))
        self.assertEqual(verified_prefix([1,2,3], 1, [2,3,4], {2}), (2,None))
        self.assertEqual(verified_prefix([1,9], 1, [2,3], {9}), (1,2))
        self.assertEqual(verified_prefix([1], 1, [2]), (1,None))

    def test_metrics_denominators_types(self):
        rec = {'answers':[{'name':'f','arguments':{'s':'abc','n':3,'b':True}}]}
        res = {'raw':'[{"name":"f","arguments":{"s":"abc","n":3.0,"b":1}}]', 'events':[], 'stop':'eos'}
        s = score(rec,res)
        self.assertEqual((s['n_args'],s['correct_args'],s['full_call_em']), (3,1,False))
        res['raw'] = '{bad'
        s = score(rec,res)
        self.assertEqual((s['n_args'],s['correct_args'],s['valid_json']), (3,0,False))
        res['raw'] = '[{"name":"f","arguments":{"s":"x","s":"abc"}}]'
        self.assertFalse(score(rec,res)['valid_json'])

    def test_scalar_json_denominators(self):
        rec = {'answers':[{'name':'f','arguments':{'s':'abc'}}]}
        for raw in ['123', 'null', '{}', '"abc"', 'true', '[]', '[1]']:
            res = {'raw':raw, 'events':[], 'stop':'eos'}
            s = score(rec,res)
            self.assertEqual((s['n_args'],s['correct_args'],s['full_call_em']), (1,0,False))
            self.assertTrue(s['valid_json'])

    def test_call_order_gold_support(self):
        rec = {'answers':[{'name':'a','arguments':{'s':'abcdef'}}, {'name':'b','arguments':{'s':'XYZ'}}]}
        event = {'path':[1,'arguments','s'], 'prefix':'abc','suffix':'def'}
        res = {'raw':'[{"name":"b","arguments":{"s":"XYZ"}},{"name":"a","arguments":{"s":"abcdef"}}]',
               'events':[event], 'stop':'eos'}
        self.assertTrue(score(rec,res)['full_call_em'])
        self.assertTrue(event['gold_value_prefix_supported'])
        rec['answers'].append({'name':'a','arguments':{'s':'abcdxx'}})
        score(rec,res)
        self.assertTrue(event['gold_support_ambiguous'])
        self.assertIsNone(event['gold_value_prefix_supported'])
        rec['answers'].pop()
        res['raw'] = '```json\n'+res['raw']+'\n```'
        self.assertFalse(score(rec,res)['valid_json'])
        self.assertTrue(event['gold_value_prefix_supported'])
        self.assertEqual(event['gold_support_parse'],'recognized_fenced_json_diagnostic')

    def test_contextual_merge_fallback(self):
        class Merge(CharTokenizer):
            def encode(self,text,**kwargs):
                return list(map(ord,text.replace('abcdef', '\uffff')))
        raw = '[{"arguments":{"x":"abc'
        self.assertEqual(source_candidates('abcdef',current_leaf(raw),Merge(),list(map(ord,raw)),
                                          [(i,i+1) for i in range(6)],0,8), [])


class ListRow:
    def __init__(self, row):
        self.row = row

    def __getitem__(self, ids):
        return SumList(self.row[i] for i in ids)


class SumList(list):
    def sum(self):
        return sum(self)


if __name__ == '__main__':
    unittest.main()
