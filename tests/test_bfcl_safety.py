import unittest
from attncopy.bfcl_safety import native_leaf,native_candidates


class NativeSafetyTests(unittest.TestCase):
    def test_argument_offsets_and_wrappers(self):
        text='<think>Reason about <tool_call> examples.</think>\n<tool_call>\n{"name":"f","arguments":{"x":"abcd'
        leaf=native_leaf(text)
        self.assertEqual(leaf.path,(0,'arguments','x'))
        self.assertEqual(text[leaf.start:],'abcd')
        done='<tool_call>\n{"name":"a","arguments":{}}\n</tool_call>\n'
        leaf=native_leaf(done+'<tool_call>\n{"name":"b","arguments":{"x":"abcd')
        self.assertEqual(leaf.path,(1,'arguments','x'))

    def test_uncertain_fallback(self):
        for text in ['<think><tool_call>\n{"arguments":{"x":"abcd',
                     'prose <tool_call>\n{"arguments":{"x":"abcd',
                     '<tool_call>{"arguments":{"x":"abcd',
                     '<tool_call>\n{"name":"abcd',
                     '<tool_call>\n{"arguments":{"abcd',
                     '<tool_call>\n{"arguments":["abcd',
                     '<tool_call>\n{"arguments":{"x":"<tool_call>abcd']:
            self.assertIsNone(native_leaf(text))

    def test_old_alias_ids_preserved_and_new_boundary_verified(self):
        class Tok:
            all_special_ids=[]
            def __call__(self,text,**kwargs):return {'input_ids':list(map(ord,text)),'offset_mapping':[(i,i+1) for i in range(len(text))]}
            def decode(self,ids,**kwargs):return ''.join('f' if i==999 else chr(i) for i in ids)
        text='<tool_call>\n{"name":"f","arguments":{"x":"abc'
        ids=list(map(ord,text));ids[text.index('f"')]=999
        query='abcdef'
        candidates=native_candidates(query,native_leaf(text),Tok(),ids,[(i,i+1) for i in range(6)],0,8)
        self.assertEqual(candidates[0]['suffix'],'def')
        self.assertEqual(Tok().decode(ids+candidates[0]['draft']),text+'def')


if __name__=='__main__':
    unittest.main()
