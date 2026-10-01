import unittest
from scripts.analyze_bfcl import gold_string_flags,subgroup_tables,ARMS
from attncopy.argument_features import string_features


class AnalysisTests(unittest.TestCase):
    def test_accepted_values_missing_and_zero_bins(self):
        gold={'ground_truth':[{'f':{'text':['ABC','abc'],'optional':['x','']}}]}
        strata=[{'id':'case','path':['text'],'reference':'ABC','accepted':['ABC','abc'],'optional_missing':False,
                 'source_present_any_accepted':True,**string_features('ABC',1)},
                {'id':'case','path':['optional'],'reference':'x','accepted':['x',''],'optional_missing':True,
                 'source_present_any_accepted':False,**string_features('x',1)}]
        correct=gold_string_flags({},gold,{'parsed_calls':[{'f':{'text':'abc'}}]},strata)
        self.assertEqual(correct,{('text',):True,('optional',):True})
        wrong=gold_string_flags({},gold,{'parsed_calls':[]},strata)
        self.assertEqual(wrong,{('text',):False,('optional',):False})
        rows=[{'id':'case','arm':a,'string_flags':correct if a=='attention+match-forced' else wrong} for a in ARMS]
        table=subgroup_tables(strata,rows)
        self.assertEqual(table['token_length']['1-4']['string_leaf_n'],2)
        self.assertEqual(table['token_length']['1-4']['paired']['greedy']['wrong_to_correct'],2)
        self.assertEqual(table['token_length']['>=65']['string_leaf_n'],0)
        self.assertIsNone(table['token_length']['>=65']['em']['greedy'])
