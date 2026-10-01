import itertools
import unittest
from attncopy.scoring_review import max_assignment,reviewed_score
import json


class ScoringReviewTests(unittest.TestCase):
    def test_assignment_against_exhaustive_oracle(self):
        for weights in [[[9,8],[8,0]],[[1,4,2],[3,0,2],[0,5,1]],[[0]]]:
            got = max_assignment(weights)
            score = sum(weights[i][j] for i,j in got.items())
            expected = max(sum(weights[i][j] for i,j in enumerate(order)) for order in itertools.permutations(range(len(weights))))
            self.assertEqual(score,expected)

    def test_duplicate_name_order_invariance(self):
        gold = [{'name':'f','arguments':{'a':1,'b':1}}, {'name':'f','arguments':{'a':2,'b':2}}]
        pred = [{'name':'f','arguments':{'a':1,'b':2}}, {'name':'f','arguments':{'a':1,'b':0}}]
        for gs in [gold,list(reversed(gold))]:
            for ps in [pred,list(reversed(pred))]:
                s = reviewed_score({'answers':gs},{'raw':json.dumps(ps),'events':[],'stop':'eos'})
                self.assertEqual((s['correct_args'],s['n_args']), (2,4))


if __name__ == '__main__':
    unittest.main()
