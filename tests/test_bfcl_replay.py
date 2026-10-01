import unittest
import torch
from attncopy.bfcl_engine import BFCLEngine


class Cache:
    def __init__(self):self.tokens=[]
    def get_seq_length(self):return len(self.tokens)


class Fake(BFCLEngine):
    def forward(self,ids,cache,all_logits=False):
        cache.tokens.extend(ids)
        # Target sequence is 1,2,EOS3; this is a serial target cache oracle.
        next_id={1:2,2:3}[ids[0]]
        return torch.nn.functional.one_hot(torch.tensor([next_id]),5).float(),cache


class ReplayTests(unittest.TestCase):
    def test_zero_partial_full_acceptance_eos_and_budget(self):
        engine=Fake.__new__(Fake);engine.eos={3}
        logits=torch.nn.functional.one_hot(torch.tensor([1]),5).float()
        for draft,expected,accepted,forwarded in [([4,2],[1],0,1),([1,4],[1,2],1,2),([1,2],[1,2],2,2),([1,2,3],[1,2,3],3,2)]:
            cache=Cache();generated=[]
            _,_,a,f,_=engine.replay_verified(draft,logits,cache,generated,20)
            self.assertEqual((generated,a,f),(expected,accepted,forwarded))
            self.assertEqual(cache.get_seq_length(),forwarded)
        cache=Cache();generated=[]
        _,_,a,f,_=engine.replay_verified([1,2],logits,cache,generated,1)
        self.assertEqual((generated,a,f),([1],1,0))
