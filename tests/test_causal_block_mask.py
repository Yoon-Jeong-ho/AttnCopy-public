from attncopy.paths import model_path
import unittest
import torch
from transformers import AutoConfig,DynamicCache
from transformers.masking_utils import create_causal_mask


class CausalBlockMaskTests(unittest.TestCase):
    def test_cached_block_cannot_attend_future_tokens(self):
        cfg=AutoConfig.from_pretrained(model_path(),local_files_only=True)
        cfg._attn_implementation='sdpa'
        cache=DynamicCache(config=cfg)
        keys=torch.zeros(1,8,683,128,dtype=torch.bfloat16)
        for layer in range(28):cache.update(keys,keys.clone(),layer)
        mask=create_causal_mask(config=cfg,inputs_embeds=torch.zeros(1,2,1024,dtype=torch.bfloat16),
              attention_mask=None,past_key_values=cache,position_ids=torch.tensor([[683,684]]))
        self.assertEqual(tuple(mask.shape),(1,1,2,685))
        self.assertTrue(bool(mask[0,0,0,:684].all()))
        self.assertFalse(bool(mask[0,0,0,684]))
        self.assertTrue(bool(mask[0,0,1].all()))
