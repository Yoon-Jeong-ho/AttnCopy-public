"""Record pre-inference review amendment; preserve original frozen evidence."""
import json
import re
from pathlib import Path
from prepare import digest,dump,template,ROOT,RUN


def main():
    original = json.loads((RUN/'frozen.json').read_text())
    cfg = ROOT/'configs/pilot.json'
    if (RUN/'frozen-v2.json').exists():
        raise RuntimeError('review amendment already frozen')
    dump(RUN/'config.initial.json',json.loads(cfg.read_text()))
    config = json.loads(cfg.read_text())
    config.update(dtype='bfloat16', dtype_rationale='original model config torch_dtype; no FP32 production inference claim',
                  tokenization='canonical full chat encoding with full-prompt character offsets',
                  greedy='unhooked SDPA; hooks attached only in attention arms')
    dump(cfg,config)
    rows = [[json.loads(l) for l in (RUN/f'{s}.jsonl').open()] for s in ['dev','test']]
    schemas = [{template({'tools':[t]}) for r in pool for t in r['tools']} for pool in rows]
    queries = [{re.sub(r'\s+',' ',r['query'].casefold()).strip() for r in pool} for pool in rows]
    assert not schemas[0]&schemas[1] and not queries[0]&queries[1]
    amendment = {**original,'parent_frozen_sha256':digest(RUN/'frozen.json'),'config_sha256':digest(cfg),
                 'review_sha256':digest(ROOT/'docs/SUPERVISOR_REVIEW.md'),
                 'schema_audit':{'individual_function_groups':[len(s) for s in schemas],'overlap':0,'normalized_query_overlap':0},
                 'amendment':'pre-GPU review; IDs unchanged; BF16/full chat/unhooked greedy/scalar denominator/call identity fixes',
                 'grouping_limits':'dataset source + catalog and individual function/schema groups + exact normalized queries disjoint; latent natural-language template equivalence unknown'}
    dump(RUN/'frozen-v2.json',amendment)
    print(json.dumps(amendment['schema_audit']))


if __name__ == '__main__':
    main()
