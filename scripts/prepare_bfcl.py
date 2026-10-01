"""CPU official native inputs, full 658 IDs, context/strata/config freeze."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from attncopy.paths import model_path
from scripts.prepare import ROOT,RUN,digest,dump
from attncopy.bfcl_native import UPSTREAM,native_prompt,handler
from attncopy.argument_features import official_gold_leaves,string_features
from tokenizers import Tokenizer

OUT=ROOT/'runs/bfcl-658'


def native_fixture_rows(tok):
    queries=['Call echo with text "ABCD987654321abcdefghijk".','Call echo with text "한글café".',
             'Call echo with text "ABCDEFGH12345", not "ABCDEFGH99999".']
    result=[]
    for i,query in enumerate(queries):
        entry={'id':f'native-fixture-{i}','question':[[{'role':'user','content':query}]],
               'function':[{'name':'echo','description':'Return the exact supplied text.',
                  'parameters':{'type':'dict','properties':{'text':{'type':'string','description':'Text to return.'}},'required':['text']}}]}
        prompt,sources=native_prompt(entry)
        result.append({'id':entry['id'],'category':'fixture','prompt_text':prompt,'sources':sources,
                       'input_ids':tok.encode(prompt,add_special_tokens=False).ids,'max_new_tokens':4096,'context_rejected':False})
    return result


def read(path):
    with path.open() as f:return [json.loads(l) for l in f]


def main():
    OUT.mkdir(exist_ok=False)
    manifest=json.loads((ROOT/'docs/BFCL_SOURCE_MANIFEST.json').read_text())
    assert subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()==manifest['commit']
    for rel,entry in manifest['artifacts'].items():assert digest(UPSTREAM/rel)==entry['sha256'],rel
    base=UPSTREAM/'berkeley-function-call-leaderboard/bfcl_eval/data'
    config=json.loads((ROOT/'configs/pilot.json').read_text())
    selected=json.loads((RUN/'selection.json').read_text())
    config.update(max_new_tokens=4096,seed=42,backend='sdpa',dtype='bfloat16',
         native_registry='Qwen/Qwen3-0.6B-FC',native_prompt='unmodified QwenFCHandler._format_prompt',
         thinking_mode='native handler assistant prefix, no injected disable-thinking prefix; native default reasoning',
         generation_cap='official OSS min(4096, max_context - input_tokens - 2)',
         source_copy_scope='original user-message content only; function docs/reasoning/XML ineligible',
         threshold=selected['threshold'],attention_margin=selected['margin'],
         temperature=0,adapter='local HF greedy vs official server backend; official prompt/parser/AST, not an official leaderboard submission')
    dump(ROOT/'configs/bfcl-658.json',config)
    model=Path(model_path(config))
    tok=Tokenizer.from_file(str(model/'tokenizer.json'))
    tok.no_truncation();tok.no_padding()
    max_context=json.loads((model/'config.json').read_text())['max_position_embeddings']
    counts={};input_stats={};source_ids={};over_limit=[];gold_strata=[]
    runtime=[]
    for category,n in [('simple_python',400),('live_simple',258)]:
        entries=read(base/f'BFCL_v4_{category}.json')
        gold=read(base/'possible_answer'/f'BFCL_v4_{category}.json')
        assert len(entries)==len(gold)==n and [e['id'] for e in entries]==[g['id'] for g in gold]
        counts[category]=n;source_ids[category]=[e['id'] for e in entries]
        lengths=[]
        for index,(entry,answer) in enumerate(zip(entries,gold)):
            prompt,sources=native_prompt(entry)
            encoding=tok.encode(prompt,add_special_tokens=False)
            length=len(encoding.ids);lengths.append(length)
            rejected=length+2>=max_context
            if rejected:over_limit.append(entry['id'])
            # Runtime input contains no gold answers or gold-side strata.
            runtime.append({'id':entry['id'],'category':category,'case_index':len(runtime),
                'category_index':index,'prompt_text':prompt,'input_ids':encoding.ids,'sources':sources,
                'max_new_tokens':min(4096,max_context-length-2) if not rejected else 0,
                'context_rejected':rejected,'input_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
            user_texts=[s['text'] for s in sources]
            for leaf in official_gold_leaves(entry,answer):
                if leaf.get('undefined_gold_reference'):
                    gold_strata.append({'id':entry['id'],'category':category,**leaf,'non_string':True,'undefined_gold_reference':True})
                    continue
                value=leaf['reference']
                if isinstance(value,str):
                    features=string_features(value,len(tok.encode(value,add_special_tokens=False).ids))
                    eligible=any(isinstance(v,str) and v!='' and any(v in text for text in user_texts) for v in leaf['accepted'])
                    safe_value=bool(value) and all(32<ord(c)<127 and not c.isspace() and c not in '"\\<>' for c in value)
                    gold_strata.append({'id':entry['id'],'category':category,**leaf,**features,
                                        'source_present_any_accepted':eligible,'ascii_nonwhitespace_safe_reference':safe_value})
                else:gold_strata.append({'id':entry['id'],'category':category,**leaf,'non_string':True})
        ordered=sorted(lengths)
        input_stats[category]={'n':n,'min':ordered[0],'p50':ordered[int(.5*(n-1))],'p95':ordered[int(.95*(n-1))],
                               'p99':ordered[int(.99*(n-1))],'max':ordered[-1],'total':sum(lengths)}
    assert len(runtime)==658 and len({r['id'] for r in runtime})==658
    (OUT/'runtime-inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in runtime))
    (OUT/'gold-strata.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in gold_strata))
    (OUT/'native-fixtures.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in native_fixture_rows(tok)))
    code={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'attncopy/bfcl_native.py',ROOT/'attncopy/bfcl_safety.py',ROOT/'attncopy/argument_features.py',Path(__file__)]}
    model_hash=json.loads((RUN/'frozen-v2.json').read_text())['source_model_sha256']
    frozen={'counts':counts,'source_ids':source_ids,'config_sha256':digest(ROOT/'configs/bfcl-658.json'),
        'source_manifest_sha256':digest(ROOT/'docs/BFCL_SOURCE_MANIFEST.json'),
        'runtime_sha256':digest(OUT/'runtime-inputs.jsonl'),'strata_sha256':digest(OUT/'gold-strata.jsonl'),
        'input_token_distribution':input_stats,'context_limit':max_context,'context_rejections':over_limit,
        'empty_gold_alternative_parameters':sum(bool(s.get('undefined_gold_reference')) for s in gold_strata),
        'empty_gold_alternative_cases':len({s['id'] for s in gold_strata if s.get('undefined_gold_reference')}),
        'selection_sha256':digest(RUN/'selection.json'),'selection':selected,'code_sha256':code,
        'model_tokenizer_sha256':{p:h for p,h in model_hash.items() if p.startswith(str(model))},
        'strata_definition':'pilot ARGUMENT_SUBGROUP_PLAN features; first non-empty official alternative defines features; all accepted alternatives retained for scoring; no model outputs inspected',
        'strata_plan_sha256':digest(ROOT/'docs/ARGUMENT_SUBGROUP_PLAN.md'),
        'command':sys.argv,'official_handler_constructor':'pure __new__ methods, no client/server/API setup',
        'adapter_limits':'full official category data + native prompt/parser/AST; local greedy SDPA replaces official server/Completions generation'}
    dump(OUT/'frozen.json',frozen)
    print(json.dumps({'counts':counts,'input_tokens':input_stats,'context_limit':max_context,'rejected':len(over_limit),'gold_leaves':len(gold_strata)}))


if __name__=='__main__':main()
